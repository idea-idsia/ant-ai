from __future__ import annotations

import json
import os

import pytest
from rich.console import Console

from ant_ai.cli import Session, config, history, main
from ant_ai.cli.render import PLAN_TOOL, Renderer, main_argument, result_summary
from ant_ai.cli.tools import coding_tools
from ant_ai.core.events import ToolCallingEvent, ToolResultEvent
from ant_ai.core.message import Message, ToolCall, ToolCallMessage, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.tool import ToolError


class ScriptedLLM(ChatLLM):
    """Returns the scripted responses in order."""

    model = "scripted"
    api_key = None
    api_base = None

    def __init__(self, *responses: ChatLLMResponse):
        self.responses = list(responses)
        self.seen: list[list] = []

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def answer(text: str) -> ChatLLMResponse:
    return ChatLLMResponse(message=Message(role="assistant", content=text))


def call(why: str, name: str, call_id: str = "c1", **args) -> ChatLLMResponse:
    return ChatLLMResponse(
        message=Message(role="assistant", content=why),
        tool_calls=[
            ToolCall(
                id=call_id, function=ToolFunction(name=name, arguments=json.dumps(args))
            )
        ],
    )


def recording_session(llm: ChatLLM, workspace, **kwargs) -> Session:
    console = Console(record=True, width=100, force_terminal=False)
    return Session(llm, workspace, renderer=Renderer(console), **kwargs)


@pytest.mark.unit
async def test_a_turn_shows_the_reason_the_call_its_result_and_the_answer(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')\nprint('bye')\n")
    llm = ScriptedLLM(
        call("I read main.py to count its lines.", "read_file", path="main.py"),
        answer("main.py has **2** lines."),
    )
    session = recording_session(llm, tmp_path)

    await session.ask("how many lines in main.py?")

    out = session.renderer.console.export_text()
    assert "I read main.py to count its lines." in out
    assert (
        "● I read main.py to count its lines.\n  ⎿ read_file  main.py  2 lines" in out
    )
    assert "main.py has 2 lines." in out
    assert "2 steps · 1 tool" in out


@pytest.mark.unit
async def test_history_is_kept_between_turns_and_cleared_on_request(tmp_path):
    llm = ScriptedLLM(answer("one"), answer("two"), answer("three"))
    session = recording_session(llm, tmp_path)

    await session.ask("first")
    await session.ask("second")
    assert [m.content for m in llm.seen[1][1:]] == ["first", "one", "second"]

    session.clear()
    await session.ask("third")
    assert [m.content for m in llm.seen[2][1:]] == ["third"]


@pytest.mark.unit
async def test_writing_needs_the_users_ok(tmp_path):
    asked: list[str] = []

    async def refuse(action: str) -> bool:
        asked.append(action)
        return False

    tools = {t.name: t for t in coding_tools(tmp_path, confirm=refuse)}
    with pytest.raises(ToolError, match="did not allow"):
        await tools["write_file"].ainvoke(path="a.txt", content="x\ny")
    with pytest.raises(ToolError, match="did not allow"):
        await tools["run_command"].ainvoke(command="touch b.txt")

    assert asked == ["write a.txt (2 lines)", "run `touch b.txt`"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
async def test_a_refused_write_reaches_the_model_as_an_error(tmp_path):
    llm = ScriptedLLM(
        call("I create the file.", "write_file", path="a.txt", content="x"),
        answer("You did not allow the write."),
    )

    async def say_no(prompt) -> str:
        return "n"

    session = recording_session(llm, tmp_path, read_answer=say_no)
    await session.ask("create a.txt")

    out = session.renderer.console.export_text()
    assert "⎿ write_file  a.txt  ✗ The user did not allow: write a.txt (1 lines)" in out
    assert not (tmp_path / "a.txt").exists()
    # The model is told, and answers from there.
    assert "did not allow" in llm.seen[1][-1].content


@pytest.mark.unit
async def test_always_allows_the_rest_of_the_session(tmp_path):
    llm = ScriptedLLM(
        call("First file.", "write_file", "c1", path="a.txt", content="x"),
        call("Second file.", "write_file", "c2", path="b.txt", content="y"),
        answer("done"),
    )
    answers = iter(["a"])

    async def always(prompt) -> str:
        return next(answers)

    session = recording_session(llm, tmp_path, read_answer=always)
    await session.ask("create two files")

    assert (tmp_path / "a.txt").read_text() == "x" and (
        tmp_path / "b.txt"
    ).read_text() == "y"


@pytest.mark.unit
async def test_always_for_writes_still_asks_before_a_command(tmp_path):
    llm = ScriptedLLM(
        call("First file.", "write_file", "c1", path="a.txt", content="x"),
        call("I list.", "run_command", "c2", command="ls"),
        answer("done"),
    )
    asked: list[str] = []

    async def always_then_no(prompt) -> str:
        asked.append(prompt)
        return "a" if len(asked) == 1 else "n"

    session = recording_session(llm, tmp_path, read_answer=always_then_no)
    await session.ask("create a file and list")

    assert asked == ["allow write a.txt (1 lines)? y/n/a", "allow run `ls`? y/n/a"]


@pytest.mark.unit
async def test_tools_stay_inside_the_workspace(tmp_path):
    async def allow(_: str) -> bool:
        return True

    tools = {t.name: t for t in coding_tools(tmp_path, confirm=allow)}
    with pytest.raises(ToolError):
        await tools["read_file"].ainvoke(path="../outside.txt")


@pytest.mark.unit
async def test_auto_approve_runs_commands_in_the_workspace(tmp_path):
    llm = ScriptedLLM(
        call("I list the files with ls.", "run_command", command="ls"),
        answer("done"),
    )
    (tmp_path / "x.txt").write_text("")
    session = recording_session(llm, tmp_path, auto_approve=True)

    await session.ask("list")

    out = session.renderer.console.export_text()
    assert "⎿ run_command  ls  exit 0 · x.txt" in out


@pytest.mark.unit
def test_main_without_a_model_explains_what_to_do(monkeypatch, capsys):
    monkeypatch.delenv("ANT_MODEL", raising=False)
    assert main([]) == 2


@pytest.mark.unit
async def test_the_plan_is_shown_as_a_checklist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    todos = [
        {"content": "read the code", "status": "completed"},
        {"content": "fix the bug", "status": "in_progress"},
        {"content": "run the tests"},
    ]
    llm = ScriptedLLM(call("I plan the work.", PLAN_TOOL, todos=todos), answer("ok"))
    session = recording_session(llm, tmp_path)

    await session.ask("fix it")

    out = session.renderer.console.export_text()
    assert (
        "⎿ plan\n      ✓ read the code\n      ▸ fix the bug\n      ○ run the tests"
        in out
    )


@pytest.mark.unit
async def test_a_conversation_is_saved_and_can_be_resumed(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    first = recording_session(
        ScriptedLLM(
            call("I read a.txt.", "read_file", path="a.txt"), answer("It says hello.")
        ),
        tmp_path,
    )
    await first.ask("what is in a.txt?")
    assert first.file.exists()

    llm = ScriptedLLM(answer("You asked about a.txt."))
    second = recording_session(llm, tmp_path)
    assert second.resume()
    await second.ask("what did I ask?")

    # The tool call comes back as a tool call, so the model gets a valid history.
    assert isinstance(llm.seen[0][2], ToolCallMessage)
    assert [m.content for m in llm.seen[0][1:]][-2:] == [
        "It says hello.",
        "what did I ask?",
    ]
    assert second.file == first.file


@pytest.mark.unit
async def test_clear_starts_a_new_file_and_resume_without_history_is_false(tmp_path):
    session = recording_session(ScriptedLLM(answer("one")), tmp_path)
    assert not session.resume()
    await session.ask("first")
    old = session.file
    session.clear()
    assert session.file != old and old.exists() and not session.file.exists()


@pytest.mark.unit
async def test_the_plan_is_saved_with_the_conversation_not_in_the_workspace(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)
    todos = [{"content": "fix the bug", "status": "in_progress"}]
    llm = ScriptedLLM(call("I plan.", PLAN_TOOL, todos=todos), answer("ok"))
    session = recording_session(llm, project)

    await session.ask("fix it")

    plan = session.file.parent / "plans" / "ant" / f"{session.ctx.session_id}.md"
    assert "fix the bug" in plan.read_text()
    assert not (tmp_path / ".plans").exists() and not (project / ".plans").exists()


@pytest.mark.unit
async def test_edit_file_changes_one_part_after_the_users_ok(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    asked: list[str] = []

    async def say_yes(question: str) -> str:
        asked.append(question)
        return "y"

    llm = ScriptedLLM(
        call(
            "I fix the operator.",
            "edit_file",
            path="calc.py",
            old_string="a - b",
            new_string="a + b",
        ),
        answer("fixed"),
    )
    session = recording_session(llm, tmp_path, read_answer=say_yes)
    await session.ask("fix add")

    assert asked == ["allow edit calc.py? y/n/a"]
    assert (tmp_path / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert (
        "⎿ edit_file  calc.py  Replaced 1 occurrence in calc.py"
        in session.renderer.console.export_text()
    )


@pytest.mark.unit
def test_settings_file_fills_the_environment_without_overriding_it(
    tmp_path, monkeypatch
):
    from ant_ai.cli import config

    env = tmp_path / "env"
    env.write_text(
        "# my settings\nANT_MODEL=openai/some-model\nANT_API_KEY='secret'\n\nANT_API_BASE=http://x\n"
    )
    env.chmod(0o600)
    monkeypatch.delenv("ANT_MODEL", raising=False)
    monkeypatch.delenv("ANT_API_KEY", raising=False)
    monkeypatch.setenv("ANT_API_BASE", "http://already-set")

    assert config.load_env(env) == []
    assert os.environ["ANT_MODEL"] == "openai/some-model"
    assert os.environ["ANT_API_KEY"] == "secret"
    assert os.environ["ANT_API_BASE"] == "http://already-set"

    env.chmod(0o644)
    assert "chmod 600" in config.load_env(env)[0]


@pytest.mark.unit
def test_the_banner_names_the_model_readably(tmp_path):
    from ant_ai.cli.render import model_label

    assert model_label("anthropic/claude-sonnet-5-5") == "claude-sonnet-5-5 · anthropic"
    assert model_label("gpt-x") == "gpt-x"
    renderer = Renderer(Console(record=True, width=100))
    renderer.banner("anthropic/claude-sonnet-5-5", "~/proj")
    assert "model claude-sonnet-5-5 · anthropic" in renderer.console.export_text()


@pytest.mark.unit
async def test_the_conversation_loop_runs_commands_and_messages(tmp_path, monkeypatch):
    from ant_ai.cli import prompt
    from ant_ai.cli.app import _repl

    lines = iter(
        ["", "/help", "/model", "/details", "hello", "/clear", "/resume", "/exit"]
    )

    async def typed(hint):
        return next(lines)

    monkeypatch.setattr(prompt, "read_message", typed)
    session = recording_session(ScriptedLLM(answer("hi there")), tmp_path)

    await _repl(session, "hint", session.renderer)

    out = session.renderer.console.export_text()
    assert "/resume   continue the last conversation" in out
    assert "model scripted" in out
    assert "details on" in out
    assert "hi there" in out
    assert "new conversation" in out
    assert "resumed: 2 messages" in out


@pytest.mark.unit
async def test_the_conversation_loop_ends_on_ctrl_d(tmp_path, monkeypatch):
    from ant_ai.cli import prompt
    from ant_ai.cli.app import _repl

    async def ctrl_d(hint):
        return None

    monkeypatch.setattr(prompt, "read_message", ctrl_d)
    session = recording_session(ScriptedLLM(), tmp_path)
    await _repl(session, "hint", session.renderer)


@pytest.mark.unit
def test_main_opens_the_full_screen_view(tmp_path, monkeypatch):
    from ant_ai.cli import tui

    runs: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(
        tui, "run", lambda command, workspace, **kw: runs.append((command, kw))
    )

    assert main(["--agent", "python my_agent.py", "-C", str(tmp_path)]) == 0
    assert main(["--tui", "-m", "openai/x", "-C", str(tmp_path), "--yes"]) == 0

    (agent_command, agent_kw), (serve_command, serve_kw) = runs
    assert agent_command == ["python", "my_agent.py"]
    assert agent_kw["max_steps"] is None
    assert serve_command[1:] == [
        "-m",
        "ant_ai.cli",
        "--serve",
        "-m",
        "openai/x",
        "-C",
        str(tmp_path),
        "--max-steps",
        "25",
        "--yes",
    ]
    assert serve_kw["max_steps"] == 25


@pytest.mark.unit
def test_main_runs_one_task_or_serves_the_agent(tmp_path, monkeypatch, capsys):
    from ant_ai.cli import serve
    from ant_ai.llm import integrations

    monkeypatch.setattr(
        integrations, "LiteLLMChat", lambda **kw: ScriptedLLM(answer("all done"))
    )
    assert main(["-m", "openai/x", "-C", str(tmp_path), "say", "hi"]) == 0
    assert "all done" in capsys.readouterr().out

    served: list[dict] = []
    monkeypatch.setattr(serve, "serve", lambda llm, workspace, **kw: served.append(kw))
    assert main(["-m", "openai/x", "-C", str(tmp_path), "--serve"]) == 0
    assert served == [{"max_steps": 25, "auto_approve": False}]


@pytest.mark.unit
def test_main_refuses_a_folder_that_does_not_exist(tmp_path):
    assert main(["-m", "openai/x", "-C", str(tmp_path / "nope")]) == 2


@pytest.mark.unit
def test_a_relative_folder_reaches_the_agent_as_an_absolute_path(tmp_path, monkeypatch):
    from ant_ai.cli import tui

    # The agent process starts inside the folder, where a relative path no longer holds.
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path)
    runs: list[list[str]] = []
    monkeypatch.setattr(
        tui, "run", lambda command, workspace, **kw: runs.append(command)
    )

    assert main(["--tui", "-m", "openai/x", "-C", "project"]) == 0

    (command,) = runs
    assert command[command.index("-C") + 1] == str(tmp_path / "project")


@pytest.mark.unit
def test_a_result_that_only_looks_like_a_command_gets_the_plain_summary():
    summary, ok = result_summary("query", '{"returncode": 0, "stdout": ["line"]}')
    assert ok and not summary.startswith("exit")


@pytest.mark.unit
def test_odd_arguments_an_empty_error_and_a_broken_plan_still_render():
    odd = ToolCall(id="c1", function=ToolFunction(name="t", arguments="[1, 2]"))
    assert main_argument(odd) == "[1, 2]"
    renderer = Renderer(Console(record=True, width=100, force_terminal=False))
    with renderer.turn():
        renderer.event(ToolCallingEvent(content="", tool_calls=(odd,)))
        renderer.event(
            ToolResultEvent(
                tool_call_id="c1", name="t", content="ERROR: ", is_error=True
            )
        )
        plan = ToolCall(
            id="p1",
            function=ToolFunction(
                name=PLAN_TOOL,
                arguments=json.dumps({"todos": [None, {"content": "fix it"}]}),
            ),
        )
        renderer.event(ToolCallingEvent(content="", tool_calls=(plan,)))
        renderer.event(ToolResultEvent(tool_call_id="p1", name=PLAN_TOOL, content="ok"))
    out = renderer.console.export_text()
    assert "✗ tool failed" in out and "fix it" in out


@pytest.mark.unit
def test_folders_whose_names_read_the_same_keep_their_own_conversations(tmp_path):
    (tmp_path / "a_b").mkdir()
    (tmp_path / "a-b").mkdir()
    assert history.folder(tmp_path / "a_b") != history.folder(tmp_path / "a-b")
    assert history.folder(tmp_path / "a_b") == history.folder(tmp_path / "x/../a_b")


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
def test_a_saved_conversation_can_be_read_only_by_its_owner(tmp_path):
    path = tmp_path / "s" / "c.json"
    history.save(path, [Message(role="user", content="secret")])
    assert path.stat().st_mode & 0o777 == 0o600
    assert history.load(path)[0].content == "secret"
    assert [p.name for p in path.parent.iterdir()] == ["c.json"]


@pytest.mark.unit
def test_a_line_without_a_name_is_skipped_and_the_rest_is_read(tmp_path, monkeypatch):
    env = tmp_path / "env"
    env.write_text("=typo\nANT_TEST_AFTER=yes\n")
    env.chmod(0o600)
    monkeypatch.delenv("ANT_TEST_AFTER", raising=False)
    warnings = config.load_env(env)
    assert os.environ["ANT_TEST_AFTER"] == "yes"
    assert len(warnings) == 1 and ":1:" in warnings[0]


@pytest.mark.unit
async def test_always_holds_until_a_new_conversation(tmp_path):
    asked: list[str] = []

    async def always(question: str) -> str:
        asked.append(question)
        return "a"

    session = recording_session(ScriptedLLM(), tmp_path, read_answer=always)
    assert await session._confirm("write a.txt (1 lines)")
    assert await session._confirm("write b.txt (1 lines)")
    assert await session._confirm("run `ls`")
    session.clear()
    assert await session._confirm("write c.txt (1 lines)")
    assert len(asked) == 3


@pytest.mark.unit
async def test_a_conversation_that_cannot_be_saved_keeps_the_session_open(
    tmp_path, monkeypatch
):
    def full_disk(*_):
        raise OSError("no space left")

    monkeypatch.setattr(history, "save", full_disk)
    session = recording_session(ScriptedLLM(answer("one"), answer("two")), tmp_path)
    await session.ask("first")
    await session.ask("second")
    out = session.renderer.console.export_text()
    assert "could not save the conversation: no space left" in out
    assert "two" in out and len(session.state.messages) == 4


@pytest.mark.unit
@pytest.mark.parametrize("command", ["  ", "'unclosed"])
def test_an_empty_or_broken_agent_command_is_a_usage_error(command, capsys):
    assert main(["--agent", command]) == 2
    assert "--agent" in capsys.readouterr().out
