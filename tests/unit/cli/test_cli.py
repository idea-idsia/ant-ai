from __future__ import annotations

import json
import os

import pytest
from rich.console import Console

from ant_ai.cli import Session, main
from ant_ai.cli.render import PLAN_TOOL, Renderer
from ant_ai.cli.tools import coding_tools
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


@pytest.fixture(autouse=True)
def ant_home(tmp_path_factory, monkeypatch):
    """Keep saved conversations out of the real home and out of the workspace."""
    home = tmp_path_factory.mktemp("ant-home")
    monkeypatch.setenv("ANT_HOME", str(home))
    return home


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
async def test_tools_stay_inside_the_workspace(tmp_path):
    tools = {t.name: t for t in coding_tools(tmp_path)}
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
async def test_the_plan_is_saved_for_the_conversation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    todos = [{"content": "fix the bug", "status": "in_progress"}]
    llm = ScriptedLLM(call("I plan.", PLAN_TOOL, todos=todos), answer("ok"))
    session = recording_session(llm, tmp_path)

    await session.ask("fix it")

    plan = tmp_path / ".plans" / "ant" / f"{session.ctx.session_id}.md"
    assert "fix the bug" in plan.read_text()


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
