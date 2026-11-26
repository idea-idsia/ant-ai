from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    PermissionOption,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
)
from rich.console import Console

from ant_ai.cli.acp_client import ACPEvents, RemoteSession, _Client
from ant_ai.cli.render import Renderer
from ant_ai.core.events import (
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    ToolCallingEvent,
    ToolResultEvent,
)

AGENT = Path(__file__).with_name("scripted_acp_agent.py")


def chunk(text: str) -> AgentMessageChunk:
    return AgentMessageChunk(
        session_update="agent_message_chunk",
        content=TextContentBlock(type="text", text=text),
    )


def start(call_id: str, name: str, **args) -> ToolCallStart:
    return ToolCallStart(
        session_update="tool_call",
        tool_call_id=call_id,
        title=name,
        status="in_progress",
        raw_input=args,
    )


def done(call_id: str, output: str, status: str = "completed") -> ToolCallProgress:
    return ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=call_id,
        status=status,
        raw_output=output,
    )


@pytest.mark.unit
def test_text_before_the_calls_is_the_reason_and_the_rest_is_the_answer():
    events = ACPEvents()
    out = []
    for update in [
        chunk("I read "),
        chunk("a.py."),
        start("c1", "read_file", path="a.py"),
        done("c1", "print(1)"),
        chunk("It prints 1."),
    ]:
        out += events.update(update)
    out += events.finish("end_turn")

    calling, result, final = out
    assert isinstance(calling, ToolCallingEvent)
    assert calling.content == "I read a.py."
    assert calling.tool_calls[0].function.name == "read_file"
    assert calling.tool_calls[0].function.arguments == '{"path": "a.py"}'
    assert isinstance(result, ToolResultEvent)
    assert (result.name, result.content, result.is_error) == (
        "read_file",
        "print(1)",
        False,
    )
    assert isinstance(final, FinalAnswerEvent)
    assert final.content == "It prints 1."


@pytest.mark.unit
def test_calls_in_a_row_are_one_step():
    events = ACPEvents()
    out = []
    for update in [
        chunk("Two reads."),
        start("c1", "read_file", path="a.py"),
        start("c2", "read_file", path="b.py"),
        done("c1", "x"),
        done("c2", "y"),
    ]:
        out += events.update(update)

    steps = [e for e in out if isinstance(e, ToolCallingEvent)]
    assert len(steps) == 1
    assert [c.id for c in steps[0].tool_calls] == ["c1", "c2"]


@pytest.mark.unit
def test_a_failed_call_is_an_error_and_thoughts_are_reasoning():
    events = ACPEvents()
    thought = AgentThoughtChunk(
        session_update="agent_thought_chunk",
        content=TextContentBlock(type="text", text="hmm"),
    )
    out = events.update(thought)
    out += events.update(start("c1", "run_command", command="pytest"))
    out += events.update(done("c1", "ERROR: no", status="failed"))

    assert isinstance(out[0], ReasoningEvent)
    assert isinstance(out[-1], ToolResultEvent) and out[-1].is_error


@pytest.mark.unit
def test_the_step_limit_ends_the_turn_without_an_answer():
    events = ACPEvents()
    events.update(chunk("partial"))

    (event,) = events.finish("max_turn_requests")

    assert isinstance(event, MaxStepsReachedEvent)


def remote(
    workspace: Path, answer: str, script: str = "write"
) -> tuple[RemoteSession, list[str]]:
    asked: list[str] = []

    async def read_answer(question: str) -> str:
        asked.append(question)
        return answer

    console = Console(record=True, width=100, force_terminal=False)
    session = RemoteSession(
        [sys.executable, str(AGENT), str(workspace), script],
        workspace,
        label="scripted",
        renderer=Renderer(console, spinner=False),
        read_answer=read_answer,
    )
    return session, asked


@pytest.mark.unit
async def test_a_turn_over_acp_looks_like_a_turn_in_process(tmp_path):
    session, asked = remote(tmp_path, "y")

    async with session.connect():
        await session.ask("create a.txt")

    out = session.renderer.console.export_text()
    assert "● I write a.txt.\n  ⎿ write_file  a.txt  written" in out
    assert "Wrote a.txt." in out
    assert "2 steps · 1 tool" in out
    assert asked == ["allow write a.txt (1 lines)? y/n/a"]
    assert (tmp_path / "a.txt").read_text() == "x"


@pytest.mark.unit
async def test_a_refusal_over_acp_reaches_the_agent(tmp_path):
    session, _ = remote(tmp_path, "n")

    async with session.connect():
        await session.ask("create a.txt")

    out = session.renderer.console.export_text()
    assert "✗ The user did not allow: write a.txt" in out
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.unit
async def test_the_step_is_printed_before_its_permission_question():
    console = Console(record=True, width=100, force_terminal=False)
    printed_before_question: list[str] = []

    async def read_answer(question: str) -> str:
        printed_before_question.append(console.export_text(clear=False))
        return "y"

    client = _Client(Renderer(console, spinner=False), read_answer)
    await client.session_update("s", chunk("I run the tests."))
    await client.session_update("s", start("c1", "run_command", command="pytest"))
    await client.request_permission(
        session_id="s",
        tool_call=ToolCallUpdate(tool_call_id="p1", title="run `pytest`"),
        options=[PermissionOption(option_id="ok", name="Allow", kind="allow_once")],
    )

    assert "● I run the tests." in printed_before_question[0]


@pytest.mark.unit
async def test_updates_during_a_question_are_printed_after_it():
    console = Console(record=True, width=100, force_terminal=False)
    client: _Client

    async def read_answer(question: str) -> str:
        # Another tool of the same step finishes while the user is asked.
        await client.session_update("s", done("c1", "a.py"))
        assert "list_dir" not in console.export_text(clear=False)
        return "y"

    client = _Client(Renderer(console, spinner=False), read_answer)
    await client.session_update("s", start("c1", "list_dir", path="."))
    await client.session_update("s", start("c2", "run_command", command="pytest"))
    await client.request_permission(
        session_id="s",
        tool_call=ToolCallUpdate(tool_call_id="p1", title="run `pytest`"),
        options=[PermissionOption(option_id="ok", name="Allow", kind="allow_once")],
    )

    assert "list_dir  .  a.py" in console.export_text()


@pytest.mark.unit
async def test_a_step_shows_while_its_tool_is_still_running():
    console = Console(record=True, width=100, force_terminal=False)

    async def read_answer(question: str) -> str:
        return "y"

    client = _Client(Renderer(console, spinner=False), read_answer)
    await client.session_update("s", chunk("I wait."))
    await client.session_update("s", start("c1", "run_command", command="sleep 20"))
    await asyncio.sleep(0.2)

    assert "● I wait." in console.export_text()


@pytest.mark.unit
async def test_text_chunks_reach_the_view_while_they_arrive():
    seen: list[str] = []

    class Recording(Renderer):
        def partial(self, text: str) -> None:
            seen.append(text)

    async def read_answer(question: str) -> str:
        return "y"

    client = _Client(Recording(Console(record=True), spinner=False), read_answer)
    await client.session_update("s", chunk("Hel"))
    await client.session_update("s", chunk("lo"))

    assert seen == ["Hel", "lo"]


@pytest.mark.unit
async def test_always_allows_the_rest_of_the_session_over_acp(tmp_path):
    session, asked = remote(tmp_path, "a", "twice")

    async with session.connect():
        await session.ask("create two files")

    assert asked == ["allow write a.txt (1 lines)? y/n/a"]
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()


@pytest.mark.unit
async def test_stopping_a_turn_stops_the_agent_and_the_session_goes_on(tmp_path):
    session, _ = remote(tmp_path, "y", "stuck")

    async with session.connect():
        turn = asyncio.ensure_future(session.ask("wait forever"))
        for _ in range(200):
            if (tmp_path / "started").exists():
                break
            await asyncio.sleep(0.05)
        turn.cancel()
        await asyncio.wait_for(turn, 5)
        await session.ask("are you there?")

    out = session.renderer.console.export_text()
    assert "interrupted" in out
    assert "done" in out


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    [
        "write_text_file",
        "read_text_file",
        "create_terminal",
        "terminal_output",
        "release_terminal",
        "wait_for_terminal_exit",
        "kill_terminal",
        "create_elicitation",
        "connect_mcp",
        "disconnect_mcp",
        "mcp_message",
    ],
)
async def test_the_client_offers_no_files_terminals_or_mcp(method):
    from acp import RequestError

    client = _Client(Renderer(Console(record=True)), read_answer=lambda q: None)
    with pytest.raises(RequestError):
        await getattr(client, method)()


@pytest.mark.unit
def test_a_result_sent_as_content_is_kept_and_a_falsy_one_too():
    events = ACPEvents()
    events.update(start("c1", "lookup"))
    as_content = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="c1",
        status="completed",
        content=[
            ContentToolCallContent(
                type="content", content=TextContentBlock(type="text", text="found it")
            )
        ],
    )
    (result,) = events.update(as_content)[-1:]
    assert result.content == "found it"

    events.update(start("c2", "count"))
    zero = ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id="c2",
        status="completed",
        raw_output=0,
    )
    assert events.update(zero)[-1].content == "0"


@pytest.mark.unit
async def test_two_questions_at_once_each_get_their_own_answer():
    async def answer(question: str) -> str:
        await asyncio.sleep(0.01)
        return "y" if "a.txt" in question else "n"

    client = _Client(
        Renderer(Console(record=True, width=100, force_terminal=False)), answer
    )
    options = [
        PermissionOption(option_id="ok", name="Allow", kind="allow_once"),
        PermissionOption(option_id="no", name="Reject", kind="reject_once"),
    ]

    def ask(path: str):
        call = ToolCallUpdate(tool_call_id=path, title=f"write {path}")
        return client.request_permission("s", call, options)

    first, second = await asyncio.gather(ask("a.txt"), ask("b.txt"))
    assert first.outcome.option_id == "ok"
    assert second.outcome.option_id == "no"


@pytest.mark.unit
async def test_a_huge_log_line_does_not_stop_reading_the_agents_stderr(tmp_path):
    script = "import sys; sys.stderr.write('x' * 200_000 + '\\nlast line\\n')"
    session = RemoteSession(
        [sys.executable, "-c", script],
        tmp_path,
        label="t",
        renderer=Renderer(Console(record=True, force_terminal=False)),
        read_answer=lambda _: asyncio.sleep(0, ""),
    )
    process = await asyncio.create_subprocess_exec(
        *session.command, stderr=asyncio.subprocess.PIPE
    )
    await asyncio.wait_for(session._drain(process), 5)
    await process.wait()
    assert session.stderr[-1] == "last line"
