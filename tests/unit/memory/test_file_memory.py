from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.message import Message, ToolCall, ToolCallResultMessage, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.protocol import ChatLLM
from ant_ai.memory import Memory
from ant_ai.memory.backends.file import FileMemory
from ant_ai.tools import ToolError

S1 = InvocationContext(session_id="s1")
S2 = InvocationContext(session_id="s2")


@pytest.fixture()
def memory_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "MEMORY.md"


@pytest.fixture()
def memory(memory_path: Path) -> FileMemory:
    return FileMemory(path=memory_path)


def saved(memory_path: Path, text: str) -> None:
    memory_path.parent.mkdir(exist_ok=True)
    memory_path.write_text(text, encoding="utf-8")


def test_is_a_memory(memory: FileMemory) -> None:
    assert isinstance(memory, Memory)


def test_path_can_be_given_as_a_string(memory_path: Path) -> None:
    assert FileMemory(path=str(memory_path)).path == memory_path


# --- read and write ----------------------------------------------------------


def test_read_of_a_missing_file_says_there_are_no_memories(
    memory: FileMemory, memory_path: Path
) -> None:
    assert memory.read() == "No memories saved yet."
    assert not memory_path.exists()


def test_read_of_an_empty_file_says_there_are_no_memories(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "\n")
    assert memory.read() == "No memories saved yet."


def test_read_returns_the_whole_file(memory: FileMemory, memory_path: Path) -> None:
    saved(memory_path, "# Memory\n\n- likes tea\n")
    assert memory.read() == "# Memory\n\n- likes tea\n"


def test_write_creates_the_file_and_its_directories(
    memory: FileMemory, memory_path: Path
) -> None:
    assert memory.write("- likes tea\n") == "Memory saved."
    assert memory_path.read_text(encoding="utf-8") == "- likes tea\n"


def test_write_replaces_the_whole_file(memory: FileMemory, memory_path: Path) -> None:
    saved(memory_path, "- likes tea\n")
    memory.read()
    memory.write("- likes coffee\n")
    assert memory.read() == "- likes coffee\n"


def test_write_with_nothing_clears_the_memory(memory: FileMemory) -> None:
    memory.write("- likes tea\n")
    assert memory.write("") == "Memory cleared."
    assert memory.read() == "No memories saved yet."


# --- a write needs a read of the memory as it is -----------------------------


def test_write_without_a_read_is_refused(memory: FileMemory, memory_path: Path) -> None:
    saved(memory_path, "- likes tea\n")
    with pytest.raises(ToolError, match="read it, then write it again"):
        memory.write("- likes coffee\n", ctx=S1)
    assert memory_path.read_text(encoding="utf-8") == "- likes tea\n"


def test_writes_after_one_read_are_allowed(memory: FileMemory) -> None:
    memory.write("- likes tea\n", ctx=S1)
    memory.write("- likes tea\n- lives in Lugano\n", ctx=S1)
    assert memory.read() == "- likes tea\n- lives in Lugano\n"


def test_write_after_another_conversation_changed_the_memory_is_refused(
    memory: FileMemory,
) -> None:
    memory.write("- likes tea\n", ctx=S1)
    memory.read(ctx=S2)
    memory.write("- likes tea\n- lives in Lugano\n", ctx=S2)
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes coffee\n", ctx=S1)
    memory.read(ctx=S1)
    memory.write("- likes coffee\n- lives in Lugano\n", ctx=S1)
    assert memory.read() == "- likes coffee\n- lives in Lugano\n"


def test_write_after_a_change_by_hand_is_refused(
    memory: FileMemory, memory_path: Path
) -> None:
    memory.write("- likes tea\n", ctx=S1)
    memory_path.write_text("- likes tea\n- edited by hand\n", encoding="utf-8")
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes coffee\n", ctx=S1)


async def test_of_two_writes_from_one_read_only_the_first_is_kept(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "- likes tea\n")
    memory.read(ctx=S1)
    memory.read(ctx=S2)
    results = await asyncio.gather(
        asyncio.to_thread(memory.write, "- likes coffee\n", ctx=S1),
        asyncio.to_thread(memory.write, "- likes juice\n", ctx=S2),
        return_exceptions=True,
    )
    assert sum(isinstance(r, ToolError) for r in results) == 1
    assert memory_path.read_text(encoding="utf-8") in (
        "- likes coffee\n",
        "- likes juice\n",
    )


def test_a_cleared_memory_is_not_restored_by_a_stale_write(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "- likes tea\n")
    memory.read(ctx=S1)
    memory.read(ctx=S2)
    assert memory.write("", ctx=S2) == "Memory cleared."
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes tea\n- likes coffee\n", ctx=S1)
    assert memory_path.read_text(encoding="utf-8") == ""


def test_a_deleted_memory_is_not_restored_by_a_stale_write(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "- likes tea\n")
    memory.read(ctx=S1)
    memory_path.unlink()
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes tea\n- likes coffee\n", ctx=S1)
    memory.read(ctx=S1)
    memory.write("- likes coffee\n", ctx=S1)


def test_an_empty_file_must_be_read_before_the_first_write(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "")
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes tea\n", ctx=S1)
    memory.read(ctx=S1)
    assert memory.write("- likes tea\n", ctx=S1) == "Memory saved."


def test_a_read_by_one_user_does_not_let_another_write(
    memory: FileMemory, memory_path: Path
) -> None:
    alice = InvocationContext(session_id="s", user_id="alice")
    bob = InvocationContext(session_id="s", user_id="bob")
    saved(memory_path, "- likes tea\n")
    memory.read(ctx=alice)
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes coffee\n", ctx=bob)
    memory.write("- likes coffee\n", ctx=alice)


def test_a_conversation_read_long_ago_must_read_again(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "- likes tea\n")
    memory.read(ctx=S1)
    for i in range(256):
        memory.read(ctx=InvocationContext(session_id=f"other-{i}"))
    with pytest.raises(ToolError, match="changed since you last read it"):
        memory.write("- likes coffee\n", ctx=S1)


def test_a_write_over_the_limit_is_refused(memory_path: Path) -> None:
    memory = FileMemory(path=memory_path, max_chars=10)
    with pytest.raises(ToolError, match="12 characters, over the limit of 10"):
        memory.write("- likes tea\n")
    assert not memory_path.exists()
    assert memory.write("- tea\n") == "Memory saved."


def test_no_limit(memory_path: Path) -> None:
    memory = FileMemory(path=memory_path, max_chars=None)
    memory.write("x" * 20_000)
    assert len(memory.read()) == 20_000


# --- files and errors --------------------------------------------------------


def test_a_failed_write_keeps_the_old_memory(
    memory: FileMemory, memory_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved(memory_path, "- likes tea\n")
    memory.read()

    def crash(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("ant_ai.core._files.os.replace", crash)
    with pytest.raises(ToolError, match="No space left on device"):
        memory.write("- likes coffee\n")
    assert memory_path.read_text(encoding="utf-8") == "- likes tea\n"
    assert [p.name for p in memory_path.parent.iterdir()] == ["MEMORY.md"]


def test_a_directory_at_the_path_raises(tmp_path: Path) -> None:
    memory = FileMemory(path=tmp_path)
    with pytest.raises(ToolError, match=str(tmp_path)):
        memory.read()
    with pytest.raises(ToolError, match=str(tmp_path)):
        memory.write("- likes tea\n")


def test_a_file_that_is_not_utf8_raises(memory: FileMemory, memory_path: Path) -> None:
    memory_path.parent.mkdir()
    memory_path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ToolError, match="not UTF-8"):
        memory.read()


def test_a_parent_that_is_a_file_raises(tmp_path: Path) -> None:
    (tmp_path / "state").write_text("", encoding="utf-8")
    memory = FileMemory(path=tmp_path / "state" / "MEMORY.md")
    with pytest.raises(ToolError, match="MEMORY.md"):
        memory.write("- likes tea\n")


# --- as seen by the model ----------------------------------------------------


def test_only_read_and_write_are_exposed_to_the_model(memory: FileMemory) -> None:
    tools = {t.name: t for t in memory._expand_namespace()}
    assert set(tools) == {"FileMemory_read", "FileMemory_write"}
    assert tools["FileMemory_read"].parameters["properties"] == {}
    assert tools["FileMemory_write"].parameters["required"] == ["content"]
    assert all(t.wants_context for t in tools.values())


class ScriptedLLM(ChatLLM):
    """Returns the scripted responses in order and records what it was sent."""

    model = "scripted"
    api_key = None
    api_base = None

    def __init__(self, *responses: ChatLLMResponse):
        self.responses = list(responses)
        self.seen: list[list] = []

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def call(cid: str, name: str, **arguments: object) -> ChatLLMResponse:
    return ChatLLMResponse(
        message=Message(role="assistant", content=""),
        tool_calls=[
            ToolCall(
                id=cid,
                function=ToolFunction(name=name, arguments=json.dumps(arguments)),
            )
        ],
    )


async def test_an_agent_with_it_as_memory_reads_and_writes_it(
    memory: FileMemory, memory_path: Path
) -> None:
    saved(memory_path, "- likes tea\n")
    llm = ScriptedLLM(
        call("c1", "FileMemory_read"),
        call("c2", "FileMemory_write", content="- likes tea\n- lives in Lugano\n"),
        ChatLLMResponse(message=Message(role="assistant", content="done")),
    )
    agent = Agent(name="a", system_prompt="sys", llm=llm, memory=memory)

    await agent.ainvoke(State(messages=[Message(role="user", content="hi")]), ctx=S1)

    results = [m for m in llm.seen[-1] if isinstance(m, ToolCallResultMessage)]
    assert [r.content for r in results] == ["- likes tea\n", "Memory saved."]
    assert memory_path.read_text(encoding="utf-8") == "- likes tea\n- lives in Lugano\n"
