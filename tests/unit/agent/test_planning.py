from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.message import (
    Message,
    ToolCall,
    ToolCallResultMessage,
    ToolFunction,
)
from ant_ai.core.response import ChatLLMResponse
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.builtins import HistoryCompressionHook
from ant_ai.hooks.builtins.planning import PlanningHook
from ant_ai.hooks.layer import HookLayer
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.builtins.planning_tool import PlanningTool, PlanStore
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import ToolError

WRITE = "PlanningTool_write_todos"
PLAN = [
    {"content": "read the code", "status": "completed"},
    {"content": "fix the bug", "status": "in_progress"},
]
CTX = InvocationContext(session_id="s1")


@pytest.fixture(autouse=True)
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Plans are written to .plans in the working directory.
    monkeypatch.chdir(tmp_path)
    return tmp_path


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


def answer(text: str) -> ChatLLMResponse:
    return ChatLLMResponse(message=Message(role="assistant", content=text))


def write(cid: str, todos: list[dict]) -> ChatLLMResponse:
    return ChatLLMResponse(
        message=Message(role="assistant", content=""),
        tool_calls=[
            ToolCall(
                id=cid,
                function=ToolFunction(
                    name=WRITE, arguments=json.dumps({"todos": todos})
                ),
            )
        ],
    )


def agent_with(llm: ChatLLM, *hooks) -> Agent:
    return Agent(
        name="a", system_prompt="sys", llm=llm, hooks=list(hooks), planning=True
    )


# --- PlanningTool and PlanStore ----------------------------------------------

STORE = PlanStore()


@pytest.mark.unit
def test_write_todos_saves_the_plan_in_the_file_of_the_conversation(in_tmp: Path):
    text = PlanningTool(agent="a").write_todos(PLAN, ctx=CTX)

    assert text == "- [x] read the code\n- [~] fix the bug"
    assert (in_tmp / ".plans" / "a" / "s1.md").read_text(encoding="utf-8") == text
    assert STORE.read("a", "s1") == text
    assert [p.name for p in (in_tmp / ".plans" / "a").iterdir()] == ["s1.md"]


@pytest.mark.unit
def test_each_conversation_has_its_own_plan():
    planning = PlanningTool(agent="a")
    planning.write_todos(PLAN, ctx=CTX)
    planning.write_todos(
        [{"content": "other task"}], ctx=InvocationContext(session_id="s2")
    )

    assert "fix the bug" in STORE.read("a", "s1")
    assert STORE.read("a", "s2") == "- [ ] other task"
    assert STORE.read("a", "s3") is None


@pytest.mark.unit
def test_two_agents_of_the_same_conversation_keep_their_own_plan():
    # Over A2A the called agent receives the session_id of the caller.
    PlanningTool(agent="caller").write_todos(PLAN, ctx=CTX)
    PlanningTool(agent="called").write_todos(
        [{"content": "search the sources"}], ctx=CTX
    )

    assert "fix the bug" in STORE.read("caller", "s1")
    assert STORE.read("called", "s1") == "- [ ] search the sources"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "start"),
    [
        ("mario@gmail.com", "mario_gmail_com."),
        ("../../.bashrc", "_______bashrc."),
        ("/etc/passwd", "_etc_passwd."),
        ("", "."),
    ],
)
def test_a_changed_name_stays_readable_and_gets_a_short_hash(name: str, start: str):
    by_session = STORE.path("a", name)
    by_agent = STORE.path(name, "s1")

    assert by_session.parent == Path(".plans", "a")
    assert by_session.name.startswith(start)
    assert len(by_session.stem) == len(start) + 8
    assert by_agent.parent.parent == Path(".plans")


@pytest.mark.unit
def test_a_plain_name_is_kept_and_two_names_never_share_a_file():
    assert STORE.path("a", "3f2a9c1e-7b4d") == Path(".plans", "a", "3f2a9c1e-7b4d.md")
    assert STORE.path("a", "a b") != STORE.path("a", "a_b")
    # Not even an id written like the changed name of another one.
    changed = STORE.path("a", "a/b")
    assert STORE.path("a", changed.stem) != changed


@pytest.mark.unit
def test_without_a_context_the_plan_is_not_saved(in_tmp: Path):
    assert PlanningTool(agent="a").write_todos(PLAN) == (
        "- [x] read the code\n- [~] fix the bug"
    )
    assert not (in_tmp / ".plans").exists()


@pytest.mark.unit
def test_two_items_in_progress_are_refused_and_the_plan_is_kept():
    planning = PlanningTool(agent="a")
    planning.write_todos(PLAN, ctx=CTX)
    with pytest.raises(ToolError, match="only one item"):
        planning.write_todos(
            [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ],
            ctx=CTX,
        )
    assert "read the code" in STORE.read("a", "s1")


@pytest.mark.unit
def test_an_empty_plan_removes_the_file():
    planning = PlanningTool(agent="a")
    planning.write_todos(PLAN, ctx=CTX)
    assert planning.write_todos([], ctx=CTX) == "Plan cleared."
    assert STORE.read("a", "s1") is None


@pytest.mark.unit
def test_a_plan_with_every_item_completed_is_cleared():
    planning = PlanningTool(agent="a")
    planning.write_todos(PLAN, ctx=CTX)
    done = [{**t, "status": "completed"} for t in PLAN]
    assert "plan is cleared" in planning.write_todos(done, ctx=CTX)
    assert STORE.read("a", "s1") is None


@pytest.mark.unit
def test_the_model_sees_the_todos_but_not_the_context():
    registry = ToolRegistry([PlanningTool(agent="a")])
    assert [t.name for t in registry.tools] == [WRITE]
    only = registry[WRITE]
    assert only.wants_context
    assert list(only.parameters["properties"]) == ["todos"]


# --- Agent(planning=True) ---------------------------------------------------


def has_plan(messages) -> bool:
    return any("[Current plan]" in (m.content or "") for m in messages)


@pytest.mark.unit
async def test_during_the_run_the_plan_is_only_in_the_write_todos_result():
    llm = ScriptedLLM(
        write("c1", PLAN),
        write("c2", [PLAN[0], {"content": "fix the bug", "status": "in_progress"}]),
        answer("fixed"),
    )
    state = State(messages=[Message(role="user", content="fix it")])

    assert await agent_with(llm).ainvoke(state, ctx=CTX) == "fixed"

    # Nothing is added, so every call starts with the messages of the call before.
    assert not any(has_plan(seen) for seen in llm.seen)
    assert llm.seen[1][: len(llm.seen[0])] == llm.seen[0]
    assert llm.seen[2][: len(llm.seen[1])] == llm.seen[1]


@pytest.mark.unit
async def test_on_the_next_message_the_plan_is_added_at_the_end():
    await agent_with(ScriptedLLM(write("c1", PLAN), answer("started"))).ainvoke(
        State(messages=[Message(role="user", content="fix it")]), ctx=CTX
    )

    # A new State with only the next message, from a caller that keeps only the session_id.
    llm = ScriptedLLM(answer("going on"))
    state = State(messages=[Message(role="user", content="continue")])
    await agent_with(llm).ainvoke(state, ctx=CTX)

    assert llm.seen[0][-1].role == "user"
    assert llm.seen[0][-1].content == (
        "continue\n\n[Current plan]\n- [x] read the code\n- [~] fix the bug"
    )
    assert not has_plan(state.messages)


@pytest.mark.unit
async def test_two_runs_without_a_context_do_not_share_a_plan():
    await agent_with(ScriptedLLM(write("c1", PLAN), answer("done"))).ainvoke(
        State(messages=[Message(role="user", content="task one")])
    )
    llm = ScriptedLLM(answer("ok"))
    await agent_with(llm).ainvoke(
        State(messages=[Message(role="user", content="task two")])
    )
    assert not has_plan(llm.seen[0])


@pytest.mark.unit
async def test_the_plan_is_back_after_history_compression():
    class Summariser:
        async def ainvoke(self, messages, **_):
            return answer("earlier work, summarised")

    llm = ScriptedLLM(write("c1", PLAN), answer("done"))
    # The write_todos call and its result are summarised before the second call.
    compression = HistoryCompressionHook(llm=Summariser(), max_messages=3, keep_last=0)
    state = State(messages=[Message(role="user", content="task")])
    await agent_with(llm, compression).ainvoke(state, ctx=CTX)

    last = llm.seen[-1]
    assert not any(isinstance(m, ToolCallResultMessage) for m in last)
    assert last[1].content.startswith("[Conversation summary]")
    assert last[-1].content.endswith(
        "[Current plan]\n- [x] read the code\n- [~] fix the bug"
    )


@pytest.mark.unit
def test_planning_hook_keeps_live_streaming():
    assert HookLayer(hooks=[PlanningHook(agent="a", store=STORE)]).is_stream_safe()


@pytest.mark.unit
async def test_planning_true_writes_to_the_folder_of_the_agent(in_tmp: Path):
    llm = ScriptedLLM(write("c1", PLAN), answer("done"))
    agent = Agent(name="worker", system_prompt="sys", llm=llm, planning=True)

    await agent.ainvoke(State(messages=[Message(role="user", content="task")]), ctx=CTX)

    assert (in_tmp / ".plans" / "worker" / "s1.md").is_file()


@pytest.mark.unit
def test_planning_true_with_a_planning_tool_in_tools_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        Agent(
            name="worker",
            system_prompt="sys",
            llm=ScriptedLLM(),
            tools=[PlanningTool(agent="worker")],
            planning=True,
        )


@pytest.mark.unit
def test_an_agent_rebuilt_from_its_fields_keeps_one_planning_tool():
    # The ACP adapter rebuilds the agent of a session this way to add MCP tools.
    agent = Agent(name="worker", system_prompt="sys", llm=ScriptedLLM(), planning=True)
    rebuilt = agent.model_validate({**dict(agent), "tools": list(agent.tools)})
    assert [t.name for t in rebuilt.registry.tools] == [WRITE]


@pytest.mark.unit
def test_planning_is_turned_on_only_with_planning_true():
    import ant_ai.hooks.builtins as hooks
    import ant_ai.tools.builtins as tools

    assert "PlanningTool" not in tools.__all__
    assert "PlanningHook" not in hooks.__all__
