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
from ant_ai.hooks.builtins import HistoryCompressionHook, PlanningHook
from ant_ai.hooks.layer import HookLayer
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.builtins.planning_tool import PlanningTool, plan_path, read_plan
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import ToolError

WRITE = "PlanningTool_write_todos"
PLAN = [
    {"content": "read the code", "status": "completed"},
    {"content": "fix the bug", "status": "in_progress"},
]
CTX = InvocationContext(session_id="s1")


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


def agent_with(planning: PlanningTool, llm: ChatLLM, *hooks) -> Agent:
    return Agent(
        name="a",
        system_prompt="sys",
        llm=llm,
        tools=[planning],
        hooks=[*hooks, PlanningHook(planning)],
    )


# --- PlanningTool -----------------------------------------------------------


@pytest.mark.unit
def test_write_todos_saves_the_plan_in_the_file_of_the_conversation(tmp_path: Path):
    planning = PlanningTool(agent="a", plan_dir=tmp_path)
    text = planning.write_todos(PLAN, ctx=CTX)

    assert text == "- [x] read the code\n- [~] fix the bug"
    assert (tmp_path / "a" / "s1.md").read_text(encoding="utf-8") == text
    assert read_plan(planning, CTX) == text


@pytest.mark.unit
def test_each_conversation_has_its_own_plan(tmp_path: Path):
    planning = PlanningTool(agent="a", plan_dir=tmp_path)
    planning.write_todos(PLAN, ctx=CTX)
    planning.write_todos(
        [{"content": "other task"}], ctx=InvocationContext(session_id="s2")
    )

    assert "fix the bug" in read_plan(planning, CTX)
    assert read_plan(planning, InvocationContext(session_id="s2")) == "- [ ] other task"
    assert read_plan(planning, InvocationContext(session_id="s3")) is None


@pytest.mark.unit
def test_two_agents_of_the_same_conversation_keep_their_own_plan(tmp_path: Path):
    # Over A2A the called agent receives the session_id of the caller.
    caller = PlanningTool(agent="caller", plan_dir=tmp_path)
    called = PlanningTool(agent="called", plan_dir=tmp_path)
    caller.write_todos(PLAN, ctx=CTX)
    called.write_todos([{"content": "search the sources"}], ctx=CTX)

    assert "fix the bug" in read_plan(caller, CTX)
    assert read_plan(called, CTX) == "- [ ] search the sources"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["../../escape", "/etc/passwd", "a b", ""])
def test_a_name_that_is_not_a_file_name_stays_inside_plan_dir(
    tmp_path: Path, name: str
):
    by_session = plan_path(
        PlanningTool(agent="a", plan_dir=tmp_path), InvocationContext(session_id=name)
    )
    by_agent = plan_path(PlanningTool(agent=name, plan_dir=tmp_path), CTX)
    assert by_session.parent == tmp_path / "a"
    assert by_agent.parent.parent == tmp_path


@pytest.mark.unit
def test_without_a_context_the_plan_goes_to_default(tmp_path: Path):
    PlanningTool(agent="a", plan_dir=tmp_path).write_todos(PLAN)
    assert (tmp_path / "a" / "default.md").is_file()


@pytest.mark.unit
def test_plans_go_to_dot_plans_by_default():
    assert plan_path(PlanningTool(agent="a"), CTX) == Path(".plans/a/s1.md")


@pytest.mark.unit
def test_two_items_in_progress_are_refused_and_the_plan_is_kept(tmp_path: Path):
    planning = PlanningTool(agent="a", plan_dir=tmp_path)
    planning.write_todos(PLAN, ctx=CTX)
    with pytest.raises(ToolError, match="only one item"):
        planning.write_todos(
            [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ],
            ctx=CTX,
        )
    assert "read the code" in read_plan(planning, CTX)


@pytest.mark.unit
def test_an_empty_plan_removes_the_file(tmp_path: Path):
    planning = PlanningTool(agent="a", plan_dir=tmp_path)
    planning.write_todos(PLAN, ctx=CTX)
    assert planning.write_todos([], ctx=CTX) == "Plan cleared."
    assert read_plan(planning, CTX) is None


@pytest.mark.unit
def test_the_model_sees_the_todos_but_not_the_context(tmp_path: Path):
    registry = ToolRegistry([PlanningTool(agent="a", plan_dir=tmp_path)])
    assert [t.name for t in registry.tools] == [WRITE]
    only = registry[WRITE]
    assert only.wants_context
    assert list(only.parameters["properties"]) == ["todos"]


# --- PlanningHook in an agent -----------------------------------------------


@pytest.mark.unit
async def test_the_plan_is_shown_to_the_model_and_not_kept_in_the_messages(
    tmp_path: Path,
):
    llm = ScriptedLLM(
        write("c1", PLAN),
        write("c2", [PLAN[0], {"content": "fix the bug", "status": "completed"}]),
        answer("fixed"),
    )
    state = State(messages=[Message(role="user", content="fix it")])

    assert (
        await agent_with(PlanningTool(agent="a", plan_dir=tmp_path), llm).ainvoke(
            state, ctx=CTX
        )
        == "fixed"
    )

    # First call: no plan yet. Then the plan, as a system message before the conversation.
    assert not any("[Current plan]" in (m.content or "") for m in llm.seen[0])
    assert llm.seen[1][1].role == "system"
    assert (
        llm.seen[1][1].content
        == "[Current plan]\n- [x] read the code\n- [~] fix the bug"
    )
    assert llm.seen[1][2].content == "fix it"
    assert "- [x] fix the bug" in llm.seen[2][1].content
    assert not any("[Current plan]" in (m.content or "") for m in state.messages)


@pytest.mark.unit
async def test_the_plan_is_back_on_the_next_message_of_the_conversation(
    tmp_path: Path,
):
    planning = PlanningTool(agent="a", plan_dir=tmp_path)
    await agent_with(
        planning, ScriptedLLM(write("c1", PLAN), answer("started"))
    ).ainvoke(State(messages=[Message(role="user", content="fix it")]), ctx=CTX)

    # A new State with only the next message, as an A2A server builds it.
    llm = ScriptedLLM(answer("going on"))
    await agent_with(planning, llm).ainvoke(
        State(messages=[Message(role="user", content="continue")]), ctx=CTX
    )

    assert llm.seen[0][1].content.startswith("[Current plan]\n- [x] read the code")


@pytest.mark.unit
async def test_the_plan_survives_history_compression(tmp_path: Path):
    class Summariser:
        async def ainvoke(self, messages, **_):
            return answer("earlier work, summarised")

    llm = ScriptedLLM(write("c1", PLAN), write("c2", PLAN), answer("done"))
    compression = HistoryCompressionHook(llm=Summariser(), max_messages=3, keep_last=1)
    agent = agent_with(PlanningTool(agent="a", plan_dir=tmp_path), llm, compression)
    await agent.ainvoke(State(messages=[Message(role="user", content="task")]), ctx=CTX)

    last = llm.seen[-1]
    # The write_todos calls were summarised away, the plan is still there, after the summary.
    assert not any(
        isinstance(m, ToolCallResultMessage) and m.name == WRITE for m in last[:-1]
    )
    assert last[1].content.startswith("[Conversation summary]")
    assert last[2].content.startswith("[Current plan]\n- [x] read the code")


@pytest.mark.unit
def test_planning_hook_keeps_live_streaming(tmp_path: Path):
    hook = PlanningHook(PlanningTool(agent="a", plan_dir=tmp_path))
    assert HookLayer(hooks=[hook]).is_stream_safe()


# --- Agent(planning=True) ---------------------------------------------------


@pytest.mark.unit
async def test_planning_true_adds_the_tool_and_the_hook_named_after_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    llm = ScriptedLLM(write("c1", PLAN), answer("done"))
    agent = Agent(name="worker", system_prompt="sys", llm=llm, planning=True)

    await agent.ainvoke(State(messages=[Message(role="user", content="task")]), ctx=CTX)

    assert (tmp_path / ".plans" / "worker" / "s1.md").is_file()
    assert llm.seen[1][1].content.startswith("[Current plan]\n- [x] read the code")


@pytest.mark.unit
def test_planning_true_with_a_planning_tool_in_tools_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="already registered"):
        Agent(
            name="worker",
            system_prompt="sys",
            llm=ScriptedLLM(),
            tools=[PlanningTool(agent="worker", plan_dir=tmp_path)],
            planning=True,
        )


@pytest.mark.unit
def test_an_agent_rebuilt_from_its_fields_keeps_one_planning_tool():
    # The ACP adapter rebuilds the agent of a session this way to add MCP tools.
    agent = Agent(name="worker", system_prompt="sys", llm=ScriptedLLM(), planning=True)
    rebuilt = agent.model_validate({**dict(agent), "tools": list(agent.tools)})
    assert [t.name for t in rebuilt.registry.tools] == [WRITE]
