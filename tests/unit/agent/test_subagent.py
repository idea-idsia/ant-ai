from __future__ import annotations

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.events import FinalAnswerEvent, ToolResultEvent
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.core.result import ClarificationNeededOutput
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.protocol import AgentHook
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.tool import ToolError, tool


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


def call(name: str, arguments: str, call_id: str = "c1") -> ChatLLMResponse:
    return ChatLLMResponse(
        message=Message(role="assistant", content=""),
        tool_calls=[
            ToolCall(id=call_id, function=ToolFunction(name=name, arguments=arguments))
        ],
    )


def agent(llm: ChatLLM, **kwargs) -> Agent:
    fields = {"name": "sub", "description": "Does a task.", "system_prompt": "s"}
    return Agent(llm=llm, **(fields | kwargs))


@pytest.mark.unit
async def test_subagent_starts_from_the_task_and_returns_only_its_answer():
    seen_ctx: list[InvocationContext | None] = []

    @tool
    def lookup(key: str, ctx: InvocationContext | None = None) -> str:
        """Look a key up."""
        seen_ctx.append(ctx)
        return f"value of {key}"

    sub_llm = ScriptedLLM(call("lookup", '{"key": "k"}'), answer("k is value of k"))
    researcher = agent(sub_llm, name="researcher", tools=[lookup])
    lead_llm = ScriptedLLM(call("researcher", '{"message": "find k"}'), answer("done"))
    lead = Agent(
        name="lead", system_prompt="lead", llm=lead_llm, tools=[researcher.as_tool()]
    )

    ctx = InvocationContext(session_id="s1")
    state = State(messages=[Message(role="user", content="what is k?")])
    events = [e async for e in lead.stream(state, ctx=ctx)]

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [(r.name, r.content, r.is_error) for r in results] == [
        ("researcher", "k is value of k", False)
    ]
    assert isinstance(events[-1], FinalAnswerEvent) and events[-1].content == "done"
    assert [m.content for m in sub_llm.seen[0][1:]] == ["find k"]
    assert not any(getattr(m, "name", None) == "lookup" for m in state.messages)
    assert seen_ctx == [ctx]


@pytest.mark.unit
def test_defaults_come_from_the_agent_and_can_be_overridden():
    sub = agent(ScriptedLLM(), name="writer", description="Writes text.")

    default = sub.as_tool()
    assert (default.name, default.description) == ("writer", "Writes text.")

    custom = sub.as_tool(name="w2", description="Other.")
    assert (custom.name, custom.description) == ("w2", "Other.")


@pytest.mark.unit
def test_the_model_sees_only_the_message_argument():
    params = agent(ScriptedLLM()).as_tool().parameters
    assert list(params["properties"]) == ["message"]
    assert params["required"] == ["message"]
    assert "ctx" not in str(params)


@pytest.mark.unit
def test_an_agent_with_planning_is_refused():
    with pytest.raises(ValueError, match="does not keep a plan"):
        agent(ScriptedLLM(), planning=True).as_tool()


@pytest.mark.unit
async def test_running_out_of_steps_is_an_error_for_the_caller():
    @tool
    def spin() -> str:
        """Spin."""
        return "again"

    llm = ScriptedLLM(*(call("spin", "{}", f"c{i}") for i in range(2)))
    sub = agent(llm, tools=[spin]).as_tool(max_steps=2)

    with pytest.raises(ToolError, match="stopped after 2 steps"):
        await sub.ainvoke(message="loop")


@pytest.mark.unit
async def test_a_question_to_the_user_is_an_error_for_the_caller():
    @tool
    def ask() -> ClarificationNeededOutput:
        """Ask the user."""
        return ClarificationNeededOutput(question="Which file?")

    sub = agent(ScriptedLLM(call("ask", "{}")), tools=[ask]).as_tool()

    with pytest.raises(ToolError, match="needs more information: Which file"):
        await sub.ainvoke(message="fix it")


@pytest.mark.unit
async def test_the_subagent_run_is_closed_before_the_error_reaches_the_caller():
    closed: list[str] = []

    class Record(AgentHook):
        async def after_agent(self, state, ctx):
            closed.append("after_agent")

    @tool
    def spin() -> str:
        """Spin."""
        return "again"

    llm = ScriptedLLM(call("spin", "{}"))
    sub = agent(llm, tools=[spin], hooks=[Record()]).as_tool(max_steps=1)

    with pytest.raises(ToolError):
        await sub.ainvoke(message="loop")
    assert closed == ["after_agent"]
