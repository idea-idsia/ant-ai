from __future__ import annotations

import json

import pytest

from ant_ai.agent import Agent, SubagentTool
from ant_ai.core.events import FinalAnswerEvent, ToolResultEvent
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.response import ChatLLMResponse
from ant_ai.core.types import InvocationContext, State
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.builtins.planning_tool import PlanningTool, PlanStore
from ant_ai.tools.tool import tool


class ScriptedLLM(ChatLLM):
    """Returns the scripted responses in order and records each request."""

    model = "scripted"
    api_key = None
    api_base = None

    def __init__(self, *responses: ChatLLMResponse):
        self.responses = list(responses)
        self.requests: list[tuple[list, list[str]]] = []

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        names = [t["function"]["name"] for t in tools or []]
        self.requests.append((list(messages), names))
        return self.responses.pop(0)


def answer(text: str) -> ChatLLMResponse:
    return ChatLLMResponse(message=Message(role="assistant", content=text))


def call(name: str, arguments: dict, call_id: str = "c1") -> ChatLLMResponse:
    return ChatLLMResponse(
        message=Message(role="assistant", content=""),
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolFunction(name=name, arguments=json.dumps(arguments)),
            )
        ],
    )


@tool
def lookup(key: str) -> str:
    """Look a key up."""
    return f"value of {key}"


@tool
def delete(key: str) -> str:
    """Delete a key."""
    return "deleted"


def spawn(name: str = "finder", task: str = "find k", tools=("lookup",)) -> dict:
    return {"name": name, "task": task, "tools": list(tools)}


async def run(lead: Agent, ctx: InvocationContext | None = None) -> list:
    state = State(messages=[Message(role="user", content="what is k?")])
    return [e async for e in lead.stream(state, ctx=ctx)]


def results(events: list) -> list[tuple[str, str, bool]]:
    return [
        (e.name, e.content, e.is_error)
        for e in events
        if isinstance(e, ToolResultEvent)
    ]


@pytest.mark.unit
async def test_the_model_creates_a_subagent_with_some_of_its_tools():
    llm = ScriptedLLM(
        call("subagent", spawn()),
        call("lookup", {"key": "k"}),
        answer("k is value of k"),
        answer("done"),
    )
    lead = Agent(
        name="lead",
        system_prompt="lead",
        llm=llm,
        tools=[lookup, delete, SubagentTool()],
    )

    events = await run(lead)

    assert results(events)[-1] == ("subagent", "k is value of k", False)
    assert isinstance(events[-1], FinalAnswerEvent) and events[-1].content == "done"
    lead_tools = llm.requests[0][1]
    assert lead_tools == ["lookup", "delete", "subagent"]
    sub_messages, sub_tools = llm.requests[1]
    assert sub_tools == ["lookup"]
    assert [m.content for m in sub_messages[1:]] == ["find k"]


@pytest.mark.unit
async def test_the_subagent_gets_the_default_or_the_given_system_prompt():
    llm = ScriptedLLM(call("subagent", spawn()), answer("k"), answer("done"))
    default = Agent(
        name="a", system_prompt="a", llm=llm, tools=[lookup, SubagentTool()]
    )
    await run(default)
    assert llm.requests[1][0][0].content == SubagentTool().system_prompt

    llm = ScriptedLLM(call("subagent", spawn()), answer("k"), answer("done"))
    custom = SubagentTool(system_prompt="Answer in one line.")
    await run(Agent(name="a", system_prompt="a", llm=llm, tools=[lookup, custom]))
    assert llm.requests[1][0][0].content == "Answer in one line."


@pytest.mark.unit
async def test_a_tool_the_agent_does_not_have_is_refused():
    llm = ScriptedLLM(call("subagent", spawn(tools=["lookup", "rm"])), answer("ok"))
    lead = Agent(name="a", system_prompt="a", llm=llm, tools=[lookup, SubagentTool()])

    events = await run(lead)

    name, content, is_error = results(events)[0]
    assert is_error
    assert content == "ERROR: unknown tool 'rm': choose from lookup"
    assert len(llm.requests) == 2


@pytest.mark.unit
async def test_a_subagent_cannot_create_subagents_or_keep_a_plan():
    llm = ScriptedLLM(
        call("subagent", spawn(tools=["subagent"])),
        call("subagent", spawn(tools=["PlanningTool_write_todos"]), "c2"),
        answer("ok"),
    )
    lead = Agent(
        name="a", system_prompt="a", llm=llm, tools=[SubagentTool()], planning=True
    )

    events = await run(lead, InvocationContext(session_id="s1"))

    assert [r[1] for r in results(events)] == [
        "ERROR: unknown tool 'subagent': choose from no tool",
        "ERROR: unknown tool 'PlanningTool_write_todos': choose from no tool",
    ]


@pytest.mark.unit
async def test_the_subagent_runs_with_its_name_and_the_callers_context():
    seen = []

    @tool
    def whoami(ctx: InvocationContext | None = None) -> str:
        """Return the session."""
        seen.append(ctx)
        return "ok"

    llm = ScriptedLLM(
        call("subagent", spawn(name="checker", tools=["whoami"])),
        call("whoami", {}),
        answer("checked"),
        answer("done"),
    )
    lead = Agent(name="a", system_prompt="a", llm=llm, tools=[whoami, SubagentTool()])
    ctx = InvocationContext(session_id="s1")

    await run(lead, ctx)

    assert seen == [ctx]


@pytest.mark.unit
async def test_a_tool_added_later_can_be_given_to_a_subagent():
    llm = ScriptedLLM(call("subagent", spawn()), answer("k"), answer("done"))
    lead = Agent(name="a", system_prompt="a", llm=llm)
    lead.add_tool(SubagentTool())
    lead.add_tool(lookup)

    events = await run(lead)

    assert results(events) == [("subagent", "k", False)]
    assert llm.requests[1][1] == ["lookup"]


@pytest.mark.unit
def test_a_planning_tool_given_explicitly_is_not_delegable():
    given = Agent(
        name="a",
        system_prompt="a",
        llm=ScriptedLLM(),
        tools=[lookup, PlanningTool(agent="a", store=PlanStore()), SubagentTool()],
    )
    added = Agent(name="a", system_prompt="a", llm=ScriptedLLM(), tools=[lookup])
    added.add_tool(SubagentTool())
    added.add_tool(PlanningTool(agent="a", store=PlanStore()))

    for lead in (given, added):
        assert "PlanningTool_write_todos" in lead.registry
        assert list(lead._delegable_tools()) == ["lookup"]


@pytest.mark.unit
def test_a_name_freed_by_a_subagent_tool_can_be_delegated_again():
    lead = Agent(
        name="a",
        system_prompt="a",
        llm=ScriptedLLM(),
        tools=[SubagentTool(), SubagentTool(name="lookup")],
    )
    lead.registry.remove("lookup")
    lead.add_tool(lookup)

    assert list(lead._delegable_tools()) == ["lookup"]


@pytest.mark.unit
def test_the_model_sees_name_task_and_tools():
    lead = Agent(name="a", system_prompt="a", llm=ScriptedLLM(), tools=[SubagentTool()])
    params = lead.registry["subagent"].parameters
    assert list(params["properties"]) == ["name", "task", "tools"]
    assert params["required"] == ["name", "task", "tools"]


@pytest.mark.unit
def test_name_and_steps_can_be_changed():
    lead = Agent(
        name="a",
        system_prompt="a",
        llm=ScriptedLLM(),
        tools=[SubagentTool(name="helper", max_steps=3)],
    )
    assert "helper" in lead.registry
