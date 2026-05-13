from __future__ import annotations

import json
from typing import Any

import pytest

from ant_ai.core.events import FinalAnswerEvent, ToolResultEvent
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.types import State
from ant_ai.llm.protocol import ChatLLM


class DummyResponse:
    """Duck-typed stand-in for ChatLLMResponse."""

    def __init__(self, message: Message, tool_calls: list[ToolCall] | None = None):
        self.message = message
        self.tool_calls = tool_calls or []


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=ToolFunction(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.unit
def test_no_skills_means_no_use_skill_tool(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    agent = make_agent(skills=[], llm=NoopLLM())
    assert "use_skill" not in agent.registry


@pytest.mark.unit
def test_skills_present_registers_use_skill_tool(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    agent = make_agent(skills=[make_skill()], llm=NoopLLM())
    assert "use_skill" in agent.registry


@pytest.mark.unit
def test_user_tool_named_use_skill_plus_skills_raises(make_skill):
    from ant_ai.agent.agent import Agent
    from ant_ai.tools.tool import Tool

    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    conflict_tool = Tool._from_function(lambda: "x", name="use_skill")
    with pytest.raises(ValueError, match="use_skill"):
        Agent(
            name="test",
            system_prompt="sys",
            llm=NoopLLM(),
            tools=[conflict_tool],
            skills=[make_skill()],
        )


@pytest.mark.unit
def test_no_skills_system_message_unchanged(make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    agent = make_agent(skills=[], llm=NoopLLM())
    assert agent.system_message.content == "You are a helpful agent."


@pytest.mark.unit
def test_skills_present_appends_discovery_section(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    skill = make_skill(name="pdf-tool", description="Handles PDFs.")
    agent = make_agent(skills=[skill], llm=NoopLLM())
    content = agent.system_message.content
    assert "## Available Skills" in content
    assert "pdf-tool" in content
    assert "Handles PDFs." in content
    assert "use_skill" in content


@pytest.mark.unit
def test_system_prompt_field_unchanged_by_skills(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    agent = make_agent(skills=[make_skill()], llm=NoopLLM())
    assert agent.system_prompt == "You are a helpful agent."


@pytest.mark.unit
async def test_llm_receives_skills_in_system_message(make_skill, make_agent):
    """The discovery section must reach the LLM as the first message."""
    received: list = []

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            received.extend(messages)
            return DummyResponse(Message(role="assistant", content="ok"))

    skill = make_skill(name="pdf-tool", description="Handles PDFs.")
    agent = make_agent(skills=[skill], llm=RecordingLLM())
    state = State(messages=[Message(role="user", content="go")])
    [_ async for _ in agent.stream(state, max_steps=1)]

    system_content = received[0].content
    assert "## Available Skills" in system_content
    assert "pdf-tool" in system_content
    assert "Handles PDFs." in system_content


@pytest.mark.unit
async def test_skill_activation_flow(make_skill, make_agent):
    """
    Two-step flow:
      1. LLM calls use_skill("my-skill") to load instructions.
      2. LLM produces a final answer using the injected instructions.

    Asserts that the tool result contains the skill's instructions.
    """
    skill = make_skill(
        name="my-skill",
        instructions="Step 1: do the thing. Step 2: profit.",
    )

    class TwoStepLLM(ChatLLM):
        def __init__(self):
            self.calls = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.calls += 1
            if self.calls == 1:
                return DummyResponse(
                    message=Message(role="assistant", content=""),
                    tool_calls=[
                        _tool_call("call-1", "use_skill", {"skill_name": "my-skill"})
                    ],
                )
            return DummyResponse(
                message=Message(role="assistant", content="done"),
                tool_calls=[],
            )

    agent = make_agent(skills=[skill], llm=TwoStepLLM())
    state = State(messages=[Message(role="user", content="Help me with my-skill.")])
    events = [e async for e in agent.stream(state, max_steps=5)]

    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_result_events) == 1
    assert skill.instructions in tool_result_events[0].content

    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert len(final_events) == 1
    assert final_events[0].content == "done"
