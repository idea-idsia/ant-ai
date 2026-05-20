from __future__ import annotations

import pytest

from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.llm.protocol import ChatLLM


class DummyResponse:
    """Duck-typed stand-in for ChatLLMResponse."""

    def __init__(self, message: Message):
        self.message = message
        self.tool_calls = []


@pytest.mark.unit
def test_no_skills_system_message_unchanged(make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    agent = make_agent(skills=None, llm=NoopLLM())
    assert agent.system_message.content == "You are a helpful agent."


@pytest.mark.unit
def test_skills_present_appends_discovery_section(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    skills_path = make_skill(name="pdf-tool", description="Handles PDFs.")
    agent = make_agent(skills=skills_path, llm=NoopLLM())
    content = agent.system_message.content
    assert "## Skills System" in content
    assert "pdf-tool" in content
    assert "Handles PDFs." in content
    assert "SKILL.md" in content


@pytest.mark.unit
def test_system_prompt_field_unchanged_by_skills(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    skills_path = make_skill()
    agent = make_agent(skills=skills_path, llm=NoopLLM())
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

    skills_path = make_skill(name="pdf-tool", description="Handles PDFs.")
    agent = make_agent(skills=skills_path, llm=RecordingLLM())
    state = State(messages=[Message(role="user", content="go")])
    [_ async for _ in agent.stream(state, max_steps=1)]

    system_content = received[0].content
    assert "## Skills System" in system_content
    assert "pdf-tool" in system_content
    assert "Handles PDFs." in system_content


@pytest.mark.unit
def test_skill_allowed_tools_shown_in_system_message(make_skill, make_agent):
    class NoopLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse(Message(role="assistant", content="ok"))

    skills_path = make_skill(
        name="git-tool",
        description="Git operations.",
        instructions="Use git.",
        allowed_tools=["Bash(git:*)", "Read"],
    )
    agent = make_agent(skills=skills_path, llm=NoopLLM())
    content = agent.system_message.content
    assert "Bash(git:*)" in content
    assert "Read" in content
