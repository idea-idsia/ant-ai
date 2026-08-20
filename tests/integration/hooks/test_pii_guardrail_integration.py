from __future__ import annotations

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.hooks.integrations.pii_guardrail import PIIGuardrailHook
from ant_ai.llm.protocol import ChatLLM


class _DummyResponse:
    def __init__(self, content: str):
        self.message = Message(role="assistant", content=content)
        self.tool_calls = []


def _state(content: str) -> State:
    s = State()
    s.add_message(Message(role="user", content=content))
    return s


@pytest.mark.integration
async def test_agent_retries_then_succeeds_once_pii_is_removed():
    """Real datafog scan, driven through the real Agent retry loop."""

    class LeaksThenFixesLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            if self.call_count == 1:
                return _DummyResponse("Contact me at foo@bar.com")
            return _DummyResponse("Please use the contact form on our website.")

    llm = LeaksThenFixesLLM()
    agent = Agent(
        name="pii-safe-agent",
        system_prompt="You are a helpful assistant. Never share contact details.",
        llm=llm,
        hooks=[PIIGuardrailHook()],
        max_retries=2,
    )

    answer = await agent.ainvoke(_state("How can I reach support?"))

    assert llm.call_count == 2
    assert "foo@bar.com" not in answer


@pytest.mark.integration
async def test_agent_exhausts_retries_when_pii_persists():
    class AlwaysLeaksLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return _DummyResponse("Contact me at foo@bar.com")

    agent = Agent(
        name="pii-unsafe-agent",
        system_prompt="You are a helpful assistant.",
        llm=AlwaysLeaksLLM(),
        hooks=[PIIGuardrailHook()],
        max_retries=1,
    )

    with pytest.raises(HookMaxRetriesError):
        await agent.ainvoke(_state("How can I reach support?"))


@pytest.mark.integration
async def test_block_mode_raises_immediately_without_retry():
    from ant_ai.core.exceptions import HookBlockedError

    class LeaksLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            return _DummyResponse("Contact me at foo@bar.com")

    llm = LeaksLLM()
    agent = Agent(
        name="pii-blocking-agent",
        system_prompt="You are a helpful assistant.",
        llm=llm,
        hooks=[PIIGuardrailHook(on_detect="block")],
        max_retries=2,
    )

    with pytest.raises(HookBlockedError):
        await agent.ainvoke(_state("How can I reach support?"))
    assert llm.call_count == 1
