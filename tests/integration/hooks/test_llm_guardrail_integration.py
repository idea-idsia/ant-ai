from __future__ import annotations

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.hooks.builtins.llm_guardrail import LLMGuardrailHook
from ant_ai.llm.protocol import ChatLLM


class _DummyResponse:
    def __init__(self, content: str):
        self.message = Message(role="assistant", content=content)
        self.tool_calls = []


class _JudgeLLM(ChatLLM):
    """Fails whenever the judged text (not the criteria) contains the forbidden marker."""

    def __init__(self):
        self.call_count = 0
        self.seen_messages: list[list[Message]] = []

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        self.call_count += 1
        self.seen_messages.append(list(messages))
        # Only the last message carries the candidate text ("Text to judge:\n<raw>");
        # the criteria (in an earlier message) also mentions "banned", so checking the
        # full joined prompt would always fail regardless of the candidate text.
        judged_text = messages[-1].content or ""
        if "FORBIDDEN_MARKER" in judged_text:
            return _DummyResponse('{"passed": false, "reason": "contains banned word"}')
        return _DummyResponse('{"passed": true}')


def _state(content: str) -> State:
    s = State()
    s.add_message(Message(role="user", content=content))
    return s


@pytest.mark.integration
async def test_agent_retries_then_succeeds_once_judge_approves():
    class SaysBannedThenCleanLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            if self.call_count == 1:
                return _DummyResponse("this contains a FORBIDDEN_MARKER word")
            return _DummyResponse("this is a clean answer")

    main_llm = SaysBannedThenCleanLLM()
    judge = _JudgeLLM()
    agent = Agent(
        name="judged-agent",
        system_prompt="You are a helpful assistant.",
        llm=main_llm,
        hooks=[
            LLMGuardrailHook(judge_llm=judge, criteria="must not contain banned words")
        ],
        max_retries=2,
    )

    answer = await agent.ainvoke(_state("say something"))

    assert main_llm.call_count == 2
    assert judge.call_count == 2
    assert answer == "this is a clean answer"


@pytest.mark.integration
async def test_agent_exhausts_retries_when_judge_never_approves():
    class AlwaysBannedLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return _DummyResponse("this contains a FORBIDDEN_MARKER word")

    agent = Agent(
        name="judged-agent-fails",
        system_prompt="You are a helpful assistant.",
        llm=AlwaysBannedLLM(),
        hooks=[
            LLMGuardrailHook(
                judge_llm=_JudgeLLM(), criteria="must not contain banned words"
            )
        ],
        max_retries=1,
    )

    with pytest.raises(HookMaxRetriesError):
        await agent.ainvoke(_state("say something"))


@pytest.mark.integration
async def test_block_mode_raises_immediately_without_retry():
    from ant_ai.core.exceptions import HookBlockedError

    class AlwaysBannedLLM(ChatLLM):
        def __init__(self):
            self.call_count = 0

        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            self.call_count += 1
            return _DummyResponse("this contains a FORBIDDEN_MARKER word")

    llm = AlwaysBannedLLM()
    agent = Agent(
        name="judged-blocking-agent",
        system_prompt="You are a helpful assistant.",
        llm=llm,
        hooks=[
            LLMGuardrailHook(
                judge_llm=_JudgeLLM(),
                criteria="must not contain banned words",
                on_fail="block",
            )
        ],
        max_retries=2,
    )

    with pytest.raises(HookBlockedError):
        await agent.ainvoke(_state("say something"))
    assert llm.call_count == 1
