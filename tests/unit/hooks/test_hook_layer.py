from __future__ import annotations

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.exceptions import HookBlockedError
from ant_ai.core.message import Message
from ant_ai.core.result import (
    LLMOutput,
    StepResult,
    Transition,
    TransitionAction,
)
from ant_ai.core.types import State
from ant_ai.hooks import (
    AgentHook,
    HookLayer,
    PostModelBlock,
    PostModelPass,
    PostModelRetry,
)
from ant_ai.llm.protocol import ChatLLM


def _llm_result(raw: str) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=raw),
        transition=Transition(action=TransitionAction.END),
    )


class _RetryHook(AgentHook):
    def __init__(self, reason: str):
        self._reason = reason

    async def after_model(self, result, ctx):
        return PostModelRetry(reason=self._reason)


class _BlockHook(AgentHook):
    async def after_model(self, result, ctx):
        return PostModelBlock(reason="blocked")


@pytest.mark.unit
async def test_merges_reasons_from_all_failing_hooks():
    layer = HookLayer(
        hooks=[
            _RetryHook("reason-alpha"),
            _RetryHook("reason-beta"),
        ]
    )
    decision = await layer.run_after_model(_llm_result("anything"), ctx=None)
    assert isinstance(decision, PostModelRetry)
    assert "reason-alpha" in decision.reason
    assert "reason-beta" in decision.reason


@pytest.mark.unit
async def test_block_wins_over_retry():
    layer = HookLayer(
        hooks=[
            _RetryHook("soft fail"),
            _BlockHook(),
        ]
    )
    decision = await layer.run_after_model(_llm_result("x"), ctx=None)
    assert isinstance(decision, PostModelBlock)


@pytest.mark.unit
async def test_empty_layer_always_passes():
    layer = HookLayer()
    result = _llm_result("anything")
    decision = await layer.run_after_model(result, ctx=None)
    assert isinstance(decision, PostModelPass)


@pytest.mark.unit
async def test_block_verdict_raises_immediately_without_retry():
    call_count = 0

    class BlockHook(AgentHook):
        async def after_model(self, result, ctx):
            return PostModelBlock(reason="policy")

    class CountingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            return _DummyResponse("some response")

    class _DummyResponse:
        def __init__(self, content):
            self.message = Message(role="assistant", content=content)
            self.tool_calls = []

    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=CountingLLM(),
        tools=[],
        hooks=[BlockHook()],
    )
    s = State()
    s.add_message(Message(role="user", content="hi"))

    with pytest.raises(HookBlockedError):
        await agent.ainvoke(s)

    assert call_count == 1  # no retry


@pytest.mark.unit
async def test_wrap_model_call_chain_is_outermost_first():
    order: list[str] = []

    class OrderedHook(AgentHook):
        def __init__(self, label: str):
            self.label = label

        async def wrap_model_call(self, call_next, state, ctx):
            order.append(f"{self.label}:enter")
            async for item in call_next(state, ctx):
                yield item
            order.append(f"{self.label}:exit")

    from ant_ai.core.result import LLMOutput, StepResult, Transition, TransitionAction

    async def core(state, ctx):
        yield StepResult(
            output=LLMOutput(raw="x"),
            transition=Transition(action=TransitionAction.END),
        )

    layer = HookLayer(hooks=[OrderedHook("A"), OrderedHook("B")])
    wrapped = layer.wrap_model_call(core)

    async for _ in wrapped(None, None):
        pass

    assert order == ["A:enter", "B:enter", "B:exit", "A:exit"]


@pytest.mark.unit
def test_is_stream_safe_true_for_empty_layer():
    assert HookLayer().is_stream_safe() is True


@pytest.mark.unit
def test_is_stream_safe_true_for_hook_overriding_unrelated_method():
    class BeforeAgentHook(AgentHook):
        async def before_agent(self, state, ctx):
            pass

    assert HookLayer(hooks=[BeforeAgentHook()]).is_stream_safe() is True


@pytest.mark.unit
def test_is_stream_safe_false_for_hook_overriding_after_model():
    assert HookLayer(hooks=[_RetryHook("x")]).is_stream_safe() is False


@pytest.mark.unit
def test_is_stream_safe_false_for_hook_overriding_wrap_model_call():
    class WrapHook(AgentHook):
        async def wrap_model_call(self, call_next, state, ctx):
            async for item in call_next(state, ctx):
                yield item

    assert HookLayer(hooks=[WrapHook()]).is_stream_safe() is False
