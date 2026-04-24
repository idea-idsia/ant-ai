from __future__ import annotations

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.core.events import FinalAnswerEvent
from ant_ai.core.message import Message
from ant_ai.core.result import LLMOutput, StepResult, Transition, TransitionAction
from ant_ai.core.types import State
from ant_ai.hooks import (
    AgentHook,
    PostModelBlock,
    PostModelFallback,
    PostModelPass,
    PostModelRetry,
)
from ant_ai.llm.protocol import ChatLLM


class DummyResponse:
    def __init__(self, content: str):
        self.message = Message(role="assistant", content=content)
        self.tool_calls = []


def _agent(llm, hooks=()):
    return Agent(
        name="test-agent",
        system_prompt="sys",
        llm=llm,
        tools=[],
        hooks=list(hooks),
    )


def _state(*contents: str) -> State:
    s = State()
    for c in contents:
        s.add_message(Message(role="user", content=c))
    return s


class SimpleLLM(ChatLLM):
    def __init__(self, response: str = "hello"):
        self._response = response

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        return DummyResponse(self._response)


# ── wrap_model_call tests (replaces old middleware tests) ────────────────────


@pytest.mark.unit
async def test_passthrough_hook_does_not_affect_output():
    class PassthroughHook(AgentHook):
        async def wrap_model_call(self, call_next, state, ctx):
            async for item in call_next(state, ctx):
                yield item

    agent = _agent(SimpleLLM("hello"), hooks=[PassthroughHook()])
    events = [e async for e in agent.stream(_state("hi"), max_steps=1)]

    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "hello"


@pytest.mark.unit
async def test_before_model_hook_can_inject_state():
    received_contents: list[str] = []

    class ContextInjectorHook(AgentHook):
        async def before_model(self, state, ctx):
            state.add_message(Message(role="system", content="[injected context]"))

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            received_contents.extend(m.content for m in messages)
            return DummyResponse("ok")

    agent = _agent(RecordingLLM(), hooks=[ContextInjectorHook()])
    await agent.ainvoke(_state("user question"))

    assert "[injected context]" in received_contents


@pytest.mark.unit
async def test_before_agent_hook_can_truncate_history():
    class SlidingWindowHook(AgentHook):
        def __init__(self, max_messages: int):
            self.max_messages = max_messages

        async def before_agent(self, state, ctx):
            if len(state.messages) > self.max_messages:
                state.messages = state.messages[-self.max_messages :]

    received_counts: list[int] = []

    class CountingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            received_counts.append(len(messages))
            return DummyResponse("done")

    agent = _agent(CountingLLM(), hooks=[SlidingWindowHook(max_messages=2)])

    # State with 5 user messages — hook trims to last 2
    state: State = _state("m1", "m2", "m3", "m4", "m5")
    await agent.ainvoke(state)

    # LLMStep prepends the system message, so LLM sees 2 (trimmed) + 1 (system) = 3
    assert received_counts[0] == 3


@pytest.mark.unit
async def test_wrap_model_call_stacking_order_is_outer_to_inner():
    log: list[str] = []

    class TracingHook(AgentHook):
        def __init__(self, label: str):
            self.label = label

        async def wrap_model_call(self, call_next, state, ctx):
            log.append(f"{self.label}:before")
            async for item in call_next(state, ctx):
                yield item
            log.append(f"{self.label}:after")

    class LoggingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            log.append("llm")
            return DummyResponse("done")

    agent = _agent(
        LoggingLLM(),
        hooks=[TracingHook("A"), TracingHook("B")],
    )
    await agent.ainvoke(_state("hi"))

    assert log == ["A:before", "B:before", "llm", "B:after", "A:after"]


@pytest.mark.unit
async def test_wrap_model_call_can_short_circuit_llm():
    """A hook that yields a StepResult directly prevents the LLM from being called."""
    llm_called = False

    cached = StepResult(
        output=LLMOutput(raw="cached response"),
        transition=Transition(action=TransitionAction.END),
    )

    class CacheHook(AgentHook):
        async def wrap_model_call(self, call_next, state, ctx):
            yield cached  # StepResult — skip the LLM entirely

    class ShouldNotBeCalled(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal llm_called
            llm_called = True
            return DummyResponse("should not reach here")

    agent = _agent(ShouldNotBeCalled(), hooks=[CacheHook()])
    events = [e async for e in agent.stream(_state("hi"), max_steps=1)]

    assert not llm_called
    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "cached response"


@pytest.mark.unit
async def test_after_model_pass_returns_result():
    class PassHook(AgentHook):
        async def after_model(self, result, ctx):
            return PostModelPass(result=result)

    agent = _agent(SimpleLLM("fine"), hooks=[PassHook()])
    answer = await agent.ainvoke(_state("hi"))
    assert answer == "fine"


@pytest.mark.unit
async def test_after_model_block_raises_hook_blocked_error():
    from ant_ai.core.exceptions import HookBlockedError

    class BlockHook(AgentHook):
        async def after_model(self, result, ctx):
            return PostModelBlock(reason="blocked by policy")

    agent = _agent(SimpleLLM("bad"), hooks=[BlockHook()])
    s = _state("hi")

    with pytest.raises(HookBlockedError):
        await agent.ainvoke(s)


@pytest.mark.unit
async def test_after_model_retry_injects_critique_and_retries():
    CRITIQUE = "CRITIQUE_MARKER"
    received_on_retry: list[str] = []
    call_count = 0

    class OnceRetryHook(AgentHook):
        def __init__(self):
            self._checked = False

        async def after_model(self, result, ctx):
            if not self._checked:
                self._checked = True
                return PostModelRetry(reason=CRITIQUE)
            return PostModelPass(result=result)

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                received_on_retry.extend(m.content for m in messages)
            return DummyResponse("response")

    agent = _agent(RecordingLLM(), hooks=[OnceRetryHook()])
    await agent.ainvoke(_state("question"))

    assert call_count == 2
    assert any(CRITIQUE in c for c in received_on_retry)


@pytest.mark.unit
async def test_post_model_fallback_returns_safe_answer():
    """PostModelFallback returned from after_model returns the pre-built result without retrying."""
    fallback = StepResult(
        output=LLMOutput(raw="safe fallback"),
        transition=Transition(action=TransitionAction.END),
    )
    llm_call_count = 0

    class FallbackHook(AgentHook):
        async def after_model(self, result, ctx):
            return PostModelFallback(result=fallback)

    class CountingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal llm_call_count
            llm_call_count += 1
            return DummyResponse("original response")

    agent = _agent(CountingLLM(), hooks=[FallbackHook()])
    answer = await agent.ainvoke(_state("hi"))

    assert answer == "safe fallback"
    assert llm_call_count == 1  # no retry — fallback bypasses LLM


# ── lifecycle ordering tests ─────────────────────────────────────────────────


@pytest.mark.unit
async def test_before_after_agent_fire_exactly_once():
    log: list[str] = []

    class LifecycleHook(AgentHook):
        async def before_agent(self, state, ctx):
            log.append("before_agent")

        async def after_agent(self, state, ctx):
            log.append("after_agent")

    agent = _agent(SimpleLLM("hi"), hooks=[LifecycleHook()])
    await agent.ainvoke(_state("question"))

    assert log.count("before_agent") == 1
    assert log.count("after_agent") == 1


@pytest.mark.unit
async def test_before_model_fires_per_loop_iteration(monkeypatch):
    """before_model fires once per LLM call (outer loop iteration)."""
    before_model_calls = 0

    class CountingHook(AgentHook):
        async def before_model(self, state, ctx):
            nonlocal before_model_calls
            before_model_calls += 1

    from ant_ai.core.message import ToolCall, ToolFunction

    tool_call = ToolCall(id="c1", function=ToolFunction(name="my_tool", arguments="{}"))

    call_count = 0

    class TwoStepLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                r = DummyResponse("")
                r.tool_calls = [tool_call]
                return r
            return DummyResponse("done")

    from pydantic import model_serializer

    from ant_ai.tools.tool import Tool

    class DummyTool(Tool):
        name: str = "my_tool"
        description: str = "test"
        parameters: dict = {}

        async def ainvoke(self, **kwargs):
            return "ok"

        @model_serializer
        def _serialize(self):
            return {"name": self.name, "description": self.description}

    agent = Agent(
        name="test",
        system_prompt="sys",
        llm=TwoStepLLM(),
        tools=[DummyTool()],
        hooks=[CountingHook()],
    )
    await agent.ainvoke(_state("go"), max_steps=5)

    # Two LLM calls: first returns tool call, second returns final answer
    assert before_model_calls == 2


@pytest.mark.unit
async def test_before_model_not_called_on_retry():
    """before_model fires exactly once per outer loop step; a retry does not trigger it again."""
    before_model_calls = 0
    after_model_calls = 0

    class TrackingHook(AgentHook):
        async def before_model(self, state, ctx):
            nonlocal before_model_calls
            before_model_calls += 1

        async def after_model(self, result, ctx):
            nonlocal after_model_calls
            after_model_calls += 1
            if after_model_calls == 1:
                return PostModelRetry(reason="first attempt rejected")
            return PostModelPass(result=result)

    class SimpleLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            return DummyResponse("response")

    agent = _agent(SimpleLLM(), hooks=[TrackingHook()])
    await agent.ainvoke(_state("question"), max_steps=1)

    # One outer loop step → before_model fires once only
    assert before_model_calls == 1
    # after_model fires twice: once to trigger retry, once to accept the result
    assert after_model_calls == 2
