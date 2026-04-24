from __future__ import annotations

import pytest

from ant_ai.agent.loop.react import (
    FinalResponse,
    ReActLoop,
    ToolRequest,
)
from ant_ai.core.events import (
    Event,
    EventOrigin,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
)
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.result import (
    ClarificationNeededOutput,
    LLMOutput,
    StepResult,
    Transition,
    TransitionAction,
)
from ant_ai.core.types import State
from ant_ai.hooks import (
    AgentHook,
    HookLayer,
    PostModelFallback,
    PostModelPass,
    PostModelRetry,
)


class FakeStep:
    """Minimal step duck-type: yields a fixed list of items then stops."""

    def __init__(self, name: str, items: list):
        self._name = name
        self._items = list(items)

    @property
    def name(self) -> str:
        return self._name

    async def run(self, state, ctx):
        for item in self._items:
            yield item


def make_llm_result(
    text: str = "answer",
    *,
    tool_calls: tuple = (),
    action: TransitionAction = TransitionAction.END,
) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=text, tool_calls=tool_calls),
        transition=Transition(action=action),
    )


def make_loop(
    reason_step,
    act_step=None,
    hooks=None,
    max_retries: int = 3,
) -> ReActLoop:
    """Construct a ReActLoop bypassing Pydantic validation."""
    return ReActLoop.model_construct(
        reason_step=reason_step,
        act_step=act_step,
        hooks=hooks if hooks is not None else HookLayer(),
        max_retries=max_retries,
    )


class _PassHook(AgentHook):
    async def after_model(self, result, ctx):
        return PostModelPass(result=result)


class _AlwaysRetryHook(AgentHook):
    async def after_model(self, result, ctx):
        return PostModelRetry(reason="bad")


@pytest.mark.unit
async def test_stream_stops_on_clarification_needed():
    """When act_step returns ClarificationNeededOutput the loop exits without a FinalAnswerEvent."""
    tool_call = ToolCall(id="c1", function=ToolFunction(name="my_tool", arguments="{}"))
    reason_step = FakeStep(
        "llm",
        [
            make_llm_result(
                "calling tool",
                tool_calls=(tool_call,),
                action=TransitionAction.CONTINUE,
            )
        ],
    )
    clarif_result = StepResult(
        output=ClarificationNeededOutput(
            question="Which one?", tool_call_id="c1", tool_name="my_tool"
        )
    )
    act_step = FakeStep("tool", [clarif_result])

    loop: ReActLoop = make_loop(reason_step, act_step=act_step)
    state = State(messages=[Message(role="user", content="go")])

    events = [e async for e in loop.stream(state, ctx=None, max_steps=5)]

    assert not any(isinstance(e, FinalAnswerEvent) for e in events)
    assert not any(isinstance(e, MaxStepsReachedEvent) for e in events)


@pytest.mark.unit
async def test_stream_with_hooks_pass_buffers_and_yields_final_answer():
    """Non-empty hooks that PASS still produce a FinalAnswerEvent via the buffered path."""
    step_event = Event(origin=EventOrigin(layer="agent"), content="thinking")
    reason_step = FakeStep("llm", [step_event, make_llm_result("the answer")])
    hooks = HookLayer(hooks=[_PassHook()])

    loop: ReActLoop = make_loop(reason_step, hooks=hooks)
    state = State(messages=[Message(role="user", content="hello")])

    events = [e async for e in loop.stream(state, ctx=None)]

    assert isinstance(events[-1], FinalAnswerEvent)
    assert events[-1].content == "the answer"


@pytest.mark.unit
async def test_apply_hooks_retry_then_pass():
    """RETRY on first check → re-run step → PASS on second check returns repaired result."""
    call_count = 0

    class RegenThenPassHook(AgentHook):
        async def after_model(self, result, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PostModelRetry(reason="too short")
            return PostModelPass(result=result)

    repaired_result: StepResult = make_llm_result("repaired")

    class RetryStep:
        name = "llm"

        async def run(self, state, ctx):
            yield repaired_result

    retry_step = RetryStep()
    loop: ReActLoop = make_loop(
        reason_step=retry_step,
        hooks=HookLayer(hooks=[RegenThenPassHook()]),
        max_retries=3,
    )

    original_result: StepResult = make_llm_result("short")
    wrapped = loop.hooks.wrap_model_call(retry_step.run)
    events, final_result = await loop._apply_hooks(
        retry_step, wrapped, State(), None, [], original_result
    )

    assert final_result is repaired_result
    assert call_count == 2


@pytest.mark.unit
async def test_apply_hooks_post_model_fallback_return():
    """PostModelFallback returned directly from a hook replaces the result without retry."""
    fallback: StepResult = make_llm_result("safe fallback")

    class FallbackHook(AgentHook):
        async def after_model(self, result, ctx):
            return PostModelFallback(result=fallback)

    loop: ReActLoop = make_loop(
        reason_step=FakeStep("llm", []),
        hooks=HookLayer(hooks=[FallbackHook()]),
        max_retries=3,
    )

    fallback_step = FakeStep("llm", [])
    events, final_result = await loop._apply_hooks(
        fallback_step,
        fallback_step.run,
        State(),
        None,
        [Event()],
        make_llm_result("bad"),
    )

    assert events == []
    assert final_result is fallback


@pytest.mark.unit
async def test_apply_hooks_max_retries_raises_error():
    """Exhausting all retries with a still-failing final check raises HookMaxRetriesError."""

    class RetryStep:
        name = "llm"

        async def run(self, state, ctx):
            yield make_llm_result("still bad")

    retry_step = RetryStep()
    loop: ReActLoop = make_loop(
        reason_step=retry_step,
        hooks=HookLayer(hooks=[_AlwaysRetryHook()]),
        max_retries=1,
    )

    with pytest.raises(HookMaxRetriesError):
        await loop._apply_hooks(
            retry_step,
            loop.hooks.wrap_model_call(retry_step.run),
            State(),
            None,
            [],
            make_llm_result("bad"),
        )


@pytest.mark.unit
async def test_apply_hooks_retry_once_then_pass():
    """Hook retries once then passes — both decisions happen inside the while loop."""
    call_count = 0

    class RegenOnceHook(AgentHook):
        async def after_model(self, result, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PostModelRetry(reason="retry")
            return PostModelPass(result=result)

    repaired: StepResult = make_llm_result("repaired")

    class RetryStep:
        name = "llm"

        async def run(self, state, ctx):
            yield repaired

    retry_step = RetryStep()
    loop: ReActLoop = make_loop(
        reason_step=retry_step,
        hooks=HookLayer(hooks=[RegenOnceHook()]),
        max_retries=1,
    )

    wrapped = loop.hooks.wrap_model_call(retry_step.run)
    events, final_result = await loop._apply_hooks(
        retry_step, wrapped, State(), None, [], make_llm_result("draft")
    )

    assert final_result is repaired
    assert call_count == 2  # once to trigger retry, once to accept repaired result


@pytest.mark.unit
async def test_retry_does_not_call_before_model():
    """_retry_with_critique must NOT call run_before_model.
    before_model is the caller's responsibility (fires once per outer loop step)."""
    before_model_calls = 0

    class CountingHook(AgentHook):
        async def before_model(self, state, ctx):
            nonlocal before_model_calls
            before_model_calls += 1

        async def after_model(self, result, ctx):
            # Retry on first attempt, pass on second.
            if result.output.raw == "draft":
                return PostModelRetry(reason="too short")
            return PostModelPass(result=result)

    repaired = make_llm_result("repaired")

    class RetryStep:
        name = "llm"

        async def run(self, state, ctx):
            yield repaired

    retry_step = RetryStep()
    loop: ReActLoop = make_loop(
        reason_step=retry_step,
        hooks=HookLayer(hooks=[CountingHook()]),
        max_retries=3,
    )
    wrapped = loop.hooks.wrap_model_call(retry_step.run)
    await loop._apply_hooks(
        retry_step, wrapped, State(), None, [], make_llm_result("draft")
    )

    # before_model must not have been called by _apply_hooks / _retry_with_critique
    assert before_model_calls == 0


@pytest.mark.unit
async def test_retry_accumulates_failed_responses_and_critiques():
    """Each retry appends the previous failed response + critique so the LLM
    sees the full correction history, not just the latest critique."""
    seen_states: list[list] = []

    class TrackingRetryStep:
        name = "llm"

        async def run(self, state, ctx):
            seen_states.append(list(state.messages))
            yield make_llm_result("attempt")

    call_count = 0

    class TwoRetryHook(AgentHook):
        async def after_model(self, result, ctx):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return PostModelRetry(reason=f"critique-{call_count}")
            return PostModelPass(result=result)

    retry_step = TrackingRetryStep()
    loop: ReActLoop = make_loop(
        reason_step=retry_step,
        hooks=HookLayer(hooks=[TwoRetryHook()]),
        max_retries=3,
    )

    initial_state = State(messages=[Message(role="user", content="go")])
    wrapped = loop.hooks.wrap_model_call(retry_step.run)
    await loop._apply_hooks(
        retry_step, wrapped, initial_state, None, [], make_llm_result("draft")
    )

    # Retry 1: original user msg + failed draft (assistant) + critique-1 (user)
    assert len(seen_states[0]) == 3
    assert seen_states[0][-2].role == "assistant"
    assert seen_states[0][-2].content == "draft"
    assert "critique-1" in seen_states[0][-1].content

    # Retry 2: previous + failed attempt (assistant) + critique-2 (user)
    assert len(seen_states[1]) == 5
    assert seen_states[1][-2].role == "assistant"
    assert seen_states[1][-2].content == "attempt"
    assert "critique-2" in seen_states[1][-1].content


@pytest.mark.unit
async def test_observe_step_reraises_step_exception():
    """Exceptions raised inside a step generator propagate out of _observe_step."""

    async def failing_gen():
        raise RuntimeError("step exploded")
        yield  # make it an async generator  # noqa: unreachable

    fake_step = FakeStep("boom", [])
    loop: ReActLoop = make_loop(reason_step=fake_step)

    with pytest.raises(RuntimeError, match="step exploded"):
        async for _ in loop._observe_step(fake_step, failing_gen()):
            pass


@pytest.mark.unit
def test_classify_llm_result_tool_request_when_act_step_present():
    """CONTINUE transition + tool calls + act_step configured → ToolRequest."""
    tool_call = ToolCall(id="c1", function=ToolFunction(name="t", arguments="{}"))
    result = make_llm_result(
        "calling", tool_calls=(tool_call,), action=TransitionAction.CONTINUE
    )
    act_step = FakeStep("tool", [])
    loop = make_loop(reason_step=FakeStep("llm", [result]), act_step=act_step)
    assert isinstance(loop._classify_llm_result(result), ToolRequest)


@pytest.mark.unit
def test_classify_llm_result_invalid_step_when_no_act_step():
    """Tool calls present but no act_step configured → RuntimeError (not silent FinalResponse)."""
    tool_call = ToolCall(id="c1", function=ToolFunction(name="t", arguments="{}"))
    result = make_llm_result(
        "calling", tool_calls=(tool_call,), action=TransitionAction.CONTINUE
    )
    loop = make_loop(reason_step=FakeStep("llm", [result]), act_step=None)
    with pytest.raises(RuntimeError, match="no tools are configured"):
        loop._classify_llm_result(result)


@pytest.mark.unit
def test_classify_llm_result_final_response_on_end_transition():
    """END transition → FinalResponse regardless of tool calls."""
    result = make_llm_result("done", action=TransitionAction.END)
    loop = make_loop(reason_step=FakeStep("llm", [result]))
    assert isinstance(loop._classify_llm_result(result), FinalResponse)


@pytest.mark.unit
async def test_final_answer_is_added_to_state():
    """FinalResponse must append an assistant message to state so subsequent turns see the answer."""
    reason_step = FakeStep("llm", [make_llm_result("final answer text")])
    loop = make_loop(reason_step)
    state = State(messages=[Message(role="user", content="hello")])

    events = [e async for e in loop.stream(state, ctx=None)]

    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert len(final_events) == 1
    assert final_events[0].content == "final answer text"

    assistant_messages = [m for m in state.messages if m.role == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "final answer text"


@pytest.mark.unit
async def test_final_answer_state_message_matches_yielded_event():
    """The content stored in state must be identical to the yielded FinalAnswerEvent content."""
    reason_step = FakeStep("llm", [make_llm_result("consistent content")])
    loop = make_loop(reason_step)
    state = State(messages=[Message(role="user", content="q")])

    events = [e async for e in loop.stream(state, ctx=None)]

    final_event = next(e for e in events if isinstance(e, FinalAnswerEvent))
    assistant_msg = next(m for m in state.messages if m.role == "assistant")
    assert final_event.content == assistant_msg.content
