from __future__ import annotations

import pytest
from pydantic import BaseModel

from ant_ai.agent.loop.react import (
    FinalResponse,
    ReActLoop,
    ToolRequest,
)
from ant_ai.core.events import (
    ContentDeltaEvent,
    Event,
    EventOrigin,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
)
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import Message, ToolCall, ToolCallMessage, ToolFunction
from ant_ai.core.result import (
    ClarificationNeededOutput,
    LLMOutput,
    StepResult,
    ToolOutput,
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

    stream = run  # same fixed replay regardless of which method the loop calls

    def model_copy(self, update: dict | None = None):
        return self


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
    streaming: bool = False,
) -> ReActLoop:
    """Construct a ReActLoop bypassing Pydantic validation."""
    return ReActLoop.model_construct(
        reason_step=reason_step,
        act_step=act_step,
        hooks=hooks if hooks is not None else HookLayer(),
        max_retries=max_retries,
        streaming=streaming,
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


@pytest.mark.unit
async def test_max_steps_with_coerce_schema_forces_structured_final_answer():
    """When max_steps is exhausted with response_schema set and tools present, the last
    iteration strips tools and applies response_format so the LLM synthesizes a structured
    final answer — FinalAnswerEvent is emitted and state gets an assistant message."""
    tool_call = ToolCall(id="c1", function=ToolFunction(name="t", arguments="{}"))
    tool_llm_result = make_llm_result(
        "calling tool", tool_calls=(tool_call,), action=TransitionAction.CONTINUE
    )
    structured_json = '{"value": "done"}'
    forced_final_result = make_llm_result(structured_json)

    class ForcedFinalReasonStep:
        """Yields tool calls on normal runs; returns structured JSON when model_copy'd."""

        name = "llm"

        async def run(self, state, ctx):
            yield tool_llm_result

        def model_copy(self, *, update=None, **_):
            return FakeStep("llm_forced", [forced_final_result])

    act_result = StepResult(
        output=ToolOutput(
            results=({"name": "t", "tool_call_id": "c1", "content": "ok"},)
        )
    )
    act_step = FakeStep("tool", [act_result])

    class MySchema(BaseModel):
        value: str

    loop = make_loop(ForcedFinalReasonStep(), act_step=act_step)
    state = State(messages=[Message(role="user", content="go")])

    events = [
        e
        async for e in loop.stream(
            state, ctx=None, max_steps=2, response_schema=MySchema
        )
    ]

    final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
    assert len(final_events) == 1
    assert MySchema.model_validate_json(final_events[0].content).value == "done"

    # The loop completed with a structured answer — MaxStepsReachedEvent must not fire.
    assert not any(isinstance(e, MaxStepsReachedEvent) for e in events)

    assistant_msgs = [
        m
        for m in state.messages
        if m.role == "assistant" and not isinstance(m, ToolCallMessage)
    ]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == structured_json


@pytest.mark.unit
async def test_max_steps_without_coerce_schema_still_emits_max_steps_reached():
    """Without response_schema, exhausting max_steps still emits MaxStepsReachedEvent
    and does not add a spurious assistant message to state."""
    tool_call = ToolCall(id="c2", function=ToolFunction(name="t", arguments="{}"))
    tool_llm_result = make_llm_result(
        "calling tool", tool_calls=(tool_call,), action=TransitionAction.CONTINUE
    )
    act_result = StepResult(
        output=ToolOutput(
            results=({"name": "t", "tool_call_id": "c2", "content": "ok"},)
        )
    )
    act_step = FakeStep("tool", [act_result])

    class AlwaysToolStep:
        name = "llm"

        async def run(self, state, ctx):
            yield tool_llm_result

        def model_copy(self, *, update=None, **_):
            return FakeStep("llm_copy", [tool_llm_result])

    loop = make_loop(AlwaysToolStep(), act_step=act_step)
    state = State(messages=[Message(role="user", content="go")])

    events = [e async for e in loop.stream(state, ctx=None, max_steps=2)]

    assert any(isinstance(e, MaxStepsReachedEvent) for e in events)
    assert not any(isinstance(e, FinalAnswerEvent) for e in events)
    assert not any(
        m.role == "assistant" and not isinstance(m, ToolCallMessage)
        for m in state.messages
    )


@pytest.mark.unit
async def test_streaming_active_forwards_content_deltas_live():
    """With streaming=True and no unsafe hooks, the loop calls .stream() (not
    .run()), ContentDeltaEvents are forwarded, and the final answer content
    matches what a buffered run would have produced."""
    calls: list[str] = []

    class RecordingStep:
        name = "llm"

        async def run(self, state, ctx):
            calls.append("run")
            yield make_llm_result("Hello")

        async def stream(self, state, ctx):
            calls.append("stream")
            yield ContentDeltaEvent(delta="Hel", stream_id="s1", is_first=True)
            yield ContentDeltaEvent(delta="lo", stream_id="s1")
            # LLMStep.stream() yields its own FinalAnswerEvent (with
            # stream_id set) before the StepResult, matching production.
            yield FinalAnswerEvent(content="Hello", stream_id="s1")
            yield make_llm_result("Hello")

        def model_copy(self, *, update=None, **_):
            return self

    loop = make_loop(RecordingStep(), streaming=True)
    state = State(messages=[Message(role="user", content="hi")])

    events = [e async for e in loop.stream(state, ctx=None)]

    assert calls == ["stream"]
    deltas = [e for e in events if isinstance(e, ContentDeltaEvent)]
    assert [d.delta for d in deltas] == ["Hel", "lo"]

    final = next(e for e in events if isinstance(e, FinalAnswerEvent))
    assert final.content == "Hello"
    # regression: ReActLoop rebuilds the terminal FinalAnswerEvent itself
    # (to support schema coercion) -- it must carry over stream_id from the
    # step's own FinalAnswerEvent so A2A can close the artifact.
    assert final.stream_id == "s1"


@pytest.mark.unit
async def test_streaming_falls_back_to_buffered_when_hook_is_unsafe():
    """A hook overriding after_model makes is_stream_safe() False, so even with
    streaming=True the loop must take the fully-buffered path unchanged."""
    reason_step = FakeStep(
        "llm",
        [
            ContentDeltaEvent(delta="Hel", stream_id="s1", is_first=True),
            make_llm_result("Hello"),
        ],
    )
    loop = make_loop(reason_step, hooks=HookLayer(hooks=[_PassHook()]), streaming=True)
    state = State(messages=[Message(role="user", content="hi")])

    events = [e async for e in loop.stream(state, ctx=None)]

    final = next(e for e in events if isinstance(e, FinalAnswerEvent))
    assert final.content == "Hello"
    # _PassHook overrides after_model, so is_stream_safe() is False and the
    # loop must have gone through _consume_wrapped/_apply_hooks, not
    # _stream_wrapped -- deltas still pass through unchanged either way since
    # ReActLoop forwards any non-FinalAnswerEvent item regardless of path.
    deltas = [e for e in events if isinstance(e, ContentDeltaEvent)]
    assert [d.delta for d in deltas] == ["Hel"]


@pytest.mark.unit
async def test_coerce_schema_final_step_never_streams():
    """The structured-output coercion branch must always call .run(), never
    .stream(), even when the loop is streaming-active, since coercion may
    silently rewrite the raw text after the fact."""
    calls: list[str] = []

    class RecordingStep:
        name = "llm"

        async def run(self, state, ctx):
            calls.append("run")
            yield make_llm_result('{"x": 1}')

        async def stream(self, state, ctx):
            calls.append("stream")
            yield make_llm_result('{"x": 1}')

        def model_copy(self, *, update=None, **_):
            return self

    class DummySchema(BaseModel):
        x: int

    # act_step must be non-None for the loop to treat response_schema as
    # coerce_schema (applied on the final step) rather than an eager override.
    loop = make_loop(RecordingStep(), act_step=FakeStep("tool", []), streaming=True)
    state = State(messages=[Message(role="user", content="hi")])

    events = [
        e
        async for e in loop.stream(
            state, ctx=None, max_steps=1, response_schema=DummySchema
        )
    ]

    assert any(isinstance(e, FinalAnswerEvent) for e in events)
    assert calls == ["run"]


@pytest.mark.unit
async def test_stream_wrapped_raises_if_hook_lies_about_stream_safety():
    """If a hook's after_model somehow returns a non-Pass decision on the
    streaming path, _stream_wrapped must raise loudly rather than silently
    drop or duplicate already-forwarded tokens."""
    reason_step = FakeStep("llm", [make_llm_result("Hello")])
    loop = make_loop(reason_step, hooks=HookLayer(hooks=[_AlwaysRetryHook()]))
    state = State(messages=[Message(role="user", content="hi")])
    wrapped = loop.hooks.wrap_model_call(reason_step.run)

    with pytest.raises(RuntimeError, match="stream-safe"):
        async for _ in loop._stream_wrapped(reason_step, wrapped, state, ctx=None):
            pass
