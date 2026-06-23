from __future__ import annotations

import pytest

from ant_ai.agent.loop.react import ReActLoop
from ant_ai.core.events import MaxStepsReachedEvent
from ant_ai.core.message import Message
from ant_ai.core.result import LLMOutput, StepResult, Transition, TransitionAction
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks import HookLayer
from ant_ai.hooks.builtins.history_compression import (
    _SUMMARY_PREFIX,
    HistoryCompressionHook,
)
from ant_ai.memory.protocol import Memory


def make_llm_result(
    text: str = "answer",
    *,
    action: TransitionAction = TransitionAction.END,
) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=text, tool_calls=()),
        transition=Transition(action=action),
    )


class FakeStep:
    def __init__(self, name: str, items: list):
        self._name: str = name
        self._items = list(items)

    @property
    def name(self) -> str:
        return self._name

    async def run(self, state, ctx):
        for item in self._items:
            yield item


def make_loop(reason_step, *, memory: Memory | None = None) -> ReActLoop:
    return ReActLoop.model_construct(
        reason_step=reason_step,
        act_step=None,
        hooks=HookLayer(),
        max_retries=3,
        memory=memory,
    )


@pytest.mark.unit
async def test_retrieve_called_before_first_model_step(stub_memory):
    """retrieve() is called exactly once, before the first LLM invocation."""
    reason_step = FakeStep("llm", [make_llm_result("done")])
    loop: ReActLoop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="user", content="hello")])

    [_ async for _ in loop.stream(state, ctx=None)]

    assert len(stub_memory.retrieve_calls) == 1
    assert stub_memory.retrieve_calls[0]["query"] == "hello"


@pytest.mark.unit
async def test_retrieved_messages_prepended_to_state(stub_memory):
    """Messages returned by retrieve() are inserted at the start of state.messages."""
    memory_msg = Message(role="system", content="User prefers Python")
    stub_memory.set_entries([memory_msg])

    reason_step = FakeStep("llm", [make_llm_result("done")])
    loop: ReActLoop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="user", content="hi")])

    [_ async for _ in loop.stream(state, ctx=None)]

    assert state.messages[0] == memory_msg


@pytest.mark.unit
async def test_retrieve_not_called_when_no_user_message(stub_memory):
    """retrieve() is skipped when state has no user message."""
    reason_step = FakeStep("llm", [make_llm_result("done")])
    loop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="system", content="sys")])

    [_ async for _ in loop.stream(state, ctx=None)]

    assert len(stub_memory.retrieve_calls) == 0


@pytest.mark.unit
async def test_update_called_on_final_response(stub_memory):
    """update() is called once after a FinalResponse with trigger + assistant messages."""
    reason_step = FakeStep("llm", [make_llm_result("final answer")])
    loop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="user", content="q")])

    [_ async for _ in loop.stream(state, ctx=None)]

    assert len(stub_memory.update_calls) == 1
    stored = stub_memory.update_calls[0]["messages"]
    roles = [m.role for m in stored]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.unit
async def test_update_called_on_max_steps_reached(stub_memory):
    """update() is called with at least the trigger message when max steps are exhausted."""
    from ant_ai.core.message import ToolCall, ToolFunction
    from ant_ai.core.result import ToolOutput

    tool_call = ToolCall(
        id="t1", function=ToolFunction(name="loop_tool", arguments="{}")
    )

    class AlwaysToolLLM:
        name = "llm"

        async def run(self, state, ctx):
            yield StepResult(
                output=LLMOutput(raw="calling tool", tool_calls=(tool_call,)),
                transition=Transition(action=TransitionAction.CONTINUE),
            )

    class AlwaysContinueTool:
        name = "tool"

        async def run(self, state, ctx):
            yield StepResult(
                output=ToolOutput(
                    results=[
                        {"name": "loop_tool", "tool_call_id": "t1", "content": "ok"}
                    ]
                ),
                transition=Transition(
                    action=TransitionAction.CONTINUE, next_step="llm"
                ),
            )

    loop = ReActLoop.model_construct(
        reason_step=AlwaysToolLLM(),
        act_step=AlwaysContinueTool(),
        hooks=HookLayer(),
        max_retries=3,
        memory=stub_memory,
    )
    state = State(messages=[Message(role="user", content="loop forever")])

    events = [e async for e in loop.stream(state, ctx=None, max_steps=2)]

    assert any(isinstance(e, MaxStepsReachedEvent) for e in events)
    assert len(stub_memory.update_calls) == 1
    stored = stub_memory.update_calls[0]["messages"]
    assert any(m.role == "user" for m in stored)


@pytest.mark.unit
async def test_no_memory_calls_when_memory_is_none():
    """No retrieve or update calls are made when memory=None."""
    reason_step = FakeStep("llm", [make_llm_result("ok")])
    loop: ReActLoop = make_loop(reason_step, memory=None)
    state = State(messages=[Message(role="user", content="hi")])

    [_ async for _ in loop.stream(state, ctx=None)]
    # No assertions needed — if memory methods were called they'd raise AttributeError


@pytest.mark.unit
async def test_update_only_current_turn_in_multi_turn(stub_memory):
    """In a multi-turn scenario, update() receives only the new exchange, not prior history."""
    reason_step = FakeStep("llm", [make_llm_result("second answer")])
    loop = make_loop(reason_step, memory=stub_memory)

    state = State(
        messages=[
            Message(role="user", content="first question"),
            Message(role="assistant", content="first answer"),
            Message(role="user", content="second question"),
        ]
    )

    [_ async for _ in loop.stream(state, ctx=None)]

    assert len(stub_memory.update_calls) == 1
    stored_contents = [m.content for m in stub_memory.update_calls[0]["messages"]]
    assert "first question" not in stored_contents
    assert "first answer" not in stored_contents
    assert "second question" in stored_contents
    assert "second answer" in stored_contents


@pytest.mark.unit
async def test_kwargs_forwarded_to_retrieve_and_update(stub_memory):
    """**kwargs passed via session context are forwarded to retrieve and update."""
    reason_step = FakeStep("llm", [make_llm_result("ans")])
    loop: ReActLoop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="user", content="q")])
    ctx = InvocationContext(session_id="sess-123")

    [_ async for _ in loop.stream(state, ctx=ctx)]

    assert stub_memory.retrieve_calls[0].get("ctx") is ctx
    assert stub_memory.update_calls[0].get("ctx") is ctx


@pytest.mark.unit
async def test_update_called_when_consumer_breaks_after_final_event(stub_memory):
    """update() must fire even if the caller stops iterating after FinalAnswerEvent."""
    from ant_ai.core.events import FinalAnswerEvent

    reason_step = FakeStep("llm", [make_llm_result("the answer")])
    loop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="user", content="q")])

    # Simulate a consumer that exits as soon as it sees FinalAnswerEvent —
    # the regression was that memory.update() was placed after the yield,
    # so an early-exit caller would never trigger it.
    async for event in loop.stream(state, ctx=None):
        if isinstance(event, FinalAnswerEvent):
            break

    assert len(stub_memory.update_calls) == 1, (
        "update() must be called before FinalAnswerEvent is yielded so that "
        "consumers that break early still persist memory"
    )


@pytest.mark.unit
async def test_update_called_before_final_event_is_yielded(stub_memory):
    """Memory consolidation happens before FinalAnswerEvent is emitted."""
    from ant_ai.core.events import FinalAnswerEvent

    update_call_count_at_event: list[int] = []

    reason_step = FakeStep("llm", [make_llm_result("ans")])
    loop = make_loop(reason_step, memory=stub_memory)
    state = State(messages=[Message(role="user", content="q")])

    async for event in loop.stream(state, ctx=None):
        if isinstance(event, FinalAnswerEvent):
            update_call_count_at_event.append(len(stub_memory.update_calls))

    assert update_call_count_at_event == [1], (
        "update() must have been called before FinalAnswerEvent reaches the consumer"
    )


@pytest.mark.unit
async def test_retrieve_called_only_once_across_multiple_loop_steps(stub_memory):
    """retrieve() fires before the first model step only — not on subsequent steps."""
    from ant_ai.core.message import ToolCall, ToolFunction
    from ant_ai.core.result import ToolOutput

    tool_call = ToolCall(id="t1", function=ToolFunction(name="my_tool", arguments="{}"))
    tool_result = StepResult(
        output=ToolOutput(
            results=[{"name": "my_tool", "tool_call_id": "t1", "content": "ok"}]
        ),
        transition=Transition(action=TransitionAction.CONTINUE, next_step="llm"),
    )

    call_count = 0

    class TwoStepLLM:
        name = "llm"

        async def run(self, state, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StepResult(
                    output=LLMOutput(raw="calling tool", tool_calls=(tool_call,)),
                    transition=Transition(action=TransitionAction.CONTINUE),
                )
            else:
                yield make_llm_result("done")

    class FakeToolStep:
        name = "tool"

        async def run(self, state, ctx):
            yield tool_result

    loop: ReActLoop = ReActLoop.model_construct(
        reason_step=TwoStepLLM(),
        act_step=FakeToolStep(),
        hooks=HookLayer(),
        max_retries=3,
        memory=stub_memory,
    )
    state = State(messages=[Message(role="user", content="use a tool")])

    [_ async for _ in loop.stream(state, ctx=None)]

    assert len(stub_memory.retrieve_calls) == 1


# ── History compression + memory together ────────────────────────────────────


class _CapturingLLM:
    """Minimal async LLM used to drive HistoryCompressionHook in unit tests."""

    def __init__(self, summary: str = "summary"):
        self.calls: list[list[Message]] = []
        self._summary = summary

    async def ainvoke(self, messages, **_):
        self.calls.append(list(messages))

        class _R:
            pass

        r = _R()
        r.message = Message(role="assistant", content=self._summary)
        return r


def _make_loop_with_compression(
    reason_step,
    *,
    memory,
    max_messages: int,
    keep_last: int,
    summary: str = "SUMMARY",
) -> tuple[ReActLoop, _CapturingLLM]:
    """Build a ReActLoop that has both a HistoryCompressionHook and a Memory."""
    cap = _CapturingLLM(summary)
    hook = HistoryCompressionHook(
        llm=cap, max_messages=max_messages, keep_last=keep_last
    )
    loop = ReActLoop.model_construct(
        reason_step=reason_step,
        act_step=None,
        hooks=HookLayer(hooks=[hook]),
        max_retries=3,
        memory=memory,
    )
    return loop, cap


@pytest.mark.unit
async def test_retrieved_memories_count_toward_compression_threshold(stub_memory):
    """Memory messages injected by retrieve() push the total over the threshold.

    retrieve() prepends messages to state *before* before_model fires.
    If those messages cause len(state.messages) >= max_messages, compression
    should fire even though the initial state alone was below the threshold.
    """
    reason_step = FakeStep("llm", [make_llm_result("answer")])
    stub_memory.set_entries(
        [
            Message(role="system", content="MEMORY_A"),
            Message(role="system", content="MEMORY_B"),
        ]
    )

    # State has 1 user message; memory adds 2 → total 3 = max_messages → trigger
    loop, cap = _make_loop_with_compression(
        reason_step, memory=stub_memory, max_messages=3, keep_last=1
    )
    state = State(messages=[Message(role="user", content="question")])
    [_ async for _ in loop.stream(state, ctx=None)]

    assert cap.calls, "Compression LLM should have been called after memory inject"
    assert state.messages[0].role == "system"
    assert (state.messages[0].content or "").startswith(_SUMMARY_PREFIX)


@pytest.mark.unit
async def test_memory_update_called_correctly_when_compression_fires(stub_memory):
    """memory.update() still receives the current turn's messages after compression.

    Compression rewrites state.messages, but update() is called with the
    pre-identified trigger message and the new assistant reply — the two
    messages that represent this turn's exchange.
    """
    reason_step = FakeStep("llm", [make_llm_result("my reply")])
    stub_memory.set_entries(
        [
            Message(role="system", content="M1"),
            Message(role="system", content="M2"),
        ]
    )

    loop, _ = _make_loop_with_compression(
        reason_step, memory=stub_memory, max_messages=3, keep_last=1
    )
    state = State(messages=[Message(role="user", content="current question")])
    [_ async for _ in loop.stream(state, ctx=None)]

    assert len(stub_memory.update_calls) == 1
    stored = stub_memory.update_calls[0]["messages"]
    roles = [m.role for m in stored if m is not None]
    assert "user" in roles, "update() must include the trigger user message"
    assert "assistant" in roles, "update() must include the assistant reply"


@pytest.mark.unit
async def test_no_compression_when_below_threshold_with_memory(stub_memory):
    """No compression when retrieved memories + history stay below max_messages."""
    reason_step = FakeStep("llm", [make_llm_result("answer")])
    stub_memory.set_entries([Message(role="system", content="one memory")])

    loop, cap = _make_loop_with_compression(
        reason_step, memory=stub_memory, max_messages=10, keep_last=2
    )
    state = State(messages=[Message(role="user", content="hello")])
    [_ async for _ in loop.stream(state, ctx=None)]

    assert not cap.calls, "Compression should not fire when below the threshold"
    assert state._compression_context is None


@pytest.mark.unit
async def test_compression_context_set_when_memory_triggers_compression(stub_memory):
    """_compression_context is populated when memory injection causes compression."""
    reason_step = FakeStep("llm", [make_llm_result("done")])
    stub_memory.set_entries(
        [
            Message(role="system", content="M1"),
            Message(role="system", content="M2"),
        ]
    )

    loop, _ = _make_loop_with_compression(
        reason_step,
        memory=stub_memory,
        max_messages=3,
        keep_last=1,
        summary="COMPRESSED",
    )
    state = State(messages=[Message(role="user", content="q")])
    [_ async for _ in loop.stream(state, ctx=None)]

    # _compression_context holds the baseline for durable A2A persistence
    assert state._compression_context is not None
    summary_content = state._compression_context[0].content or ""
    assert "COMPRESSED" in summary_content
