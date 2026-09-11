from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from a2a.types import Message as A2AMessage, Role

from ant_ai.a2a.executor import A2AExecutor
from ant_ai.core.message import (
    Message,
    ToolCall,
    ToolCallMessage,
    ToolCallResultMessage,
    ToolFunction,
)
from ant_ai.core.types import InvocationContext


def _make_executor(*, stream_artifacts: bool = False) -> A2AExecutor:
    return A2AExecutor(
        agent=MagicMock(), workflow=MagicMock(), stream_artifacts=stream_artifacts
    )


def _a2a_msg(role: Role = Role.ROLE_AGENT, text: str = "hello") -> A2AMessage:
    msg = MagicMock(spec=A2AMessage)
    msg.role = role
    msg.metadata = None
    msg.task_id = "task-1"
    msg.context_id = "ctx-1"
    return msg


def _tool_call(call_id: str = "id-1", name: str = "my_tool") -> ToolCall:
    fn = MagicMock(spec=ToolFunction)
    fn.name = name
    fn.arguments = "{}"
    tc = MagicMock(spec=ToolCall)
    tc.id = call_id
    tc.function = fn
    return tc


@pytest.mark.unit
def test_convert_history_passes_through_non_none():
    """_convert_history returns whatever to_history_message produces, filtering None."""
    executor = _make_executor()
    tc = _tool_call()
    messages = [
        ToolCallMessage(tool_calls=[tc]),
        ToolCallResultMessage(tool_call_id="id-1", name="my_tool", content="42"),
        Message(role="assistant", content="done"),
    ]
    msgs = [_a2a_msg() for _ in messages]

    with patch.object(executor._a2a_to_hv, "to_history_message", side_effect=messages):
        result = executor._convert_history(msgs)

    assert result == messages


@pytest.mark.unit
def test_convert_history_filters_none():
    """None values from to_history_message are excluded from the result."""
    executor = _make_executor()
    kept = Message(role="assistant", content="done")
    msgs = [_a2a_msg(), _a2a_msg()]

    with patch.object(
        executor._a2a_to_hv, "to_history_message", side_effect=[None, kept]
    ):
        result = executor._convert_history(msgs)

    assert result == [kept]


@pytest.mark.unit
def test_convert_history_agent_message_no_metadata_fallback():
    """Agent messages with no event metadata fall back to plain assistant Message."""
    executor = _make_executor()
    msg = _a2a_msg(role=Role.ROLE_AGENT)
    fallback = Message(role="assistant", content="fallback text")

    with patch.object(executor._a2a_to_hv, "to_history_message", return_value=fallback):
        result = executor._convert_history([msg])

    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].role == "assistant"
    assert result[0].content == "fallback text"


@pytest.mark.unit
def test_convert_history_agent_message_no_metadata_empty_text_skipped():
    """Agent fallback messages with no text content are not added."""
    executor = _make_executor()
    msg = _a2a_msg(role=Role.ROLE_AGENT)

    with patch.object(executor._a2a_to_hv, "to_history_message", return_value=None):
        result = executor._convert_history([msg])

    assert result == []


@pytest.mark.unit
def test_convert_history_user_message():
    """User-role messages always become user Messages."""
    executor = _make_executor()
    msg = _a2a_msg(role=Role.ROLE_USER)
    user_msg = Message(role="user", content="user input")

    with patch.object(executor._a2a_to_hv, "to_history_message", return_value=user_msg):
        result = executor._convert_history([msg])

    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].role == "user"
    assert result[0].content == "user input"


@pytest.mark.unit
def test_convert_history_mixed_sequence():
    """Full tool round-trip sequence reconstructs correctly."""
    executor = _make_executor()
    tc = _tool_call()
    history_messages = [
        ToolCallMessage(tool_calls=[tc]),
        ToolCallResultMessage(tool_call_id="id-1", name="my_tool", content="result"),
        Message(role="assistant", content="answer"),
    ]
    msgs = [_a2a_msg() for _ in history_messages]

    with patch.object(
        executor._a2a_to_hv, "to_history_message", side_effect=history_messages
    ):
        result = executor._convert_history(msgs)

    assert len(result) == 3
    assert isinstance(result[0], ToolCallMessage)
    assert isinstance(result[1], ToolCallResultMessage)
    assert isinstance(result[2], Message)
    assert result[2].role == "assistant"


@pytest.mark.unit
def test_stream_artifacts_defaults_to_disabled_on_translator():
    executor = _make_executor()
    assert executor._translator._stream_artifacts is False


@pytest.mark.unit
def test_stream_artifacts_flag_threaded_into_translator():
    executor = _make_executor(stream_artifacts=True)
    assert executor._translator._stream_artifacts is True


def _stream_capturing_ctx(captured: dict):
    async def fake_stream(*, agent, ctx, state):
        captured["ctx"] = ctx
        if False:
            yield

    return fake_stream


def _request_context(metadata: dict) -> MagicMock:
    context = MagicMock()
    context.metadata = metadata
    context.related_tasks = []
    context.get_user_input.return_value = "hi"
    return context


@pytest.mark.unit
async def test_execute_builds_default_context_from_metadata():
    """Base fields (`user_id`, `llm_settings`, ...) are filled from the A2A request
    metadata by name; unknown keys are ignored."""
    captured: dict = {}
    workflow = MagicMock()
    workflow.stream = _stream_capturing_ctx(captured)
    executor = A2AExecutor(agent=MagicMock(), workflow=workflow)
    task = MagicMock()
    task.context_id = "ctx-1"

    await executor._execute(
        _request_context(
            {"user_id": "u-42", "llm_settings": {"temperature": 0}, "traceparent": "x"}
        ),
        updater=MagicMock(),
        task=task,
    )

    ctx = captured["ctx"]
    assert type(ctx) is InvocationContext
    assert ctx.session_id == "ctx-1"
    assert ctx.user_id == "u-42"
    assert ctx.llm_settings == {"temperature": 0}


@pytest.mark.unit
async def test_execute_builds_custom_context_class_from_metadata():
    """`context_class` picks the InvocationContext subclass; its extra fields are
    filled from metadata the same way."""

    class TaggedContext(InvocationContext):
        tags: list[str] | None = None
        tenant: str = ""

    captured: dict = {}
    workflow = MagicMock()
    workflow.stream = _stream_capturing_ctx(captured)
    executor = A2AExecutor(
        agent=MagicMock(), workflow=workflow, context_class=TaggedContext
    )
    task = MagicMock()
    task.context_id = "ctx-1"

    await executor._execute(
        _request_context({"user_id": "u-42", "tags": ["team:a"], "tenant": "acme"}),
        updater=MagicMock(),
        task=task,
    )

    ctx = captured["ctx"]
    assert isinstance(ctx, TaggedContext)
    assert ctx.session_id == "ctx-1"
    assert ctx.tags == ["team:a"]
    assert ctx.tenant == "acme"


@pytest.mark.unit
async def test_cancel_publishes_canceled_status():
    """`cancel` publishes the canceled status the SDK's `on_cancel_task` needs
    (it refuses any other final state) instead of raising."""
    from a2a.server.events import EventQueueLegacy
    from a2a.types import TaskState, TaskStatusUpdateEvent

    queue = EventQueueLegacy()
    task = MagicMock()
    task.id = "task-1"
    task.context_id = "ctx-1"
    context = MagicMock()
    context.current_task = task

    await _make_executor().cancel(context, queue)

    raw = await queue.dequeue_event()
    assert isinstance(raw, TaskStatusUpdateEvent)
    assert raw.status.state == TaskState.TASK_STATE_CANCELED


@pytest.mark.unit
async def test_cancel_without_task_raises():
    context = MagicMock()
    context.current_task = None
    with pytest.raises(Exception, match="No task to cancel"):
        await _make_executor().cancel(context, MagicMock())


@pytest.mark.unit
async def test_execute_reraises_cancelled_error_without_wrapping():
    """A cancel is not a crash: `CancelledError` propagates as-is so the event
    loop sees the task stop, rather than being wrapped in `InternalError`."""
    import asyncio

    from a2a.server.events import EventQueueLegacy
    from a2a.types import Message as A2AMsg, Part, Role

    async def cancelled_stream(*, agent, ctx, state):
        raise asyncio.CancelledError()
        yield

    workflow = MagicMock()
    workflow.stream = cancelled_stream
    executor = A2AExecutor(agent=MagicMock(), workflow=workflow)

    context = MagicMock()
    context.message = A2AMsg(
        message_id="m-1", role=Role.ROLE_USER, parts=[Part(text="hi")]
    )
    context.current_task = None
    context.metadata = {}
    context.related_tasks = []
    context.get_user_input.return_value = "hi"

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(context, EventQueueLegacy())


@pytest.mark.unit
def test_convert_history_tool_result_keeps_is_error():
    """A tool result rebuilt from A2A history keeps the `is_error` flag the
    event recorded, so a resumed conversation does not lose it."""
    from ant_ai.core.events import ToolResultEvent

    executor = _make_executor()
    event = ToolResultEvent(
        content="ERROR: nope", tool_call_id="c1", name="t", is_error=True
    )
    msg = _a2a_msg()
    with patch.object(executor._a2a_to_hv, "translate", return_value=event):
        result = executor._a2a_to_hv.to_history_message(msg)
    assert isinstance(result, ToolCallResultMessage)
    assert result.is_error is True


@pytest.mark.unit
def test_history_after_cancel_mid_tool_call_is_resumable():
    """A task cancelled while its tools ran ends on the assistant's `tool_calls`
    turn. Replaying it answers each unanswered call with an error result, so
    the next user message can follow; answered calls are left alone."""
    from ant_ai.core.events import ToolCallingEvent, ToolResultEvent

    executor = _make_executor()
    events = [
        ToolCallingEvent(
            tool_calls=[_tool_call("c1", "fast"), _tool_call("c2", "slow")]
        ),
        ToolResultEvent(content="done", tool_call_id="c1", name="fast"),
    ]
    msgs = [_a2a_msg() for _ in events]

    with patch.object(executor._a2a_to_hv, "translate", side_effect=events):
        result = executor._convert_history(msgs)

    assert [type(m).__name__ for m in result] == [
        "ToolCallMessage",
        "ToolCallResultMessage",
        "ToolCallResultMessage",
    ]
    assert (result[1].tool_call_id, result[1].content, result[1].is_error) == (
        "c1",
        "done",
        False,
    )
    assert (result[2].tool_call_id, result[2].name, result[2].is_error) == (
        "c2",
        "slow",
        True,
    )


@pytest.mark.unit
def test_dangling_tool_call_is_answered_before_the_next_message():
    """The synthetic result goes right after the `tool_calls` turn it answers,
    not at the end of the history."""
    executor = _make_executor()
    history = [
        ToolCallMessage(tool_calls=[_tool_call("c1", "slow")]),
        Message(role="user", content="still there?"),
    ]

    result = executor._answer_dangling_tool_calls(history)

    assert [type(m).__name__ for m in result] == [
        "ToolCallMessage",
        "ToolCallResultMessage",
        "Message",
    ]
    assert result[1].tool_call_id == "c1" and result[1].is_error


@pytest.mark.unit
def test_history_after_clarification_is_resumable():
    """A task that ended on a clarification rebuilds as tool_calls -> tool result
    (the question); the ClarificationNeededEvent itself adds nothing, so the
    question is not duplicated and the next user message can follow."""
    from ant_ai.core.events import (
        ClarificationNeededEvent,
        ToolCallingEvent,
        ToolResultEvent,
    )

    executor = _make_executor()
    tc = _tool_call(call_id="c1", name="ask")
    events = [
        ToolCallingEvent(tool_calls=[tc]),
        ToolResultEvent(content="Which one?", tool_call_id="c1", name="ask"),
        ClarificationNeededEvent(content="Which one?"),
    ]
    msgs = [_a2a_msg() for _ in events]

    with patch.object(executor._a2a_to_hv, "translate", side_effect=events):
        result = executor._convert_history(msgs)

    assert [type(m).__name__ for m in result] == [
        "ToolCallMessage",
        "ToolCallResultMessage",
    ]
    assert result[1].content == "Which one?"


# --- what the caller sees when a run fails ---------------------------------


def _failing_executor(exc: Exception) -> A2AExecutor:
    async def failing_stream(*, agent, ctx, state):
        raise exc
        yield

    workflow = MagicMock()
    workflow.stream = failing_stream
    return A2AExecutor(agent=MagicMock(), workflow=workflow)


def _request_context_with_message() -> MagicMock:
    from a2a.types import Message as A2AMsg, Part, Role

    context = MagicMock()
    context.message = A2AMsg(
        message_id="m-1", role=Role.ROLE_USER, parts=[Part(text="hi")]
    )
    context.current_task = None
    context.metadata = {}
    context.related_tasks = []
    context.get_user_input.return_value = "hi"
    return context


@pytest.mark.unit
async def test_a2a_error_raised_inside_the_run_reaches_the_caller_unchanged():
    """Something that knew what the caller should hear raised a protocol error
    with a message; the executor must not flatten it to a bare InternalError."""
    from a2a.server.events import EventQueueLegacy
    from a2a.types import InternalError, InvalidParamsError

    with pytest.raises(InternalError) as info:
        await _failing_executor(
            InternalError("This conversation is too long.")
        ).execute(_request_context_with_message(), EventQueueLegacy())
    assert info.value.message == "This conversation is too long."

    with pytest.raises(InvalidParamsError) as info:
        await _failing_executor(InvalidParamsError("bad tenant")).execute(
            _request_context_with_message(), EventQueueLegacy()
        )
    assert info.value.message == "bad tenant"


@pytest.mark.unit
async def test_other_exceptions_are_hidden_behind_a_bare_internal_error():
    from a2a.server.events import EventQueueLegacy
    from a2a.types import InternalError

    with pytest.raises(InternalError) as info:
        await _failing_executor(RuntimeError("secret path /etc/x")).execute(
            _request_context_with_message(), EventQueueLegacy()
        )
    assert info.value.message == "Internal error"
    assert isinstance(info.value.__cause__, RuntimeError)
