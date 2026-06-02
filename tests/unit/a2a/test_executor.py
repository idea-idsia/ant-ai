from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from a2a.types import Message as A2AMessage, Role

from ant_ai.a2a.executor import A2AExecutor
from ant_ai.core.events import (
    FinalAnswerEvent,
    ReasoningEvent,
    ToolCallingEvent,
    ToolResultEvent,
)
from ant_ai.core.message import (
    Message,
    ToolCall,
    ToolCallMessage,
    ToolCallResultMessage,
    ToolFunction,
)


def _make_executor() -> A2AExecutor:
    return A2AExecutor(agent=MagicMock(), workflow=MagicMock())


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
def test_convert_history_tool_calling_event():
    """ToolCallingEvent in history becomes ToolCallMessage."""
    executor = _make_executor()
    tc = _tool_call()
    event = ToolCallingEvent(tool_calls=(tc,))
    msg = _a2a_msg()

    with patch.object(executor._a2a_to_hv, "translate", return_value=event):
        result = executor._convert_history([msg])

    assert len(result) == 1
    assert isinstance(result[0], ToolCallMessage)
    assert result[0].tool_calls == [tc]


@pytest.mark.unit
def test_convert_history_tool_result_event():
    """ToolResultEvent in history becomes ToolCallResultMessage with correct fields."""
    executor = _make_executor()
    event = ToolResultEvent(content="42", tool_call_id="id-1", name="my_tool")
    msg = _a2a_msg()

    with patch.object(executor._a2a_to_hv, "translate", return_value=event):
        result = executor._convert_history([msg])

    assert len(result) == 1
    assert isinstance(result[0], ToolCallResultMessage)
    assert result[0].tool_call_id == "id-1"
    assert result[0].name == "my_tool"
    assert result[0].content == "42"


@pytest.mark.unit
def test_convert_history_final_answer_event():
    """FinalAnswerEvent in history becomes an assistant Message."""
    executor = _make_executor()
    event = FinalAnswerEvent(content="done")
    msg = _a2a_msg()

    with patch.object(executor._a2a_to_hv, "translate", return_value=event):
        result = executor._convert_history([msg])

    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].role == "assistant"
    assert result[0].content == "done"


@pytest.mark.unit
def test_convert_history_reasoning_event_skipped():
    """ReasoningEvent and other non-conversation events are skipped."""
    executor = _make_executor()
    event = ReasoningEvent(content="thinking...")
    msg = _a2a_msg()

    with patch.object(executor._a2a_to_hv, "translate", return_value=event):
        result = executor._convert_history([msg])

    assert result == []


@pytest.mark.unit
def test_convert_history_agent_message_no_metadata_fallback():
    """Agent messages with no event metadata fall back to plain assistant Message."""
    executor = _make_executor()
    msg = _a2a_msg(role=Role.ROLE_AGENT)

    with (
        patch.object(executor._a2a_to_hv, "translate", return_value=None),
        patch("ant_ai.a2a.executor.get_message_text", return_value="fallback text"),
    ):
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

    with (
        patch.object(executor._a2a_to_hv, "translate", return_value=None),
        patch("ant_ai.a2a.executor.get_message_text", return_value=""),
    ):
        result = executor._convert_history([msg])

    assert result == []


@pytest.mark.unit
def test_convert_history_user_message():
    """User-role messages always become user Messages."""
    executor = _make_executor()
    msg = _a2a_msg(role=Role.ROLE_USER)

    with (
        patch.object(executor._a2a_to_hv, "translate", return_value=None),
        patch("ant_ai.a2a.executor.get_message_text", return_value="user input"),
    ):
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
    events = [
        ToolCallingEvent(tool_calls=(tc,)),
        ToolResultEvent(content="result", tool_call_id="id-1", name="my_tool"),
        FinalAnswerEvent(content="answer"),
    ]
    msgs = [_a2a_msg() for _ in events]

    with patch.object(executor._a2a_to_hv, "translate", side_effect=events):
        result = executor._convert_history(msgs)

    assert len(result) == 3
    assert isinstance(result[0], ToolCallMessage)
    assert isinstance(result[1], ToolCallResultMessage)
    assert isinstance(result[2], Message)
    assert result[2].role == "assistant"
