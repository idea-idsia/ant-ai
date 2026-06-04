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
