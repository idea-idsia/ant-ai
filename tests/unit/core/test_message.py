from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from ant_ai.core.message import (
    AnyMessage,
    BaseMessage,
    Message,
    ToolCall,
    ToolCallMessage,
    ToolCallResultMessage,
    ToolFunction,
)

_any_message_adapter: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


@pytest.mark.unit
def test_tool_result_is_error_defaults_to_false():
    msg = ToolCallResultMessage(tool_call_id="c1", name="t", content="x")
    assert msg.is_error is False


@pytest.mark.unit
def test_to_provider_dict_omits_internal_fields():
    """`kind` and `is_error` are internal; the OpenAI-style dict omits them."""
    msg = ToolCallResultMessage(tool_call_id="c1", name="t", content="x", is_error=True)
    d = msg.to_provider_dict()
    assert "is_error" not in d
    assert "kind" not in d
    assert d["role"] == "tool"
    assert d["tool_call_id"] == "c1"
    assert d["content"] == "x"


@pytest.mark.unit
def test_to_provider_dict_on_plain_message():
    d = Message(role="user", content="hi").to_provider_dict()
    assert "kind" not in d
    assert d["role"] == "user"


@pytest.mark.unit
def test_to_provider_dict_keeps_metadata_internal():
    """Provenance kept in `metadata` must never reach the provider payload."""
    msg = Message(role="user", content="hi", metadata={"envelope_id": "e-1"})
    d = msg.to_provider_dict()
    assert "metadata" not in d
    assert msg.metadata == {"envelope_id": "e-1"}


@pytest.mark.parametrize(
    "msg",
    [
        Message(role="user", content="hi"),
        ToolCallMessage(
            tool_calls=[
                ToolCall(id="c1", function=ToolFunction(name="t", arguments="{}"))
            ]
        ),
        ToolCallResultMessage(tool_call_id="c1", name="t", content="x"),
    ],
)
@pytest.mark.unit
def test_every_message_kind_round_trips_through_any_message(msg: BaseMessage):
    restored = _any_message_adapter.validate_python(msg.model_dump())
    assert type(restored) is type(msg)
    assert restored == msg


@pytest.mark.unit
def test_tool_messages_share_the_base_not_the_generic_message():
    """Each concrete message owns its `kind`; only `BaseMessage` is shared."""
    msg = ToolCallResultMessage(tool_call_id="c1", name="t", content="x")
    assert isinstance(msg, BaseMessage)
    assert not isinstance(msg, Message)
    assert "kind" not in BaseMessage.model_fields
