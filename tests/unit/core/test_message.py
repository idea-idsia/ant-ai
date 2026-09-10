from __future__ import annotations

import pytest

from ant_ai.core.message import Message, ToolCallResultMessage


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
