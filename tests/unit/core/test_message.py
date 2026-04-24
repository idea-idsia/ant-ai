from __future__ import annotations

import pytest

from ant_ai.core.message import Message, MessageChunk


@pytest.mark.unit
def test_merge_concatenates_deltas():
    a = MessageChunk(role="assistant", delta="Hello, ")
    b = MessageChunk(role="assistant", delta="world")
    merged: MessageChunk = a.merge(b)
    assert merged.role == "assistant"
    assert merged.delta == "Hello, world"


@pytest.mark.unit
def test_merge_empty_metadata_produces_empty_metadata():
    a = MessageChunk(role="assistant", delta="a")
    b = MessageChunk(role="assistant", delta="b")
    merged: MessageChunk = a.merge(b)
    assert merged.metadata == {}


@pytest.mark.unit
def test_merge_unions_metadata_right_wins_on_collision():
    a = MessageChunk(role="assistant", delta="a", metadata={"k": "left", "a": 1})
    b = MessageChunk(role="assistant", delta="b", metadata={"k": "right", "b": 2})
    merged: MessageChunk = a.merge(b)
    assert merged.metadata == {"k": "right", "a": 1, "b": 2}


@pytest.mark.unit
def test_merge_raises_on_role_mismatch():
    a = MessageChunk(role="user", delta="hello")
    b = MessageChunk(role="assistant", delta="world")
    with pytest.raises(ValueError, match="roles must match"):
        a.merge(b)


@pytest.mark.unit
def test_to_message_produces_message_with_correct_fields():
    chunk = MessageChunk(role="assistant", delta="content text", metadata={"k": "v"})
    msg = chunk.to_message()
    assert isinstance(msg, Message)
    assert msg.role == "assistant"
    assert msg.content == "content text"
    assert msg.metadata == {"k": "v"}


@pytest.mark.unit
def test_to_message_with_empty_metadata():
    chunk = MessageChunk(role="user", delta="hi")
    msg: Message = chunk.to_message()
    assert msg.metadata == {}
