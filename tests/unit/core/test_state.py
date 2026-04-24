from __future__ import annotations

import pytest

from ant_ai.core.message import Message
from ant_ai.core.types import State


def _msg(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


@pytest.mark.unit
def test_add_message_appends_single_message():
    state = State()
    state.add_message(_msg("hi"))
    assert len(state.messages) == 1
    assert state.messages[0].content == "hi"


@pytest.mark.unit
def test_add_message_preserves_insertion_order():
    state = State()
    for text in ("first", "second", "third"):
        state.add_message(_msg(text))
    assert [m.content for m in state.messages] == ["first", "second", "third"]


@pytest.mark.unit
def test_last_message_returns_most_recent():
    state = State()
    state.add_message(_msg("earlier"))
    state.add_message(_msg("later"))
    assert state.last_message.content == "later"


@pytest.mark.unit
def test_last_message_raises_on_empty_state():
    state = State()
    with pytest.raises(ValueError, match="No messages"):
        _ = state.last_message


@pytest.mark.unit
def test_state_initialises_with_empty_messages_and_artefacts():
    state = State()
    assert state.messages == []
    assert state.artefacts == []
