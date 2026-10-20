from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from ant_ai.core.events import AnyEvent, ContentDeltaEvent

_any_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


@pytest.mark.unit
def test_content_delta_event_round_trips_through_any_event():
    event = ContentDeltaEvent(
        delta="tok",
        stream_id="stream-1",
        is_first=True,
        tool_call_index=0,
        tool_call_id="call-1",
        tool_call_name="my_tool",
    )
    dumped = event.model_dump()
    restored = _any_event_adapter.validate_python(dumped)
    assert isinstance(restored, ContentDeltaEvent)
    assert restored == event


@pytest.mark.unit
def test_event_origin_agent_defaults_to_none_and_round_trips():
    from ant_ai.core.events import EventOrigin, FinalAnswerEvent

    assert EventOrigin().agent is None
    event = FinalAnswerEvent(
        content="done", origin=EventOrigin(layer="agent", agent="coder")
    )
    restored = _any_event_adapter.validate_python(event.model_dump())
    assert restored == event
    assert restored.origin.agent == "coder"


@pytest.mark.unit
def test_event_from_a_peer_without_origin_agent_still_validates():
    """Events serialised by an older ant-ai (no `origin.agent`) must still load."""
    from ant_ai.core.events import FinalAnswerEvent

    payload = FinalAnswerEvent(content="done").model_dump()
    del payload["origin"]["agent"]
    restored = _any_event_adapter.validate_python(payload)
    assert restored.origin.agent is None
