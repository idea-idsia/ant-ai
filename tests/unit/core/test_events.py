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
