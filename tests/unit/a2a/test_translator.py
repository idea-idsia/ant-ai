from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from a2a.types import Artifact, Part, TaskArtifactUpdateEvent

from ant_ai.a2a.translator import A2AToHVEvent, HVEventToA2A
from ant_ai.core.events import (
    ContentDeltaEvent,
    FinalAnswerEvent,
    ReasoningEvent,
    ToolCallingEvent,
)
from ant_ai.core.message import ToolCall, ToolFunction


def _artifact_event(
    *,
    artifact_id: str = "art-1",
    text: str = "hello",
    event_metadata: dict | None = None,
    append: bool = True,
    last_chunk: bool = False,
) -> TaskArtifactUpdateEvent:
    artifact = Artifact(
        artifact_id=artifact_id, parts=[Part(text=text)] if text else []
    )
    if event_metadata is not None:
        artifact.metadata.update({"event": event_metadata})
    return TaskArtifactUpdateEvent(
        task_id="task-1",
        context_id="ctx-1",
        artifact=artifact,
        append=append,
        last_chunk=last_chunk,
    )


@pytest.mark.unit
def test_artifact_with_event_metadata_reconstructs_any_event():
    raw = _artifact_event(
        event_metadata={"kind": "final_answer", "content": "hi there"},
    )
    event = A2AToHVEvent().translate(raw)
    assert isinstance(event, FinalAnswerEvent)
    assert event.content == "hi there"
    assert event.task_id == "task-1"
    assert event.session_id == "ctx-1"


@pytest.mark.unit
def test_artifact_without_metadata_reconstructs_content_delta():
    raw = _artifact_event(artifact_id="stream-42", text="tok")
    event = A2AToHVEvent().translate(raw)
    assert isinstance(event, ContentDeltaEvent)
    assert event.delta == "tok"
    assert event.stream_id == "stream-42"
    assert event.task_id == "task-1"
    assert event.session_id == "ctx-1"


@pytest.mark.unit
def test_empty_artifact_returns_none():
    raw = _artifact_event(text="")
    event = A2AToHVEvent().translate(raw)
    assert event is None


def _mock_updater() -> AsyncMock:
    updater = AsyncMock()
    updater.new_agent_message = lambda parts: AsyncMock(metadata={}, parts=parts)
    return updater


@pytest.mark.unit
async def test_content_delta_dropped_when_stream_artifacts_disabled():
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=False)
    event = ContentDeltaEvent(delta="tok", stream_id="s1", is_first=True)

    await translator.apply(event, updater)

    updater.add_artifact.assert_not_called()


@pytest.mark.unit
async def test_content_delta_calls_add_artifact_with_append_and_artifact_id():
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)

    first = ContentDeltaEvent(delta="Hel", stream_id="s1", is_first=True)
    second = ContentDeltaEvent(delta="lo", stream_id="s1", is_first=False)
    await translator.apply(first, updater)
    await translator.apply(second, updater)

    assert updater.add_artifact.call_count == 2
    first_call, second_call = updater.add_artifact.call_args_list
    assert first_call.kwargs["artifact_id"] == "s1:content"
    assert first_call.kwargs["append"] is False
    assert second_call.kwargs["artifact_id"] == "s1:content"
    assert second_call.kwargs["append"] is True


@pytest.mark.unit
async def test_content_delta_for_tool_call_uses_composite_artifact_id():
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = ContentDeltaEvent(
        target_kind="tool_calling",
        delta="{}",
        stream_id="s1",
        is_first=True,
        tool_call_index=0,
    )

    await translator.apply(event, updater)

    assert updater.add_artifact.call_args.kwargs["artifact_id"] == "s1:tool:0"


@pytest.mark.unit
async def test_content_delta_for_reasoning_uses_distinct_artifact_id():
    """Reasoning and content deltas from the same generation must not share
    an artifact id, or a peer reading raw artifact text would see them
    concatenated together."""
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = ContentDeltaEvent(
        target_kind="reasoning", delta="thinking...", stream_id="s1", is_first=True
    )

    await translator.apply(event, updater)

    assert updater.add_artifact.call_args.kwargs["artifact_id"] == "s1:reasoning"


@pytest.mark.unit
async def test_agent_message_closes_artifact_when_stream_id_and_streaming_enabled():
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = FinalAnswerEvent(content="done", stream_id="s1")

    await translator.apply(event, updater)

    updater.add_artifact.assert_called_once_with(
        parts=[], artifact_id="s1:content", append=True, last_chunk=True
    )
    updater.update_status.assert_called_once()


@pytest.mark.unit
async def test_agent_message_closes_reasoning_artifact_separately():
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = ReasoningEvent(content="thinking...", stream_id="s1")

    await translator.apply(event, updater)

    updater.add_artifact.assert_called_once_with(
        parts=[], artifact_id="s1:reasoning", append=True, last_chunk=True
    )


@pytest.mark.unit
async def test_agent_message_closes_content_and_each_tool_artifact():
    """A ToolCallingEvent whose model also narrated text before calling
    tools must close both the content artifact and one artifact per tool
    call -- not just a single bare-stream_id artifact."""
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = ToolCallingEvent(
        content="Let me check that.",
        stream_id="s1",
        tool_calls=(
            ToolCall(id="tc-1", function=ToolFunction(name="echo", arguments="{}")),
            ToolCall(id="tc-2", function=ToolFunction(name="echo", arguments="{}")),
        ),
    )

    await translator.apply(event, updater)

    closed_ids = {c.kwargs["artifact_id"] for c in updater.add_artifact.call_args_list}
    assert closed_ids == {"s1:content", "s1:tool:0", "s1:tool:1"}


@pytest.mark.unit
async def test_agent_message_skips_content_artifact_when_no_narration():
    """A ToolCallingEvent with no narration text (pure tool call, no
    preceding content deltas) must not attempt to close a content artifact
    that was never created."""
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = ToolCallingEvent(
        content="",
        stream_id="s1",
        tool_calls=(
            ToolCall(id="tc-1", function=ToolFunction(name="echo", arguments="{}")),
        ),
    )

    await translator.apply(event, updater)

    updater.add_artifact.assert_called_once_with(
        parts=[], artifact_id="s1:tool:0", append=True, last_chunk=True
    )


@pytest.mark.unit
async def test_agent_message_skips_artifact_close_without_stream_id():
    updater = _mock_updater()
    translator = HVEventToA2A(stream_artifacts=True)
    event = FinalAnswerEvent(content="done")

    await translator.apply(event, updater)

    updater.add_artifact.assert_not_called()
    updater.update_status.assert_called_once()
