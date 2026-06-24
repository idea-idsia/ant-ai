from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ToolCallProgress,
    ToolCallStart,
)

from ant_ai.acp.translator import HVEventToACP
from ant_ai.core.events import (
    ClarificationNeededEvent,
    CompletedEvent,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    StartEvent,
    ToolCallingEvent,
    ToolResultEvent,
    UpdateEvent,
)
from ant_ai.core.message import ToolCall, ToolFunction


def _make_client() -> MagicMock:
    client = MagicMock()
    client.session_update = AsyncMock()
    return client


SESSION = "test-session"


@pytest.fixture
def translator() -> HVEventToACP:
    return HVEventToACP()


@pytest.mark.asyncio
async def test_final_answer_sends_agent_message(translator):
    client = _make_client()
    await translator.apply(FinalAnswerEvent(content="Hello!"), client, SESSION)

    client.session_update.assert_awaited_once()
    update = client.session_update.call_args.kwargs["update"]
    assert isinstance(update, AgentMessageChunk)
    assert update.content.text == "Hello!"


@pytest.mark.asyncio
async def test_clarification_sends_agent_message(translator):
    client = _make_client()
    await translator.apply(
        ClarificationNeededEvent(content="Which one?"), client, SESSION
    )

    update = client.session_update.call_args.kwargs["update"]
    assert isinstance(update, AgentMessageChunk)
    assert update.content.text == "Which one?"


@pytest.mark.asyncio
async def test_reasoning_sends_thought_chunk(translator):
    client = _make_client()
    await translator.apply(ReasoningEvent(content="Thinking..."), client, SESSION)

    update = client.session_update.call_args.kwargs["update"]
    assert isinstance(update, AgentThoughtChunk)
    assert update.content.text == "Thinking..."


@pytest.mark.asyncio
async def test_tool_calling_sends_tool_call_start_per_tool(translator):
    client = _make_client()
    tool_calls = (
        ToolCall(id="tc1", function=ToolFunction(name="search", arguments="{}")),
        ToolCall(id="tc2", function=ToolFunction(name="read", arguments="{}")),
    )
    await translator.apply(ToolCallingEvent(tool_calls=tool_calls), client, SESSION)

    # 1 empty AgentMessageChunk to create the container + 2 ToolCallStart updates
    assert client.session_update.await_count == 3
    updates = [call.kwargs["update"] for call in client.session_update.call_args_list]
    assert isinstance(updates[0], AgentMessageChunk)
    assert updates[0].content.text == ""
    assert all(isinstance(u, ToolCallStart) for u in updates[1:])
    assert updates[1].tool_call_id == "tc1"
    assert updates[1].title == "search"
    assert updates[1].status == "in_progress"
    assert updates[2].tool_call_id == "tc2"
    assert updates[2].title == "read"


@pytest.mark.asyncio
async def test_tool_result_sends_tool_call_progress(translator):
    client = _make_client()
    await translator.apply(
        ToolResultEvent(tool_call_id="tc1", name="search", content="results"),
        client,
        SESSION,
    )

    update = client.session_update.call_args.kwargs["update"]
    assert isinstance(update, ToolCallProgress)
    assert update.tool_call_id == "tc1"
    assert update.status == "completed"
    assert update.raw_output == "results"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        StartEvent(),
        UpdateEvent(content="Starting action 'run'"),
        MaxStepsReachedEvent(),
        CompletedEvent(),
    ],
)
async def test_lifecycle_events_are_noop(translator, event):
    client = _make_client()
    await translator.apply(event, client, SESSION)
    client.session_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_event_raises(translator):
    from ant_ai.core.events import Event

    client = _make_client()
    with pytest.raises(ValueError, match="No handler registered"):
        await translator.apply(Event(), client, SESSION)
