from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from a2a.types import (
    CancelTaskRequest,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from httpx import HTTPError

from ant_ai.a2a.client import A2AClient, AgentClientError
from ant_ai.a2a.client.config import A2AConfig
from ant_ai.a2a.server.types import A2AMetadata
from ant_ai.core.events import FinalAnswerEvent, ToolResultEvent, UpdateEvent


def _client() -> A2AClient:
    return A2AClient(config=A2AConfig(endpoint="http://127.0.0.1:1/"))


@pytest.mark.unit
async def test_cancel_task_sends_request_and_returns_task():
    canceled = Task(
        id="task-1",
        context_id="ctx-1",
        status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
    )
    sdk_client = AsyncMock()
    sdk_client.cancel_task.return_value = canceled

    with patch.object(A2AClient, "_ensure_client", return_value=sdk_client):
        result = await _client().cancel_task("task-1")

    assert result is canceled
    (request,), _ = sdk_client.cancel_task.call_args
    assert isinstance(request, CancelTaskRequest)
    assert request.id == "task-1"


@pytest.mark.unit
async def test_cancel_task_wraps_transport_errors():
    """Same error contract as `send_message`: transport failures become
    `AgentClientError`."""
    sdk_client = AsyncMock()
    sdk_client.cancel_task.side_effect = HTTPError("boom")

    with (
        patch.object(A2AClient, "_ensure_client", return_value=sdk_client),
        pytest.raises(AgentClientError, match="HTTP error"),
    ):
        await _client().cancel_task("task-1")


# -- send_message: what a whole task answers with ------------------------------------


def _agent_message(message_id: str, event) -> Message:
    msg = Message(
        role=Role.ROLE_AGENT, message_id=message_id, parts=[Part(text=event.content)]
    )
    msg.metadata.update(A2AMetadata(event=event).model_dump())
    return msg


def _finished_task() -> Task:
    """A task as a non-streaming call returns it: the answer is in its history, its
    metadata is only the last status update."""
    task = Task(
        id="task-1",
        context_id="ctx-1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )
    task.history.append(
        Message(role=Role.ROLE_USER, message_id="u", parts=[Part(text="100 EUR?")])
    )
    task.history.append(
        _agent_message("m1", ToolResultEvent(content="100 EUR = 94 CHF"))
    )
    task.history.append(_agent_message("m2", FinalAnswerEvent(content="It is 94 CHF.")))
    task.metadata.update(
        A2AMetadata(event=UpdateEvent(content="Completed action 'answer'")).model_dump()
    )
    return task


async def _events(*chunks: StreamResponse) -> list:
    async def stream(_request):
        for chunk in chunks:
            yield chunk

    sdk_client = AsyncMock()
    sdk_client.send_message = stream
    with patch.object(A2AClient, "_ensure_client", return_value=sdk_client):
        return [ev async for ev in _client().send_message("100 EUR?")]


@pytest.mark.unit
async def test_a_finished_task_answers_with_its_agent_messages():
    events = await _events(StreamResponse(task=_finished_task()))
    assert [type(e) for e in events] == [ToolResultEvent, FinalAnswerEvent]
    assert events[-1].content == "It is 94 CHF."
    assert (events[-1].task_id, events[-1].session_id) == ("task-1", "ctx-1")


@pytest.mark.unit
async def test_a_task_after_streamed_messages_does_not_repeat_them():
    streamed = TaskStatusUpdateEvent(
        task_id="task-1",
        context_id="ctx-1",
        status=TaskStatus(
            state=TaskState.TASK_STATE_WORKING,
            message=_agent_message("m2", FinalAnswerEvent(content="It is 94 CHF.")),
        ),
    )
    streamed.metadata.update(
        A2AMetadata(event=FinalAnswerEvent(content="It is 94 CHF.")).model_dump()
    )
    events = await _events(
        StreamResponse(status_update=streamed), StreamResponse(task=_finished_task())
    )
    assert [type(e) for e in events] == [FinalAnswerEvent, ToolResultEvent]


@pytest.mark.unit
async def test_a_peer_without_event_metadata_still_gives_its_answer():
    task = Task(
        id="task-1",
        context_id="ctx-1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )
    task.history.append(
        Message(role=Role.ROLE_AGENT, message_id="a", parts=[Part(text="94 CHF")])
    )
    (event,) = await _events(StreamResponse(task=task))
    assert isinstance(event, FinalAnswerEvent) and event.content == "94 CHF"


@pytest.mark.unit
async def test_an_ant_ai_task_without_an_answer_gets_no_made_up_one():
    task = Task(
        id="task-1",
        context_id="ctx-1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )
    task.history.append(
        _agent_message("m1", ToolResultEvent(content="100 EUR = 94 CHF"))
    )
    events = await _events(StreamResponse(task=task))
    assert [type(e) for e in events] == [ToolResultEvent]
