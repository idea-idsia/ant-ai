from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from a2a.types import CancelTaskRequest, Task, TaskState, TaskStatus
from httpx import HTTPError

from ant_ai.a2a.client import A2AClient, AgentClientError
from ant_ai.a2a.config import A2AConfig


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
