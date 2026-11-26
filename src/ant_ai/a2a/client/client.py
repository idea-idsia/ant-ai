from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from uuid import uuid4

from a2a.client import Client
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
)
from a2a.types.a2a_pb2 import Task
from httpx import HTTPError, TimeoutException
from pydantic import BaseModel, Field, PrivateAttr

from ant_ai.a2a.client.config import A2AConfig
from ant_ai.a2a.client.connection import A2AConnection, AgentClientError
from ant_ai.a2a.client.response import ResponseReader
from ant_ai.core.events import BaseEvent
from ant_ai.observer import obs


class A2AClient(BaseModel):
    """
    Client for interacting with an agent via the A2A protocol. Encapsulates connection management, message sending, and response handling.
    """

    config: A2AConfig = Field(
        description="Configuration for the A2A client, including endpoint, timeouts, and supported transports."
    )

    _agent_card: AgentCard | None = PrivateAttr(default=None)
    _connection: A2AConnection = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self._connection = A2AConnection(self.config)

    async def __aenter__(self) -> A2AClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self._agent_card = None
        await self._connection.aclose()

    async def get_agent_card(self) -> AgentCard:
        """Fetch the AgentCard from the remote.

        Returns:
            An AgentCard object representing the remote agent's information.
        """
        await self._ensure_client()
        if self._agent_card is None:
            raise RuntimeError("Agent card not initialized after _ensure_client()")
        return self._agent_card

    async def _ensure_client(self) -> Client:
        try:
            return await self._connection.open(self._agent_card)
        finally:
            self._agent_card = self._connection.agent_card or self._agent_card

    async def send_message(
        self,
        message: str,
        *,
        request_metadata: dict | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        reference_task_ids: list[str] | None = None,
    ) -> AsyncIterator[BaseEvent]:
        """Sends a message to the agent and yields events as responses are received.

        Args:
            message: The message to send to the agent.
            request_metadata: Optional metadata to include with the request.
            context_id: Optional context ID for the message. If not provided, a new UUID will be
                generated, unless `task_id` is given: the agent then takes it from the task.
            task_id: Optional ID of a task waiting on input (`input-required`), to continue
                that task with this message instead of starting a new one.
            reference_task_ids: Optional list of task IDs that this message references, for better traceability in task management.

        Yields:
            An event representing a response from the agent, translated from the raw A2A response.
        """

        client: Client = await self._ensure_client()
        request: SendMessageRequest = _send_request(
            message,
            context_id=context_id,
            task_id=task_id,
            reference_task_ids=reference_task_ids,
        )
        if self.config.trusted:
            request.metadata.update(obs.propagation_headers())
        if request_metadata:
            request.metadata.update(request_metadata)

        reader = ResponseReader()
        with _client_errors():
            async for chunk in client.send_message(request):
                for ev in reader.read(chunk):
                    yield ev

    async def cancel_task(self, task_id: str) -> Task:
        """Ask the agent to cancel a running task and return its final state.

        The counterpart to `A2AExecutor.cancel`. Shares `send_message`'s error
        contract: every transport failure becomes `AgentClientError`.

        Args:
            task_id: The id of the task to cancel, as the agent reported it.

        Returns:
            The `Task` in its final state, which the protocol guarantees is canceled.
        """
        client: Client = await self._ensure_client()
        with _client_errors():
            return await client.cancel_task(CancelTaskRequest(id=task_id))


def _send_request(
    message: str,
    *,
    context_id: str | None,
    task_id: str | None,
    reference_task_ids: list[str] | None,
) -> SendMessageRequest:
    msg = Message(
        role=Role.ROLE_USER,
        parts=[Part(text=message)],
        message_id=str(uuid4()),
    )
    if task_id:
        msg.task_id = task_id
    if context_id or not task_id:
        msg.context_id = context_id or str(uuid4())
    if reference_task_ids:
        msg.reference_task_ids.extend(reference_task_ids)
    return SendMessageRequest(message=msg)


@contextmanager
def _client_errors() -> Generator[None]:
    try:
        yield
    except TimeoutException as e:
        raise AgentClientError(f"Agent request timed out: {e}") from e
    except HTTPError as e:
        raise AgentClientError(f"Agent HTTP error: {e}") from e
    except Exception as e:
        raise AgentClientError(f"Agent client error: {e}") from e
