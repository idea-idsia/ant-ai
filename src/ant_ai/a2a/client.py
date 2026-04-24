from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from a2a.client import A2ACardResolver, Client, ClientConfig, create_client
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
)
from a2a.types.a2a_pb2 import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent
from httpx import AsyncClient, HTTPError, TimeoutException
from pydantic import BaseModel, Field, PrivateAttr

from ant_ai.a2a.config import A2AConfig
from ant_ai.a2a.translator import A2AToHVEvent
from ant_ai.core.events import Event
from ant_ai.observer import obs


class AgentClientError(RuntimeError):
    pass


class A2AClient(BaseModel):
    """
    Client for interacting with an agent via the A2A protocol. Encapsulates connection management, message sending, and response handling.
    """

    config: A2AConfig = Field(
        description="Configuration for the A2A client, including endpoint, timeouts, and supported transports."
    )

    _agent_card: AgentCard | None = PrivateAttr(default=None)
    _httpx: AsyncClient | None = PrivateAttr(default=None)
    _client: Client | None = PrivateAttr(default=None)
    _init_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _translator: A2AToHVEvent = PrivateAttr(default_factory=A2AToHVEvent)

    async def __aenter__(self) -> A2AClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        httpx_client: AsyncClient | None = self._httpx
        self._httpx = None
        self._client = None
        self._agent_card = None
        if httpx_client is not None:
            await httpx_client.aclose()

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
        if self._client is not None and self._httpx is not None:
            return self._client

        async with self._init_lock:
            if self._client is not None and self._httpx is not None:
                return self._client

            httpx_client = AsyncClient(timeout=self.config.timeout)
            try:
                client: Client = await self._build_a2a_client(httpx_client)
            except Exception:
                await httpx_client.aclose()
                raise

            self._httpx: AsyncClient = httpx_client
            self._client: Client = client
            return client

    async def _build_a2a_client(self, httpx_client: AsyncClient) -> Client:
        if self._agent_card is None:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=self.config.endpoint,
                agent_card_path=self.config.agent_card_path,
            )
            try:
                self._agent_card: AgentCard = await resolver.get_agent_card()
            except (HTTPError, Exception) as e:
                raise AgentClientError(f"Failed to resolve AgentCard: {e}") from e

        cfg = ClientConfig(
            httpx_client=httpx_client,
            supported_protocol_bindings=list(self.config.supported_protocol_bindings),
            streaming=self.config.streaming,
        )
        return await create_client(self._agent_card, client_config=cfg)

    async def send_message(
        self,
        message: str,
        *,
        request_metadata: dict | None = None,
        context_id: str | None = None,
        reference_task_ids: list[str] | None = None,
    ) -> AsyncIterator[Event]:
        """Sends a message to the agent and yields events as responses are received.

        Args:
            message: The message to send to the agent.
            request_metadata: Optional metadata to include with the request.
            context_id: Optional context ID for the message. If not provided, a new UUID will be generated.
            reference_task_ids: Optional list of task IDs that this message references, for better traceability in task management.

        Yields:
            An event representing a response from the agent, translated from the raw A2A response.
        """

        client: Client = await self._ensure_client()

        msg = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=message)],
            message_id=str(uuid4()),
            context_id=context_id or str(uuid4()),
        )
        if reference_task_ids:
            msg.reference_task_ids.extend(reference_task_ids)

        request = SendMessageRequest(message=msg)

        if self.config.propagate_trace_context:
            request.metadata.update(obs.propagation_headers())
        if request_metadata:
            request.metadata.update(request_metadata)

        try:
            async for chunk in client.send_message(request):
                if chunk.HasField("status_update"):
                    raw: TaskStatusUpdateEvent = chunk.status_update
                elif chunk.HasField("message"):
                    raw: Message = chunk.message
                elif chunk.HasField("task"):
                    raw: Task = chunk.task
                elif chunk.HasField("artifact_update"):
                    raw: TaskArtifactUpdateEvent = chunk.artifact_update
                else:
                    continue
                ev: Event | None = self._translator.translate(raw)
                if ev is not None:
                    yield ev

        except TimeoutException as e:
            raise AgentClientError(f"Agent request timed out: {e}") from e
        except HTTPError as e:
            raise AgentClientError(f"Agent HTTP error: {e}") from e
        except Exception as e:
            raise AgentClientError(f"Agent client error: {e}") from e
