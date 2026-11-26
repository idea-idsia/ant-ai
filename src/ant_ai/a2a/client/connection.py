from __future__ import annotations

import asyncio

from a2a.client import A2ACardResolver, Client, ClientConfig, create_client
from a2a.types import AgentCard
from httpx import AsyncClient, HTTPError

from ant_ai.a2a.client.config import A2AConfig


class AgentClientError(RuntimeError):
    pass


class A2AConnection:
    """Lazily opened connection to a remote agent: its card, HTTP client and SDK client."""

    def __init__(self, config: A2AConfig) -> None:
        self.config: A2AConfig = config
        self.agent_card: AgentCard | None = None
        self._httpx: AsyncClient | None = None
        self._client: Client | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def open(self, agent_card: AgentCard | None = None) -> Client:
        """Return the SDK client, connecting on first use.

        Args:
            agent_card: The remote agent's card, if already known; fetched otherwise.
        """
        if self._client is not None and self._httpx is not None:
            return self._client

        async with self._lock:
            if self._client is not None and self._httpx is not None:
                return self._client

            httpx_client = AsyncClient(timeout=self.config.timeout)
            try:
                client: Client = await self._build_client(httpx_client, agent_card)
            except Exception:
                await httpx_client.aclose()
                raise

            self._httpx = httpx_client
            self._client = client
            return client

    async def aclose(self) -> None:
        httpx_client: AsyncClient | None = self._httpx
        self._httpx = None
        self._client = None
        self.agent_card = None
        if httpx_client is not None:
            await httpx_client.aclose()

    async def _build_client(
        self, httpx_client: AsyncClient, agent_card: AgentCard | None
    ) -> Client:
        if agent_card is None:
            agent_card = await self._resolve_card(httpx_client)
        self.agent_card = agent_card

        cfg = ClientConfig(
            httpx_client=httpx_client,
            supported_protocol_bindings=list(self.config.supported_protocol_bindings),
            streaming=self.config.streaming,
        )
        return await create_client(agent_card, client_config=cfg)

    async def _resolve_card(self, httpx_client: AsyncClient) -> AgentCard:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=self.config.endpoint,
            agent_card_path=self.config.agent_card_path,
        )
        try:
            return await resolver.get_agent_card()
        except (HTTPError, Exception) as e:
            raise AgentClientError(f"Failed to resolve AgentCard: {e}") from e
