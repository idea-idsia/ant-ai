from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.server.tasks.database_task_store import DatabaseTaskStore
from a2a.types import AgentCard
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from starlette.applications import Starlette

from ant_ai.a2a.client.config import A2AConfig
from ant_ai.a2a.client.tool import A2AAgentTool
from ant_ai.a2a.server.app import A2AServer
from ant_ai.agent.agent import Agent
from ant_ai.workflow.workflow import Workflow


def _normalize_url(url: str) -> str:
    return url.rstrip("/") + "/"


def _primary_url(card: AgentCard) -> str:
    if not card.supported_interfaces:
        raise ValueError(f"AgentCard '{card.name}' has no supported interfaces")
    return card.supported_interfaces[0].url


class Colony(BaseModel):
    """
    Class defining the Colony. It's the world for the agents that are part of the system.
    """

    db_url: str | None = Field(default=None)

    _specs: dict[str, AgentSpec] = PrivateAttr(default_factory=dict)
    _edges: dict[str, dict[str, A2AConfig]] = PrivateAttr(default_factory=dict)
    _task_stores: TaskStores = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        self._task_stores = TaskStores(self.db_url)

    async def aclose(self) -> None:
        """Dispose the async engine, returning all connections to the pool."""
        await self._task_stores.aclose()

    def get_task_store(self, table) -> TaskStore:
        return self._task_stores.get(table)

    def agent(
        self,
        name: str,
        *,
        agent: Agent,
        workflow: Workflow[Any],
        card: AgentCard,
        stream_artifacts: bool = True,
    ) -> Colony:
        """Adds an agent to the colony.

        Args:
            name: Name of the agent in the system.
            agent: The agent to be registered.
            workflow: The agent's workflow that will guide the agent execution.
            card: The A2A card for the agent.
            stream_artifacts: Whether this agent's A2A server should translate
                ContentDeltaEvent into artifact-update chunks. See `A2AServer`.
                Defaults on since it's additive (the terminal message is
                always sent regardless) and harmless for peers that ignore
                it, like `A2AAgentTool`. Set False to skip the wire overhead
                for an agent that's only ever called by other agents.

        Raises:
            ValueError: If the agent is already registered.

        Returns:
            The Colony instance with the registered agent.
        """
        if name in self._specs:
            raise ValueError(f"Agent '{name}' already registered.")

        url = _primary_url(card)
        parsed = urlparse(url)
        self._specs[name] = AgentSpec(
            agent=agent,
            workflow=workflow,
            url=url,
            host=parsed.hostname or "",
            port=parsed.port or 80,
            card=card,
            stream_artifacts=stream_artifacts,
        )
        return self

    def collab(
        self,
        source: str,
        target: str,
        *,
        config: A2AConfig | None = None,
        mutual: bool = False,
    ) -> Colony:
        """
        Register collaboration edges.

        - If config is omitted, a2a_defaults are used.
        - If mutual=True, inserts both source->target and target->source.
        - If the same edge is added twice, the last config wins.

        Args:
            source: The agent that can call another.
            target: The agent that becomes the tool to be called.
            config: Configuration to be used for the connection. If None, defaults will be used, which assume the target agent serves A2A at the root of its URL with default settings. Defaults to None.
            mutual: If True then the ability to initiate the conversation is given to both. Defaults to False.

        Returns:
            Colony: The Colony instance with the new collaboration defined.
        """
        if source not in self._specs:
            raise KeyError(f"Unknown agent '{source}' in collaboration.")
        if target not in self._specs:
            raise KeyError(f"Unknown agent '{target}' in collaboration.")

        _config: A2AConfig = config or A2AConfig(endpoint=self._specs[target].url)
        self._add_edge(source, target, config=_config)
        if mutual:
            _config: A2AConfig = config or A2AConfig(endpoint=self._specs[source].url)
            self._add_edge(target, source, config=_config)
        return self

    def asgi(
        self,
        *,
        agent_name: str,
        use_fastapi: bool = True,
    ) -> FastAPI | Starlette:
        """Creates the A2A server, with the specified ASGI app for the given agent name.

        Args:
            agent_name: The name of the agent to create a app for.
            use_fastapi: If True then FastAPI is used to create the app. Defaults to True.

        Raises:
            KeyError: If the agent name is not registered in Colony.

        Returns:
            The ASGI app and server configured for the agent.
        """
        if agent_name not in self._specs:
            raise KeyError(f"Agent '{agent_name}' is not registered in Colony.")

        server: A2AServer = self._build_server(agent_name)
        self._wire_a2a_tools(agent_name, agent=server.agent)

        return server.fastapi_app() if use_fastapi else server.starlette_app()

    def _build_server(self, agent_name: str) -> A2AServer:
        spec: AgentSpec = self._specs[agent_name]
        task_store: TaskStore = self.get_task_store(agent_name)
        return A2AServer(
            agent=spec.agent,
            workflow=spec.workflow,
            host=spec.host,
            port=spec.port,
            agent_card=spec.card,
            task_store=task_store,
            stream_artifacts=spec.stream_artifacts,
        )

    def _add_edge(self, source: str, target: str, *, config: A2AConfig) -> None:
        if source not in self._specs:
            raise KeyError(f"Unknown agent '{source}' in collaboration.")
        if target not in self._specs:
            raise KeyError(f"Unknown agent '{target}' in collaboration.")

        self._edges.setdefault(source, {})[target] = config

    def _wire_a2a_tools(self, name: str, *, agent: Agent) -> None:
        """
        Wires the remotes agents as tool to the agent.
        Args:
            name: The name of the agent to wire tools for.
            agent: The agent to wire tools for.
        """
        outgoing: dict[str, A2AConfig] = self._edges.get(name, {})
        if not outgoing:
            return

        for target_key, cfg in outgoing.items():
            target_spec: AgentSpec = self._specs[target_key]
            if not _has_endpoint_tool(agent, target_spec.url):
                agent.add_tool(
                    A2AAgentTool.from_config(config=cfg, agent_card=target_spec.card)
                )

    def get_agent_host(self, agent_name: str) -> tuple[str, int]:
        """Get the base URL of the specified agent."""
        if agent_name not in self._specs:
            raise KeyError(f"Agent '{agent_name}' is not registered in the colony.")
        spec: AgentSpec = self._specs[agent_name]
        return spec.host, spec.port


class TaskStores:
    """Makes the task store of each agent: a table of `db_url`, else in memory."""

    def __init__(self, db_url: str | None) -> None:
        self._db_engine: AsyncEngine | None = (
            create_async_engine(db_url, echo=False, pool_pre_ping=True)
            if db_url
            else None
        )

    def get(self, table: str) -> TaskStore:
        if self._db_engine is not None:
            return DatabaseTaskStore(
                engine=self._db_engine, create_table=True, table_name=table
            )
        return InMemoryTaskStore()

    async def aclose(self) -> None:
        """Dispose the async engine, returning all connections to the pool."""
        if self._db_engine is not None:
            await self._db_engine.dispose()


def _has_endpoint_tool(agent: Agent, endpoint: str) -> bool:
    endpoint = _normalize_url(endpoint)
    return any(
        isinstance(t, A2AAgentTool)
        and t.config.endpoint
        and _normalize_url(t.config.endpoint) == endpoint
        for t in agent.tools
    )


class AgentSpec(BaseModel):
    """
    Specification of a remote agent.
    """

    agent: Agent
    workflow: Workflow
    url: str
    host: str
    port: int
    card: AgentCard
    stream_artifacts: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)
