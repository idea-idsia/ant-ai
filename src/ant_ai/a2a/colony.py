from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore, TaskStore
from a2a.types import AgentCard
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from starlette.applications import Starlette

from ant_ai.a2a.agent import A2AAgentTool
from ant_ai.a2a.config import A2AConfig
from ant_ai.a2a.server import A2AServer
from ant_ai.agent.agent import Agent
from ant_ai.workflow.workflow import Workflow

if TYPE_CHECKING:
    # Type-only: `ant_ai.topology` imports `ant_ai.a2a.agent`, so importing it
    # at module level would cycle. The names are still real to a type checker
    # and to an IDE, which is the point — they used to be `Any`.
    from ant_ai.topology.heal import Detector
    from ant_ai.topology.materialise import TopologyMaterialiser
    from ant_ai.topology.runtime import Ensemble
    from ant_ai.topology.strategy import TopologyStrategy


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
    _db_engine: AsyncEngine | None = PrivateAttr(default=None)
    _topology: Any = PrivateAttr(default=None)
    _detectors: list[Any] = PrivateAttr(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        if self.db_url:
            self._db_engine: AsyncEngine = create_async_engine(
                self.db_url, echo=False, pool_pre_ping=True
            )

    async def aclose(self) -> None:
        """Dispose the async engine, returning all connections to the pool."""
        if self._db_engine is not None:
            await self._db_engine.dispose()

    def get_task_store(self, table) -> TaskStore:
        if self.db_url and self._db_engine:
            return DatabaseTaskStore(
                engine=self._db_engine, create_table=True, table_name=table
            )  # ty: ignore[invalid-return-type]
        else:
            return InMemoryTaskStore()

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

    def topology(
        self,
        strategy: TopologyStrategy,
        *,
        detectors: list[Detector] | None = None,
    ) -> Colony:
        """Declare an adaptive topology for this colony.

        Without this call the colony behaves exactly as before: `collab()` edges
        are the static topology.

        Args:
            strategy: A `TopologyStrategy` — see `ant_ai.topology.builtins`.
                Compose two with `DyTopo(...) | DigToHeal()`.
            detectors: Structural failure detectors appended as one extra `Heal`
                stage, for a one-off that does not warrant its own strategy.

        Returns:
            The Colony instance, for chaining.
        """
        self._topology = strategy
        self._detectors = list(detectors or [])
        return self

    def ensemble(
        self,
        *,
        local: bool = True,
        use_workflows: bool | None = None,
        max_rounds: int | None = None,
        materialiser: TopologyMaterialiser | None = None,
    ) -> Ensemble:
        """Build an `Ensemble` over the registered agents.

        Args:
            local: True builds `LocalParticipant`s that run in this process, so no
                servers are needed. False builds `A2AParticipant`s that drive the
                deployed colony over the wire.
            use_workflows: Whether each participant's turn runs its registered
                workflow. Running it is faithful to how a colony serves a request,
                but `Workflow.stream` takes no response schema, so such a
                participant answers with one plain public message: no query/key
                descriptors, no addressed messages, no declared reactions and
                nothing ever submitted. None therefore means *pick*: the agent is
                invoked directly when the pipeline has a component that reads any
                of those, and the workflow runs when it does not. Pass True or
                False to state the choice yourself; True with such a strategy is
                honoured, and a matcher warns once its fallback is total.
            max_rounds: Override the strategy's round cap.
            materialiser: Override how the topology is realised.

        Returns:
            A configured `Ensemble`.
        """
        # Imported lazily: `ant_ai.topology` imports `ant_ai.a2a.agent`, which
        # initialises this package, so a module-level import would cycle.
        from ant_ai.topology.builtins.shapes import Baseline
        from ant_ai.topology.heal import Heal
        from ant_ai.topology.materialise import VisibilityMaterialiser
        from ant_ai.topology.participant import A2AParticipant, LocalParticipant
        from ant_ai.topology.runtime import Ensemble

        strategy = self._topology or Baseline()
        pipeline = strategy.pipeline()
        if max_rounds is not None:
            pipeline = pipeline.model_copy(update={"max_rounds": max_rounds})
        if materialiser is not None:
            pipeline = pipeline.model_copy(update={"materialiser": materialiser})
        if self._detectors:
            pipeline = pipeline.model_copy(
                update={"stages": [*pipeline.stages, Heal(detectors=self._detectors)]}
            )

        if use_workflows is None:
            # A workflow-driven turn cannot carry a response schema, so it degrades
            # to one plain public message: no descriptors, no addressing, no
            # reactions, nothing ever submitted. A strategy built on any of those
            # would run to completion and do nothing — a matcher scoring static
            # card text, or a detector that never sees a symptom. Deciding here
            # rather than defaulting to True is what keeps `colony.ensemble()` from
            # quietly being a static, unsupervised baseline.
            use_workflows = not pipeline.needs_structured_turns

        if (
            not local
            and pipeline.stages
            and isinstance(pipeline.materialiser, VisibilityMaterialiser)
        ):
            # Visibility means reachability *is* the peer tool set, and there is no
            # A2A operation for attaching a tool to an agent in another process, so
            # every remote participant reports itself unbindable and the decided
            # topology constrains nothing at all. Gated on there being a stage: a
            # colony with no strategy decides nothing, and its remote agents stay
            # wired as their servers wired them, which is the pre-topology
            # behaviour rather than a silent failure.
            warnings.warn(
                "Remote (A2A) participants cannot be rebound, so a topology "
                "materialised as peer tools has no effect on them. Pass "
                "`materialiser=DeliveryMaterialiser()` to route their messages "
                "instead, or build local participants.",
                RuntimeWarning,
                stacklevel=2,
            )

        participants: dict[str, Any] = {}
        for name, spec in self._specs.items():
            if local:
                participants[name] = LocalParticipant(
                    spec.agent,
                    workflow=spec.workflow if use_workflows else None,
                    name=name,
                    max_depth=pipeline.max_depth,
                )
            else:
                participants[name] = A2AParticipant(
                    A2AConfig(endpoint=spec.url),
                    spec.card,
                    name=name,
                    max_depth=pipeline.max_depth,
                )

        return Ensemble(
            participants=participants,
            pipeline=pipeline,
            # Round 0 is seeded from the declared collab() edges, so the very
            # first turn behaves exactly as a colony does today. A colony with no
            # strategy has no stage writing links, so these govern every round —
            # which is precisely the pre-topology behaviour.
            seed=self._declared_links(),
            provenance=strategy.provenance(),
        )

    def _declared_links(self) -> tuple[Any, ...]:
        """`collab()` edges as information-flow links.

        Note the reversal. `collab(source, target)` means *source may call
        target*, so target is the one offering — and `Link` direction is
        information flow. The edge therefore becomes `Link(src=target, dst=source)`.
        """
        from ant_ai.topology.graph import Link

        return tuple(
            Link(src=target, dst=source, reason="declared via Colony.collab()")
            for source, targets in self._edges.items()
            for target in targets
        )

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
            if self._agent_has_endpoint_tool(agent, target_spec.url):
                continue

            tool: A2AAgentTool = A2AAgentTool.from_config(
                config=cfg, agent_card=target_spec.card
            )
            agent.add_tool(tool)

    def _agent_has_endpoint_tool(self, agent: Agent, endpoint: str) -> bool:
        endpoint = _normalize_url(endpoint)
        for t in agent.tools:
            if (
                isinstance(t, A2AAgentTool)
                and t.config.endpoint
                and _normalize_url(t.config.endpoint) == endpoint
            ):
                return True
        return False

    def get_agent_host(self, agent_name: str) -> tuple[str, int]:
        """Get the base URL of the specified agent."""
        if agent_name not in self._specs:
            raise KeyError(f"Agent '{agent_name}' is not registered in the colony.")
        spec: AgentSpec = self._specs[agent_name]
        return spec.host, spec.port


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
