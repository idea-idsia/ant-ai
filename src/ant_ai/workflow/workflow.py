from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from ant_ai.agent.base import BaseAgent
from ant_ai.core.events import (
    CompletedEvent,
    Event,
    EventOrigin,
    StartEvent,
    UpdateEvent,
)
from ant_ai.core.types import InvocationContext, State
from ant_ai.observer import obs

START = "START"
END = "END"

type NodeYield = Event | State
"""What a node can yield"""

type NodeResult = AsyncIterator[NodeYield]
"""What a node can yield when it is awaited"""

type NodeAction = Callable[
    [BaseAgent, State, InvocationContext | None],
    NodeResult | Awaitable[NodeResult],
]
"""Interface for the nodes in the Workflow"""

type RouterAction = Callable[
    [BaseAgent, State, InvocationContext | None],
    str | Awaitable[str],
]
"""Interface for a conditional edge in the Workflow"""


@dataclass
class _RunState:
    """Mutable execution context for a single workflow run."""

    state: State
    step: int = 0


def _final_content(state: State) -> str | None:
    return state.last_message.content if state.messages else None


async def _maybe_await(x: Any) -> Any:
    if inspect.isawaitable(x):
        return await cast(Awaitable[Any], x)
    return x


class Workflow(BaseModel):
    """
    Graph that orchestrates agent behaviour across a sequence of nodes.

    Iterates over nodes, passing State, Agent, and InvocationContext
    between them. Supports static edges and conditional (router) edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: dict[str, NodeAction] = Field(default_factory=dict)
    edges: dict[str, str] = Field(default_factory=dict)
    conditional_edges: dict[str, RouterAction] = Field(default_factory=dict)
    max_steps: int = Field(
        default=100,
        description="Hard cap on graph traversal steps.",
    )

    def add_node(self, name: str, action: NodeAction) -> Workflow:
        if name.upper() in (START, END):
            raise ValueError(f"'{name}' is reserved.")
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists.")
        if not callable(action):
            raise TypeError("Node action must be callable.")
        self.nodes[name] = action
        return self

    def add_edge(self, src: str, dst: str) -> Workflow:
        if src in self.conditional_edges:
            raise ValueError("Cannot mix static and conditional edges.")

        self._validate_src_node(src)
        self._validate_dst_node(dst)

        self.edges[src] = dst
        return self

    def add_conditional_edge(self, src: str, router: RouterAction) -> Workflow:
        if not callable(router):
            raise TypeError("Router must be callable.")
        if src in self.edges:
            raise ValueError("Cannot mix static and conditional edges.")

        self._validate_src_node(src)
        self.conditional_edges[src] = router
        return self

    def _validate_src_node(self, name: str) -> None:
        if name == END:
            raise ValueError(f"'{END}' cannot have an outgoing edge.")
        if name != START and name not in self.nodes:
            raise ValueError(f"Unknown node '{name}'.")
        if name in self.edges or name in self.conditional_edges:
            raise ValueError(f"Node '{name}' already has an outgoing edge.")

    def _validate_dst_node(self, name: str) -> None:
        if name == START:
            raise ValueError("Cannot route back to START.")
        if name != END and name not in self.nodes:
            raise ValueError(f"Unknown node '{name}'.")

    def _validate_graph(self) -> None:
        if START not in self.edges and START not in self.conditional_edges:
            raise ValueError("START must have an outgoing edge.")

    async def _next(
        self,
        agent: BaseAgent,
        state: State,
        ctx: InvocationContext | None,
        current: str,
    ) -> str:
        if current in self.conditional_edges:
            nxt: str = await _maybe_await(
                self.conditional_edges[current](agent, state, ctx)
            )
            await obs.event("node.edge.router", src=current, dst=nxt)
            if not isinstance(nxt, str) or not nxt:
                raise RuntimeError("Router must return a non-empty str.")
            if nxt not in (START, END) and nxt not in self.nodes:
                raise RuntimeError(f"Unknown node '{nxt}'.")
            return nxt

        if current in self.edges:
            nxt = self.edges[current]
            await obs.event("node.edge.static", src=current, dst=nxt)
            return nxt

        raise RuntimeError(f"No outgoing edge from '{current}'.")

    async def _run_node(
        self,
        agent: BaseAgent,
        run: _RunState,
        ctx: InvocationContext | None,
        node: str,
    ) -> AsyncGenerator[Event]:
        action: NodeAction | None = self.nodes.get(node)
        if action is None:
            raise RuntimeError(f"Unknown node '{node}'.")

        yield UpdateEvent(
            origin=EventOrigin(layer="workflow", node=node, run_step=run.step),
            content=f"Starting action '{node}'",
        )

        with obs.bind(node=node):
            node_input = run.state.messages[-1].content if run.state.messages else None
            await obs.event(
                "node.start", node=node, run_step=run.step, input=node_input
            )

            try:
                result: NodeYield | None = await _maybe_await(
                    action(agent, run.state, ctx)
                )

                if isinstance(result, AsyncIterator):
                    async for item in result:
                        if isinstance(item, Event):
                            if item.origin.node is None:
                                item.origin.node = node
                            yield item
                        elif isinstance(item, State):
                            run.state = item
                        else:
                            raise RuntimeError(f"Invalid yield from node '{node}'.")
                else:
                    if result is not None and not isinstance(result, State):
                        raise RuntimeError(f"Invalid return from node '{node}'.")
                    if result is not None:
                        run.state = result

            except Exception as exc:
                await obs.exception("node.error", exc, node=node, run_step=run.step)
                raise

            node_output = run.state.messages[-1].content if run.state.messages else None
            await obs.event(
                "node.end", node=node, run_step=run.step, output=node_output
            )

        yield UpdateEvent(
            origin=EventOrigin(layer="workflow", node=node, run_step=run.step),
            content=f"Completed action '{node}'",
        )

    async def _arun(
        self,
        agent: BaseAgent,
        *,
        ctx: InvocationContext | None,
        start_at: str,
        run: _RunState,
    ) -> AsyncGenerator[Event]:
        self._validate_graph()

        current: str = start_at

        with obs.bind(
            session_id=ctx.session_id if ctx else "",
            agent_name=agent.name,
            user_input=run.state.last_message,
        ):
            await obs.event(
                "workflow.start",
                agent_name=agent.name,
                session_id=ctx.session_id if ctx else None,
                input=run.state.last_message.content,
                start_at=start_at,
                max_steps=self.max_steps,
            )

            yield StartEvent(
                origin=EventOrigin(layer="workflow", node=current, run_step=run.step),
                content="Workflow started",
            )

            while True:
                if current == END:
                    final_content = _final_content(run.state)
                    await obs.event(
                        "workflow.end",
                        steps=run.step,
                        finish_reason="completed",
                        output=final_content,
                    )
                    yield CompletedEvent(
                        origin=EventOrigin(
                            layer="workflow", node=current, run_step=run.step
                        ),
                        content=final_content or "Workflow completed",
                    )
                    return

                if run.step >= self.max_steps:
                    await obs.event(
                        "workflow.max_steps",
                        steps=run.step,
                        max_steps=self.max_steps,
                        finish_reason="max_steps",
                        output=_final_content(run.state),
                    )
                    raise RuntimeError("Max steps exceeded.")

                if current != START:
                    run.step += 1
                    async for ev in self._run_node(agent, run, ctx, current):
                        yield ev

                current = await self._next(agent, run.state, ctx, current)

    async def ainvoke(
        self,
        agent: BaseAgent,
        *,
        ctx: InvocationContext | None = None,
        start_at: str = START,
        state: State | None = None,
    ) -> State:
        run = _RunState(state=state or State())
        async for _ in self._arun(agent, ctx=ctx, start_at=start_at, run=run):
            pass
        return run.state

    def stream(
        self,
        agent: BaseAgent,
        *,
        ctx: InvocationContext | None = None,
        start_at: str = START,
        state: State | None = None,
    ) -> AsyncIterator[Event]:
        run = _RunState(state=state or State())
        return self._arun(agent, ctx=ctx, start_at=start_at, run=run)
