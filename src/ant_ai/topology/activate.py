"""Read-only selection, recorded.

Retrieval, tool selection, skill lookup and peer binding are the same operation
under four names: a query picks a subset of a persistent graph, that subset is
what the next decision is made from, and nothing about the graph changes. The
paper calls it `Activate` (Eq. 11) and makes it a first-class operator precisely
because treating it as "not an edit, therefore not worth recording" is what
leaves a system unable to answer whether a decision was grounded in the right
evidence, or in evidence that did not exist yet.

Which is what this module fixes. `RecordingMemory` wraps any `Memory` so that a
retrieval leaves a `SupportSubgraph` behind, and `record_support` is the same
two lines for callers that already know what they selected — the round loop,
recording the reachability a materialiser just bound.

Nothing here changes what any of those components return. A wrapped memory
retrieves exactly what the memory underneath it retrieved, minus anything the
temporal guard rules out.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any

from pydantic import ConfigDict, Field, PrivateAttr, SkipValidation

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext
from ant_ai.memory.protocol import Memory
from ant_ai.topology.log import RewriteLog
from ant_ai.topology.rewrite import NodeKind, Rewrite
from ant_ai.topology.state import StateGraph, SupportSubgraph

__all__ = ["RecordingMemory", "memory_node_id", "record_support"]


def memory_node_id(content: str, *, namespace: str = "mem") -> str:
    """A stable id for a remembered message.

    Content-addressed, so the same fact written twice is one node and a
    retrieval of it is attributable across runs. A counter would make every
    replay produce a different graph and every cross-run metric meaningless.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{namespace}:{digest}"


def record_support(
    state: StateGraph,
    *,
    kind: NodeKind,
    id: str,  # noqa: A002
    nodes: tuple[str, ...],
    at: int,
    query: str = "",
    reason: str = "",
    log: RewriteLog | None = None,
) -> SupportSubgraph | None:
    """Record one activation, through the log when there is one.

    Through the log rather than straight onto the graph, so that an activation
    appears in the same ordered trail as the edits around it — a cascade that
    begins with a retrieval is only legible if the retrieval is in the record
    next to what it caused.
    """
    rewrite = Rewrite.activate(kind, id, query=query, nodes=nodes, reason=reason)
    rewrite = rewrite.model_copy(update={"at": at})
    if log is not None:
        log.record(rewrite, state)
    else:
        state.apply(rewrite)
    return state.supports[-1] if state.supports else None


class RecordingMemory(Memory):
    """Any memory backend, with its reads and writes visible in the state graph.

    A drop-in `Memory`: pass one to an agent exactly as you would the backend it
    wraps. What it adds is that `update` inserts memory nodes and `retrieve`
    leaves a `SupportSubgraph` naming what it grounded the next decision in.

    `leakage_guard` is on by default and is the reason this is worth having at
    all. A backend ranks by similarity and has no notion of when a decision is
    being made, so replaying a recorded run — or evaluating one against a state
    graph that outlived it — will cheerfully return evidence written after the
    decision under test. The guard drops those, and `evaluate.leaked` reports
    what a run without the guard would have used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Annotated[Memory, SkipValidation]
    state: StateGraph = Field(default_factory=StateGraph)
    log: RewriteLog | None = None
    clock: int = Field(
        default=0,
        description="Decision time for the next call. The round, in an ensemble.",
    )
    leakage_guard: bool = Field(
        default=True,
        description="Drop retrieved memories that did not exist at `clock`.",
    )
    namespace: str = "mem"

    _activations: int = PrivateAttr(default=0)

    def tick(self, clock: int) -> None:
        """Advance decision time. Called by whatever owns the loop."""
        self.clock = clock

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        ctx: InvocationContext | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        results = await self.inner.retrieve(query, top_k=top_k, ctx=ctx, **kwargs)

        kept: list[Message] = []
        nodes: list[str] = []
        for message in results:
            node_id = memory_node_id(message.content or "", namespace=self.namespace)
            node = self.state.nodes.get(node_id)
            if (
                self.leakage_guard
                and node is not None
                and not node.alive_at(self.clock)
            ):
                continue
            kept.append(message)
            nodes.append(node_id)

        self._activations += 1
        record_support(
            self.state,
            kind="memory",
            id=f"{self.namespace}:act:{self.clock}:{self._activations}",
            nodes=tuple(nodes),
            at=self.clock,
            query=query,
            reason="memory retrieval",
            log=self.log,
        )
        return kept

    async def update(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        **kwargs: Any,
    ) -> None:
        await self.inner.update(messages, ctx=ctx, **kwargs)
        for message in messages:
            content = message.content or ""
            if not content:
                continue
            node_id = memory_node_id(content, namespace=self.namespace)
            rewrite = Rewrite.insert(
                "memory",
                node_id,
                label=content[:80],
                role=str(getattr(message, "role", "")),
                content=content,
            ).model_copy(update={"at": self.clock, "reason": "memory write"})
            if node_id in self.state.nodes:
                continue
            if self.log is not None:
                self.log.record(rewrite, self.state)
            else:
                self.state.apply(rewrite)
