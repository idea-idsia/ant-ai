from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from ant_ai.core.events import TopologyLink as Link
from ant_ai.topology.participant import Envelope, Reaction

__all__ = [
    "Activation",
    "EdgeAction",
    "EdgeKind",
    "GraphEdge",
    "InteractionGraph",
    "Link",
]

type EdgeKind = Literal["generates", "delivers", "visible", "invokes", "intervenes"]
"""What an edge records.

- `generates` / `delivers` connect activation and message **node ids**, and are
  what make the record bipartite rather than a flat agent-to-agent edge list.
- `visible` / `invokes` connect **participant names**, and record the
  distinction between reachability the topology granted and the calls an agent
  actually chose to make.
- `intervenes` records a rewrite by a stage: source is the message (or
  participant) acted on, destination the recipient it was aimed at.
"""

type EdgeAction = Reaction | Literal["inject", "emit"]
"""What was done to a message on an edge.

On a `delivers` edge it is the recipient's own reaction — `consume` for a
delivery it took in, `wait` for one it kept for later, `discard` for one it
dropped, `reroute` for one it handed on. On an `intervenes` edge it is a stage
doing the same kinds of thing from outside. The vocabulary is shared because
the acts are the same acts; who performed them is the edge kind.
"""


def _now() -> datetime:
    return datetime.now(UTC)


class Activation(BaseModel):
    """One participant taking one turn.

    Per-round, not per-agent: `architect@round1` and `architect@round2` are
    distinct nodes, so the same agent at different times never collapses into
    a single vertex.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    participant: str
    round: int = 0
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    error: str | None = None
    """Set when the participant raised. Recorded rather than swallowed — a
    failed activation is structural signal, not noise."""


class GraphEdge(BaseModel):
    src: str
    dst: str
    kind: EdgeKind
    round: int = 0
    action: EdgeAction | None = None
    weight: float | None = None
    reason: str | None = None
    """Why this edge exists, in words.

    Carried on the edge itself so replaying the graph reconstructs not just
    who talked to whom but why, with no separate log to keep in sync.
    """


class InteractionGraph(BaseModel):
    """A time-evolving record of what a run actually did.

    Bipartite by construction — activations and messages are separate nodes —
    because a flat agent-to-agent edge list cannot express message lineage, and
    lineage is what any causal question about a run has to be asked of.

    The methods here answer *what happened*: who acted, what they produced, what
    reached whom, what was consumed. They deliberately stop short of *what it
    means* — whether an unconsumed message is a failure, how long is too long for
    one to sit undelivered — because that is a published method's claim, not the
    record's. Those interpretations live with the strategy making them, in
    `ant_ai.topology.builtins`.

    Deliberately plain pydantic with no arbitrary types, so a whole run
    round-trips through `model_dump_json()` and an analysis can be developed and
    tested against recorded traces with no agents running.
    """

    activations: dict[str, Activation] = Field(default_factory=dict)
    messages: dict[str, Envelope] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)

    def record_activation(self, participant: str, *, round: int) -> str:
        node = Activation(participant=participant, round=round)
        self.activations[node.id] = node
        return node.id

    def end_activation(self, activation_id: str, *, error: str | None = None) -> None:
        node = self.activations.get(activation_id)
        if node is None:
            return
        node.ended_at = _now()
        node.error = error

    def record_message(self, envelope: Envelope, *, activation_id: str) -> str:
        self.messages[envelope.id] = envelope
        self.edges.append(
            GraphEdge(
                src=activation_id,
                dst=envelope.id,
                kind="generates",
                round=envelope.round,
            )
        )
        return envelope.id

    def record_delivery(
        self,
        message_id: str,
        activation_id: str,
        *,
        round: int,
        weight: float | None = None,
        reason: str | None = None,
        action: EdgeAction | None = None,
    ) -> None:
        self.edges.append(
            GraphEdge(
                src=message_id,
                dst=activation_id,
                kind="delivers",
                round=round,
                weight=weight,
                reason=reason,
                action=action,
            )
        )

    def record_links(self, links: tuple[Link, ...], *, round: int) -> None:
        """Record a decided topology as `visible` edges.

        Takes links and a round rather than a plan, so the graph stays a plain
        record with no knowledge of the pipeline that produced it.
        """
        for link in links:
            self.edges.append(
                GraphEdge(
                    src=link.src,
                    dst=link.dst,
                    kind="visible",
                    round=round,
                    weight=link.weight,
                    reason=link.reason,
                )
            )

    def record_invocation(self, caller: str, callee: str, *, round: int) -> None:
        """Record that *caller* actually called *callee* — selection, not reachability."""
        self.edges.append(
            GraphEdge(src=callee, dst=caller, kind="invokes", round=round)
        )

    def record_emission(self, envelope: Envelope, *, reason: str = "") -> str:
        """Add a message a supervisor created, with no generating activation.

        Ordinary nodes, so that everything downstream — lineage, consumption,
        outstanding work — treats a repaired run exactly as it treats a healthy
        one.
        """
        self.messages[envelope.id] = envelope
        self.record_intervention(
            envelope.id,
            envelope.sender,
            action="emit",
            round=envelope.round,
            reason=reason,
        )
        return envelope.id

    def record_intervention(
        self,
        source: str,
        target: str,
        *,
        action: EdgeAction,
        round: int,
        reason: str = "",
    ) -> None:
        """Record a supervisor rewrite as a first-class edge.

        Healing that leaves no trace is unfalsifiable: without this, a run that
        was corrected and a run that never needed correcting are
        indistinguishable in the record, and a detector counting how often a
        message has been moved has nothing to count.
        """
        self.edges.append(
            GraphEdge(
                src=source,
                dst=target,
                kind="intervenes",
                round=round,
                action=action,
                reason=reason,
            )
        )

    # -- traversal --------------------------------------------------------
    #
    # Neutral primitives, so an analysis never walks the edge list itself and a
    # new one costs a few lines rather than a graph traversal. Nothing here
    # decides whether what it finds is a problem.

    def inputs_of(self, activation_id: str) -> tuple[str, ...]:
        """Message ids delivered to an activation."""
        return tuple(
            e.src
            for e in self.edges
            if e.kind == "delivers" and e.dst == activation_id and e.action != "discard"
        )

    def outputs_of(self, activation_id: str) -> tuple[str, ...]:
        """Message ids an activation generated."""
        return tuple(
            e.dst
            for e in self.edges
            if e.kind == "generates" and e.src == activation_id
        )

    def consumed(self) -> set[str]:
        """Ids of messages some activation actually took in."""
        return {
            e.src for e in self.edges if e.kind == "delivers" and e.action == "consume"
        }

    def discarded(self) -> set[str]:
        """Ids of messages that were dropped — by a recipient or by a stage."""
        return {
            e.src
            for e in self.edges
            if e.action == "discard" and e.kind in ("delivers", "intervenes")
        }

    def reroutes(self, message_id: str) -> int:
        """How often this message was handed on, by anyone.

        A recipient rerouting a message and a supervisor rerouting it are the
        same event from the message's point of view, and a detector counting
        how often one has been bounced has to count both or it is measuring
        the supervisor's behaviour rather than the message's.
        """
        return sum(
            1
            for e in self.edges
            if e.action == "reroute"
            and e.src == message_id
            and e.kind in ("delivers", "intervenes")
        )

    def delivered(self) -> set[str]:
        """Ids of messages that reached at least one recipient."""
        return {e.src for e in self.edges if e.kind == "delivers"}

    def generator_of(self, message_id: str) -> str | None:
        """Id of the activation that produced *message_id*."""
        for e in self.edges:
            if e.kind == "generates" and e.dst == message_id:
                return e.src
        return None

    def siblings(self, message_id: str) -> tuple[str, ...]:
        """Messages produced by the same activation, *message_id* included.

        A turn emits its contribution twice — once public, once private — and a
        delivering materialiser routes whichever the recipient should see. They
        are two visibilities of one event, not two events, so structural
        accounting has to settle them together. Without this, every public
        message in delivery mode looks permanently unconsumed and orphaned,
        and Early Termination fires on every round of a healthy run.
        """
        activation = self.generator_of(message_id)
        if activation is None:
            return (message_id,)
        return self.outputs_of(activation) or (message_id,)

    def _generators(self, ids: set[str]) -> set[str]:
        """Activations that produced any of *ids*."""
        return {a for mid in ids if (a := self.generator_of(mid)) is not None}

    def unsettled(self) -> tuple[Envelope, ...]:
        """Messages nothing has taken in and no stage discarded.

        Settled per *contribution*, not per message — see `siblings`. Discarded
        messages are excluded: a stage that dropped a message decided it was not
        work, and counting it here would make an intervention look like a fresh
        problem.

        A statement of fact, not of fault. Whether outstanding messages mean
        something has gone wrong is a strategy's call.
        """
        settled = self.consumed() | self.discarded()
        settled_by = self._generators(settled)
        return tuple(
            m
            for m in self.messages.values()
            if not m.terminal
            and m.id not in settled
            and self.generator_of(m.id) not in settled_by
        )

    def undelivered(self) -> tuple[Envelope, ...]:
        """Non-terminal messages that reached nobody.

        Counted per contribution: a private copy that was routed means the
        contribution was heard, whatever became of the public one. How long a
        message may sit here before that is a problem is a strategy's threshold,
        not the record's.
        """
        delivered = self.delivered()
        heard = self._generators(delivered)
        return tuple(
            m
            for m in self.messages.values()
            if not m.terminal
            and m.id not in delivered
            and self.generator_of(m.id) not in heard
        )

    def terminal_message(self) -> Envelope | None:
        """The message that claimed the task was complete, if any."""
        for message in self.messages.values():
            if message.terminal:
                return message
        return None

    def intervention_count(
        self, message_id: str, *, action: EdgeAction | None = None
    ) -> int:
        """How often a stage rewrote this message, optionally of one kind."""
        return sum(
            1
            for e in self.edges
            if e.kind == "intervenes"
            and e.src == message_id
            and (action is None or e.action == action)
        )

    def ancestors(self, message_id: str) -> set[str]:
        """Transitive closure of `Envelope.parents`, including *message_id* itself."""
        seen: set[str] = set()
        stack = [message_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            message = self.messages.get(current)
            if message is not None:
                stack.extend(message.parents)
        return seen

    def activations_in(self, round: int) -> tuple[Activation, ...]:
        return tuple(a for a in self.activations.values() if a.round == round)

    def links(self, round: int, *, kind: EdgeKind = "visible") -> list[Link]:
        """Project the graph to agent-to-agent links for one round."""
        return [
            Link(src=e.src, dst=e.dst, weight=e.weight or 1.0, reason=e.reason)
            for e in self.edges
            if e.round == round and e.kind == kind
        ]

    def unused_visibility(self, round: int) -> list[Link]:
        """Peers a participant could reach this round but never called.

        A matcher wiring agents nobody calls is a coordination pathology, and
        it is only observable because reachability and selection are recorded
        separately.
        """
        invoked = {
            (e.src, e.dst)
            for e in self.edges
            if e.round == round and e.kind == "invokes"
        }
        return [
            link
            for link in self.links(round, kind="visible")
            if (link.src, link.dst) not in invoked
        ]

    def in_neighbours(self, participant: str, round: int) -> list[str]:
        return [link.src for link in self.links(round) if link.dst == participant]

    def rounds(self) -> list[int]:
        return sorted({a.round for a in self.activations.values()})

    def snapshot(self, round: int) -> InteractionGraph:
        """The graph as of *round*."""
        activations = {k: v for k, v in self.activations.items() if v.round <= round}
        messages = {k: v for k, v in self.messages.items() if v.round <= round}
        return InteractionGraph(
            activations=activations,
            messages=messages,
            edges=[e for e in self.edges if e.round <= round],
        )

    def to_mermaid(
        self, *, round: int | None = None, kind: EdgeKind = "visible"
    ) -> str:
        """Render the agent-to-agent projection as a Mermaid flowchart."""
        rounds = [round] if round is not None else self.rounds()
        lines = ["flowchart LR"]
        for r in rounds:
            links = self.links(r, kind=kind)
            if not links:
                continue
            lines.append(f'    subgraph R{r}["Round {r}"]')
            lines.append("        direction TB")
            for link in links:
                label = f"|{link.weight:.2f}|" if link.weight is not None else ""
                lines.append(
                    f"        R{r}_{_ident(link.src)} -->{label} R{r}_{_ident(link.dst)}"
                )
            lines.append("    end")
        return "\n".join(lines)


def _ident(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)
