"""The agent-state graph: what evolves, as opposed to what happened.

`InteractionGraph` is the run's episodic record — activations, messages, and the
edges between them. It answers *what happened in this run*. This is the other
half: the persistent, typed graph of the agent itself — its memories, tools,
skills, workflows and the agents holding them — which answers *what the system
is now*, and which outlives any single run.

Keeping them apart is deliberate. Collapsing the two would make the trace grow
without bound and would make the question "what does this agent know?"
answerable only by replaying every round it has ever taken. They share node ids
and the same edge families, so a cascade that starts at an observed message and
ends at a distilled skill crosses between them by following an edge rather than
by changing representation halfway.

This is `G(t) = (V(t), E(t), X_V(t), X_E(t), Y(t))` from arXiv:2608.18104, with
one addition the paper implies and does not name: every node and edge carries a
validity interval, so `at(t)` is a real operation rather than a convention. That
is what the leakage-free temporal protocol (their V-B) needs, and it is why
`delete` tombstones by default instead of dropping — a record that forgets it
ever held something cannot be audited, and cannot be rolled back.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ant_ai.topology.problem import Problem
from ant_ai.topology.rewrite import EdgeFamily, EdgeRef, NodeKind, NodeRef, Rewrite

__all__ = [
    "Schema",
    "SchemaViolation",
    "StateEdge",
    "StateGraph",
    "StateNode",
    "SupportSubgraph",
]


class SchemaViolation(ValueError):
    """A rewrite that the graph's schema forbids.

    Carries the problems rather than only their text, for the same reason
    `TopologyConfigurationError` does: a caller generating rewrites
    programmatically branches on `code`.
    """

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        super().__init__("\n".join(p.render() for p in problems))


class StateNode(BaseModel):
    """One typed component of the agent, valid over an interval."""

    id: str
    kind: NodeKind
    label: str = ""
    attrs: dict[str, Any] = Field(default_factory=dict)
    valid_from: int = 0
    valid_to: int | None = Field(
        default=None,
        description="Exclusive upper bound; None means still valid. Set rather "
        "than dropped on delete, so an audit can still see what was removed.",
    )

    def alive_at(self, t: int) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


class StateEdge(BaseModel):
    """One typed relation, valid over an interval."""

    src: str
    dst: str
    family: EdgeFamily = "dependency"
    attrs: dict[str, Any] = Field(default_factory=dict)
    valid_from: int = 0
    valid_to: int | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.family)

    def alive_at(self, t: int) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


class Schema(BaseModel):
    """Which edge families may connect which node kinds.

    The typing half of the paper's "schema-constrained rewrites". It is checked
    on every edge operator, because the failure it prevents is silent: an edge
    written between the wrong two kinds does not raise anywhere, it simply makes
    every later traversal — affected scope, activation, rollback — quietly wrong
    about what depends on what.

    `("*", "*")` is the wildcard, and provenance uses it: derivation can relate
    anything to anything, which is exactly why it is the family audit questions
    are asked of.
    """

    allowed: dict[str, tuple[tuple[str, str], ...]] = Field(
        default_factory=lambda: {
            "communication": (("agent", "agent"),),
            "dependency": (
                ("agent", "tool"),
                ("agent", "skill"),
                ("agent", "memory"),
                ("agent", "workflow"),
                ("skill", "tool"),
                ("skill", "skill"),
                ("workflow", "skill"),
                ("workflow", "tool"),
                ("memory", "memory"),
            ),
            "provenance": (("*", "*"),),
        }
    )

    def allows(self, family: EdgeFamily, src: NodeKind, dst: NodeKind) -> bool:
        pairs = self.allowed.get(family, ())
        return any((a == "*" or a == src) and (b == "*" or b == dst) for a, b in pairs)


class SupportSubgraph(BaseModel):
    """What one read-only activation selected, and when.

    The paper's `G_act(q, t_q)`. Recorded rather than returned and forgotten,
    because every question their evaluation section asks about retrieval —
    was the evidence valid at decision time, was it the right evidence, was it
    more evidence than the answer needed — is a question about this object, and
    a system that does not keep one can answer none of them.
    """

    id: str
    query: str = ""
    kind: NodeKind = "memory"
    at: int = 0
    nodes: tuple[str, ...] = ()
    reason: str = ""

    def dice(self, gold: Iterable[str]) -> float:
        """Support-subgraph accuracy against an acceptable gold set (their Eq. 18).

        Dice rather than recall, because recall alone rewards activating the
        whole graph — the paper's "favours compact evidence use" is the half
        that a precision term is doing.
        """
        selected, reference = set(self.nodes), set(gold)
        total = len(selected) + len(reference)
        if not total:
            return 1.0
        return 2 * len(selected & reference) / total


class StateGraph(BaseModel):
    """The typed, time-indexed graph every rewrite is applied to.

    Plain pydantic with no arbitrary types, like `InteractionGraph`, so a whole
    evolved state round-trips through `model_dump_json()` and an analysis runs
    against a recorded one with no agents.
    """

    nodes: dict[str, StateNode] = Field(default_factory=dict)
    edges: list[StateEdge] = Field(default_factory=list)
    supports: list[SupportSubgraph] = Field(default_factory=list)
    schema_: Schema = Field(
        default_factory=Schema,
        alias="schema",
        description="Trailing underscore because `schema` shadows a pydantic "
        "method; the alias is what serialisation and construction use.",
    )

    model_config = ConfigDict(populate_by_name=True)

    # -- reading ----------------------------------------------------------

    def kind_of(self, node_id: str) -> NodeKind | None:
        node = self.nodes.get(node_id)
        return node.kind if node else None

    def of_kind(
        self, kind: NodeKind, *, at: int | None = None
    ) -> tuple[StateNode, ...]:
        return tuple(
            n
            for n in self.nodes.values()
            if n.kind == kind and (at is None or n.alive_at(at))
        )

    def live_edges(self, *, at: int | None = None) -> tuple[StateEdge, ...]:
        """Edges in force — right now, or as of *at*."""
        if at is None:
            return tuple(e for e in self.edges if e.valid_to is None)
        return tuple(e for e in self.edges if e.alive_at(at))

    def _edge(self, ref: EdgeRef) -> StateEdge | None:
        for edge in self.edges:
            if edge.key == (ref.src, ref.dst, ref.family) and edge.valid_to is None:
                return edge
        return None

    def _attached(self, node_id: str) -> tuple[StateEdge, ...]:
        """Live edges with *node_id* at either end — the dangling condition's subject."""
        return tuple(
            e for e in self.edges if e.valid_to is None and node_id in (e.src, e.dst)
        )

    def at(self, t: int) -> StateGraph:
        """The graph as it was at logical time *t*.

        The leakage-free protocol in one method: an evaluation that asks a
        question at `t_q` asks it of `graph.at(t_q)`, and evidence written later
        is not merely down-ranked, it is absent.
        """
        return StateGraph(
            nodes={k: v for k, v in self.nodes.items() if v.alive_at(t)},
            edges=[e for e in self.edges if e.alive_at(t)],
            supports=[s for s in self.supports if s.at <= t],
            schema=self.schema_,
        )

    def affected(
        self,
        node_id: str,
        *,
        families: tuple[EdgeFamily, ...] = ("dependency", "provenance"),
        at: int | None = None,
    ) -> set[str]:
        """Everything downstream of a node — their affected-scope analysis (V-D).

        What must be revalidated after a local update, and what a deletion would
        orphan. Follows dependency and provenance by default and not
        communication: who an agent may talk to is decided fresh every round, so
        treating it as a dependency would make every local edit look like it
        affects the whole colony.
        """
        out: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.family not in families:
                continue
            if at is not None and not edge.alive_at(at):
                continue
            if at is None and edge.valid_to is not None:
                continue
            out.setdefault(edge.dst, []).append(edge.src)
            if edge.family == "dependency":
                out.setdefault(edge.src, []).append(edge.dst)

        seen: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            for neighbour in out.get(current, ()):
                if neighbour not in seen and neighbour != node_id:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen

    # -- writing ----------------------------------------------------------

    def check(self, rewrite: Rewrite) -> list[Problem]:
        """Everything provably wrong with this edit, before it is applied.

        Errors only. A rewrite that is merely surprising — updating a node that
        does not exist yet, say — is applied, because an evolving graph whose
        components arrive out of order is the normal case rather than a bug.
        """
        problems: list[Problem] = []
        if isinstance(rewrite.target, EdgeRef):
            problems.extend(self._check_edge(rewrite, rewrite.target))
        else:
            problems.extend(self._check_node(rewrite, rewrite.target))
        return problems

    def _check_edge(self, rewrite: Rewrite, ref: EdgeRef) -> list[Problem]:
        if rewrite.op in ("unlink", "edge_feature_update", "rewire"):
            return []
        src_kind, dst_kind = self.kind_of(ref.src), self.kind_of(ref.dst)
        if src_kind is None or dst_kind is None:
            # An edge between nodes the graph has not been told about yet is how
            # every seeded topology starts; the endpoints are implied agents.
            return []
        if self.schema_.allows(ref.family, src_kind, dst_kind):
            return []
        return [
            Problem(
                code="E101",
                level="error",
                message=(
                    f"A '{ref.family}' edge from a {src_kind} to a {dst_kind} is "
                    "not allowed by the schema."
                ),
                hint=(
                    "Use the family that fits the relation — 'dependency' for "
                    "what a component needs, 'provenance' for what it was "
                    "derived from — or widen `StateGraph.schema_.allowed`."
                ),
            )
        ]

    def _check_node(self, rewrite: Rewrite, ref: NodeRef) -> list[Problem]:
        if rewrite.op != "delete":
            return []
        attached = self._attached(ref.id)
        if not attached or rewrite.payload.get("cascade"):
            return []
        return [
            Problem(
                code="E102",
                level="error",
                message=(
                    f"Deleting {ref.kind} '{ref.id}' would leave "
                    f"{len(attached)} edge(s) dangling."
                ),
                hint=(
                    "Unlink its edges first, or pass `cascade=True` to have them "
                    "closed with it."
                ),
            )
        ]

    def apply(self, rewrite: Rewrite) -> Rewrite | None:
        """Apply one edit and return the rewrite that would undo it.

        The inverse is produced *here* rather than by `Rewrite.invert()` because
        only the graph knows the before-state: undoing a `feature_update` means
        restoring attributes the edit itself never carried. That is also what
        makes rollback exact rather than approximate.

        Raises:
            SchemaViolation: If `check` finds an error.
        """
        problems = self.check(rewrite)
        if problems:
            raise SchemaViolation(problems)
        if isinstance(rewrite.target, NodeRef):
            return self._apply_node(rewrite, rewrite.target)
        return self._apply_edge(rewrite, rewrite.target)

    def _apply_node(self, rewrite: Rewrite, ref: NodeRef) -> Rewrite | None:
        if rewrite.op == "activate":
            self.supports.append(
                SupportSubgraph(
                    id=ref.id,
                    kind=ref.kind,
                    query=str(rewrite.payload.get("query", "")),
                    at=rewrite.at,
                    nodes=tuple(rewrite.payload.get("nodes", ())),
                    reason=rewrite.reason,
                )
            )
            return None  # read-only: there is nothing to undo

        if rewrite.op == "insert":
            self.nodes[ref.id] = StateNode(
                id=ref.id,
                kind=ref.kind,
                label=str(rewrite.payload.get("label", "")),
                attrs={k: v for k, v in rewrite.payload.items() if k != "label"},
                valid_from=rewrite.at,
            )
            return Rewrite.delete(ref.kind, ref.id, cascade=True)

        node = self.nodes.get(ref.id)
        if node is None:
            return None

        if rewrite.op == "delete":
            node.valid_to = rewrite.at
            for edge in self._attached(ref.id):
                edge.valid_to = rewrite.at
            return Rewrite.insert(ref.kind, ref.id, label=node.label, **node.attrs)

        if rewrite.op == "feature_update":
            before = dict(node.attrs)
            node.attrs = (
                dict(rewrite.payload)
                if rewrite.mode == "replace"
                else {**node.attrs, **rewrite.payload}
            )
            return Rewrite.feature_update(ref.kind, ref.id, mode="replace", **before)

        # merge: archive both originals under the consolidated node, with
        # provenance edges pointing back at what it was built from.
        consolidated = rewrite.to or f"{ref.id}+merged"
        self.nodes[consolidated] = StateNode(
            id=consolidated,
            kind=ref.kind,
            label=str(rewrite.payload.get("label", node.label)),
            attrs={k: v for k, v in rewrite.payload.items() if k != "label"},
            valid_from=rewrite.at,
        )
        self.edges.append(
            StateEdge(
                src=consolidated,
                dst=ref.id,
                family="provenance",
                valid_from=rewrite.at,
                attrs={"reason": rewrite.reason or "merged"},
            )
        )
        node.attrs = {**node.attrs, "archived": True}
        return Rewrite.delete(ref.kind, consolidated, cascade=True)

    def _apply_edge(self, rewrite: Rewrite, ref: EdgeRef) -> Rewrite | None:
        existing = self._edge(ref)

        if rewrite.op == "link":
            if existing is not None:
                return None
            self.edges.append(
                StateEdge(
                    src=ref.src,
                    dst=ref.dst,
                    family=ref.family,
                    attrs=dict(rewrite.payload),
                    valid_from=rewrite.at,
                )
            )
            return Rewrite.unlink(ref.src, ref.dst, family=ref.family)

        if existing is None:
            return None

        if rewrite.op == "unlink":
            existing.valid_to = rewrite.at
            return Rewrite.link(
                ref.src,
                ref.dst,
                family=ref.family,
                weight=float(existing.attrs.get("weight", 1.0)),
                reason=str(existing.attrs.get("reason", "")),
            )

        if rewrite.op == "rewire":
            existing.valid_to = rewrite.at
            self.edges.append(
                StateEdge(
                    src=ref.src,
                    dst=rewrite.to or ref.dst,
                    family=ref.family,
                    attrs=dict(existing.attrs),
                    valid_from=rewrite.at,
                )
            )
            return Rewrite.rewire(
                ref.src, rewrite.to or ref.dst, to=ref.dst, family=ref.family
            )

        before = dict(existing.attrs)
        existing.attrs = (
            dict(rewrite.payload)
            if rewrite.mode == "replace"
            else {**existing.attrs, **rewrite.payload}
        )
        return Rewrite.edge_feature_update(
            ref.src, ref.dst, family=ref.family, mode="replace", **before
        )

    # -- deletion that actually deletes ------------------------------------

    def purge(self, node_id: str) -> set[str]:
        """Remove a node and every trace of it, reporting what depended on it.

        Their V-C in one call. `delete` tombstones, which is right for audit and
        wrong for compliance: a node whose content is still sitting in `attrs`
        has not been deleted in any sense a data subject would recognise. This
        drops the record — and returns the affected scope first, because those
        are the components whose cached derivations still carry its influence
        and have to be revalidated by whoever asked for the deletion.
        """
        downstream = self.affected(node_id)
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if node_id not in (e.src, e.dst)]
        for support in self.supports:
            support.nodes = tuple(n for n in support.nodes if n != node_id)
        return downstream

    # -- convenience -------------------------------------------------------

    def ensure(self, kind: NodeKind, *ids: str, at: int = 0) -> None:
        """Insert nodes that are not there yet. Idempotent.

        How the participants of a run get into the graph without every caller
        writing the same guard.
        """
        for node_id in ids:
            if node_id not in self.nodes:
                self.nodes[node_id] = StateNode(id=node_id, kind=kind, valid_from=at)
