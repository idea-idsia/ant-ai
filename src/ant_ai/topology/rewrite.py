"""The one vocabulary every evolution step is written in.

Until this module existed, a stage could change exactly one thing — who may
reach whom — and it changed it by handing back a *new set of links*. That shape
has two costs. A method whose delta is a memory write, a tool acquisition or a
distilled skill has nowhere to put its output, so it cannot be a stage at all;
and because each round hands back a state rather than an edit, the run records
what the topology *was* and never what was *done to it*, which is what rollback,
affected-scope analysis and locality checking all have to be asked of.

So every change any component makes is expressed here as one typed `Rewrite`,
following the operator set in arXiv:2608.18104 (Eq. 3-11): four node operators,
four edge operators, and `activate` — the read-only one, which selects a support
subgraph and changes nothing.

The operators are the paper's; the double-pushout machinery it derives them from
is deliberately not. `L <- K -> R` is a proof that the operator set is closed and
that deletion is well-behaved, not an implementation strategy: computing pushouts
would buy a generality nothing in the layer asks for. What is kept is the part
that pays — the fixed vocabulary, the dangling condition (see
`ant_ai.topology.state`), and the fact that every edit is a value that can be
logged, inverted and replayed.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ant_ai.topology.graph import Link

__all__ = [
    "EDGE_OPS",
    "NODE_OPS",
    "EdgeFamily",
    "EdgeRef",
    "NodeKind",
    "NodeRef",
    "Op",
    "Rewrite",
    "diff_links",
    "fold_links",
]

type NodeKind = Literal[
    "agent", "memory", "tool", "skill", "workflow", "message", "activation"
]
"""What a node in the agent-state graph is.

The first four are the paper's; the last three are this framework's own run
record — a message, an activation, and the workflow a participant runs — which
are nodes in the same graph so that a cascade can cross from an observed message
to the memory it should be written to without changing representation halfway.
"""

type EdgeFamily = Literal["dependency", "communication", "provenance"]
"""The paper's three edge families.

`communication` is what the topology layer has always decided: a `Link`.
`dependency` is what a skill needs from a tool, or a workflow from a skill.
`provenance` is what something was derived from, and is the family every audit
and rollback question is asked of.
"""

type Op = Literal[
    "insert",
    "delete",
    "feature_update",
    "merge",
    "link",
    "unlink",
    "rewire",
    "edge_feature_update",
    "activate",
]

NODE_OPS: frozenset[str] = frozenset({"insert", "delete", "feature_update", "merge"})
EDGE_OPS: frozenset[str] = frozenset(
    {"link", "unlink", "rewire", "edge_feature_update"}
)


class NodeRef(BaseModel):
    """Which node an operator matches."""

    ref: Literal["node"] = "node"
    kind: NodeKind = "agent"
    id: str


class EdgeRef(BaseModel):
    """Which edge an operator matches.

    Endpoints are node ids, and for a `communication` edge those ids are
    participant names — the same convention `Link` has always used, so an edge
    rewrite and a link are two spellings of one fact rather than two facts to
    keep in step.
    """

    ref: Literal["edge"] = "edge"
    family: EdgeFamily = "communication"
    src: str
    dst: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)


class Rewrite(BaseModel):
    """One typed edit, with what caused it and why.

    `cause` is what makes a cascade a cascade rather than a pile of unrelated
    edits: the paper's `C(c) = ((rho_i, mu_i, t_i))` is exactly the rewrites
    sharing one cause id, and cross-component when they touch more than one
    `NodeKind`. `RewriteLog.cascade` reads it back out.

    Construct these through the classmethods rather than the initialiser. They
    are what fixes which fields an operator needs — `rewire` without `to` is not
    a rewrite with a missing field, it is not a rewrite — and the validator below
    is the one that says so.
    """

    op: Op
    target: Annotated[NodeRef | EdgeRef, Field(discriminator="ref")]
    to: str | None = Field(
        default=None,
        description="Rewire's new destination, or merge's consolidated node id.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="The replacement side R: attributes to set, or for "
        "`activate`, the query and the nodes it selected.",
    )
    scope: Literal["persistent", "transient"] = Field(
        default="persistent",
        description="Whether this edit outlives the run. `activate` is the only "
        "operator that is transient by construction; the paper's A.3 branch is "
        "precisely the set of methods whose every edit is.",
    )
    cause: str = Field(
        default="",
        description="What triggered this. Rewrites sharing a cause are one cascade.",
    )
    reason: str = Field(default="", description="Why this edit exists, in words.")
    at: int = Field(default=0, description="Logical time — the round, by default.")
    mode: Literal["merge", "replace"] = Field(
        default="merge",
        description="How `feature_update` and `edge_feature_update` combine "
        "`payload` with the attributes already there. Merging is the paper's "
        "phi(x_v) and the right default — an update naming one attribute should "
        "not erase the rest. Replacing exists because the *inverse* of a merge "
        "cannot itself be a merge: restoring three attributes by merging them "
        "back leaves a fourth one the edit added still sitting there, and a "
        "rollback that does not roll back is worse than none.",
    )

    def model_post_init(self, _context: object) -> None:
        node = isinstance(self.target, NodeRef)
        if self.op in NODE_OPS and not node:
            raise ValueError(f"Rewrite '{self.op}' needs a NodeRef target.")
        if self.op in EDGE_OPS and node:
            raise ValueError(f"Rewrite '{self.op}' needs an EdgeRef target.")
        if self.op in ("rewire", "merge") and not self.to:
            raise ValueError(f"Rewrite '{self.op}' requires `to`.")

    # -- what it touches ---------------------------------------------------

    @property
    def kinds(self) -> frozenset[str]:
        """The `NodeKind`s this edit concerns, for cross-component accounting."""
        if isinstance(self.target, NodeRef):
            return frozenset({self.target.kind})
        return frozenset()

    @property
    def is_edge(self) -> bool:
        return isinstance(self.target, EdgeRef)

    @property
    def touches_topology(self) -> bool:
        """Whether this edit changes who may reach whom."""
        return (
            isinstance(self.target, EdgeRef) and self.target.family == "communication"
        )

    # -- the operators (Eq. 3-11) -----------------------------------------

    @classmethod
    def insert(cls, kind: NodeKind, id: str, /, **attrs: Any) -> Rewrite:  # noqa: A002
        """Eq. 3 — a new typed node. `L = K = {}`."""
        return cls(op="insert", target=NodeRef(kind=kind, id=id), payload=attrs)

    @classmethod
    def delete(cls, kind: NodeKind, id: str, /, **meta: Any) -> Rewrite:  # noqa: A002
        """Eq. 4 — remove a node. `K = R = {}`, subject to the dangling condition."""
        return cls(op="delete", target=NodeRef(kind=kind, id=id), payload=meta)

    @classmethod
    def feature_update(
        cls,
        kind: NodeKind,
        id: str,  # noqa: A002
        /,
        *,
        mode: Literal["merge", "replace"] = "merge",
        **attrs: Any,
    ) -> Rewrite:
        """Eq. 5 — revise attributes, keeping identity. `K = {v}`."""
        return cls(
            op="feature_update",
            target=NodeRef(kind=kind, id=id),
            payload=attrs,
            mode=mode,
        )

    @classmethod
    def merge(cls, kind: NodeKind, id: str, /, *, into: str, **attrs: Any) -> Rewrite:  # noqa: A002
        """Eq. 6 — consolidate into a new node, archiving the original.

        The paper's `Merge` keeps both originals and adds provenance edges from
        the consolidated node to each. `ant_ai.topology.state` does the same:
        this is not a delete, and reading it as one is how a memory system loses
        the evidence it consolidated from.
        """
        return cls(op="merge", target=NodeRef(kind=kind, id=id), to=into, payload=attrs)

    @classmethod
    def link(
        cls,
        src: str,
        dst: str,
        /,
        *,
        family: EdgeFamily = "communication",
        weight: float = 1.0,
        reason: str = "",
        **attrs: Any,
    ) -> Rewrite:
        """Eq. 7 — a new typed edge between two existing nodes."""
        return cls(
            op="link",
            target=EdgeRef(family=family, src=src, dst=dst),
            payload={"weight": weight, "reason": reason, **attrs},
            reason=reason,
        )

    @classmethod
    def unlink(
        cls,
        src: str,
        dst: str,
        /,
        *,
        family: EdgeFamily = "communication",
        reason: str = "",
    ) -> Rewrite:
        """Eq. 8 — remove an edge, keeping both endpoints."""
        return cls(
            op="unlink",
            target=EdgeRef(family=family, src=src, dst=dst),
            reason=reason,
        )

    @classmethod
    def rewire(
        cls,
        src: str,
        dst: str,
        /,
        *,
        to: str,
        family: EdgeFamily = "communication",
        reason: str = "",
    ) -> Rewrite:
        """Eq. 9 — redirect an edge to a new target, carrying its attributes over."""
        return cls(
            op="rewire",
            target=EdgeRef(family=family, src=src, dst=dst),
            to=to,
            reason=reason,
        )

    @classmethod
    def edge_feature_update(
        cls,
        src: str,
        dst: str,
        /,
        *,
        family: EdgeFamily = "communication",
        mode: Literal["merge", "replace"] = "merge",
        **attrs: Any,
    ) -> Rewrite:
        """Eq. 10 — revise an edge's attributes."""
        return cls(
            op="edge_feature_update",
            target=EdgeRef(family=family, src=src, dst=dst),
            payload=attrs,
            mode=mode,
            reason=str(attrs.get("reason", "")),
        )

    @classmethod
    def activate(
        cls,
        kind: NodeKind,
        id: str,
        /,
        *,
        query: str = "",
        nodes: tuple[str, ...] = (),
        reason: str = "",
    ) -> Rewrite:  # noqa: A002
        """Eq. 11 — select a support subgraph. Read-only, and always transient.

        `id` names the selection, not a node being changed: nothing is changed.
        Recorded anyway, because an activation nobody wrote down is one no
        support-subgraph metric can be computed against — which is most of what
        the paper's evaluation section is about.
        """
        return cls(
            op="activate",
            target=NodeRef(kind=kind, id=id),
            payload={"query": query, "nodes": list(nodes)},
            scope="transient",
            reason=reason,
        )

    def caused_by(self, cause: str, *, at: int | None = None) -> Rewrite:
        """This edit, attributed. What a detector stamps its prescriptions with."""
        update: dict[str, Any] = {"cause": cause}
        if at is not None:
            update["at"] = at
        return self.model_copy(update=update)


# -- the communication projection ------------------------------------------


def fold_links(
    base: tuple[Link, ...], rewrites: tuple[Rewrite, ...]
) -> tuple[Link, ...]:
    """Apply a trail of edge rewrites to a standing topology.

    The inverse of `diff_links`, and the reason `RoundPlan` can keep both a
    materialised `links` tuple and the trail that produced it without the two
    drifting: a test folds the trail and compares.
    """
    current: dict[tuple[str, str], Link] = {(x.src, x.dst): x for x in base}
    for rewrite in rewrites:
        edge = rewrite.target
        if not isinstance(edge, EdgeRef) or edge.family != "communication":
            continue
        if rewrite.op == "link":
            current[edge.key] = Link(
                src=edge.src,
                dst=edge.dst,
                weight=float(rewrite.payload.get("weight", 1.0)),
                reason=rewrite.payload.get("reason") or None,
            )
        elif rewrite.op == "unlink":
            current.pop(edge.key, None)
        elif rewrite.op == "rewire":
            moved = current.pop(edge.key, None)
            if moved is not None and rewrite.to:
                current[(moved.src, rewrite.to)] = moved.model_copy(
                    update={"dst": rewrite.to}
                )
        elif rewrite.op == "edge_feature_update":
            existing = current.get(edge.key)
            if existing is not None:
                update = {
                    k: v
                    for k, v in rewrite.payload.items()
                    if k in ("weight", "reason")
                }
                current[edge.key] = existing.model_copy(update=update)
    return tuple(current.values())


def diff_links(
    current: tuple[Link, ...], target: tuple[Link, ...]
) -> tuple[Rewrite, ...]:
    """The edits that turn one topology into another.

    A diff rather than "unlink everything, then link the new set", because the
    difference is the whole point of keeping a trail: a round that dropped one
    edge and a round that rebuilt an identical graph from scratch are not the
    same event, and affected-scope analysis reads the former as one node's
    neighbourhood and the latter as everybody's.
    """
    before = {(x.src, x.dst): x for x in current}
    after = {(x.src, x.dst): x for x in target}
    edits: list[Rewrite] = []
    for key in before.keys() - after.keys():
        edits.append(Rewrite.unlink(*key, reason="dropped by the round's topology"))
    for key in sorted(after.keys() - before.keys()):
        link = after[key]
        edits.append(
            Rewrite.link(
                link.src, link.dst, weight=link.weight, reason=link.reason or ""
            )
        )
    for key in sorted(before.keys() & after.keys()):
        old, new = before[key], after[key]
        if old.weight != new.weight or old.reason != new.reason:
            edits.append(
                Rewrite.edge_feature_update(
                    *key, weight=new.weight, reason=new.reason or ""
                )
            )
    return tuple(edits)
