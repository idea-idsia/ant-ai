"""The vocabulary every stage speaks.

A topology strategy is an ordered list of transforms over one value: the plan for
the next round. Two published methods motivate that shape rather than one
guess — DyTopo decides reachability *before* a round from self-declared
descriptors, DIG repairs collaboration *after* one from the causal graph. They
take different inputs, but they produce the same output: an edit to who receives
what next. Making that output a single type is what lets them compose instead of
being reconciled by hand in the round loop.
"""

from __future__ import annotations

import uuid
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.participant import Envelope, ParticipantProfile, Turn
from ant_ai.topology.rewrite import Rewrite, diff_links, fold_links
from ant_ai.topology.state import StateGraph

__all__ = [
    "Finding",
    "Intervention",
    "RoundPlan",
    "RunContext",
    "ScoreMatrix",
    "Stage",
]

SUPERVISOR = "supervisor"
"""Sender attributed to messages a stage creates, so injected content is never
mistaken for something a participant said."""


class RunContext(BaseModel):
    """The state of the run right now — what a stage may look at but not change.

    Read-only by convention: everything a stage *writes* goes in the `RoundPlan`
    it returns. Keeping the two apart is what makes a stage replayable against a
    recorded trace.
    """

    round: int = 0
    task: str = ""
    participants: tuple[ParticipantProfile, ...] = ()
    active: frozenset[str] = frozenset()
    """Who took a turn this round. Empty is what makes deadlock observable."""
    graph: InteractionGraph = Field(default_factory=InteractionGraph)
    """The full history, not just this round — so a stage needing decay, momentum
    or accumulated centrality already has it, with no protocol change."""
    state: StateGraph = Field(default_factory=StateGraph)
    """What the system *is*, as opposed to what this run did: the typed graph of
    agents, memories, tools and skills that outlives the run.

    Read-only here for the same reason `graph` is — a stage writes by returning
    rewrites on the plan, never by reaching into the graph — but present, because
    a stage that cannot see the skills an agent holds cannot decide to distil a
    new one, and every method outside the edge-rewriting branch needs to."""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.participants)


class ScoreMatrix(BaseModel):
    """Dense pairwise relevance, before any decision is taken.

    `scores[dst][src]` is how well what *src* offers matches what *dst* needs.
    Carried on the plan rather than passed between two objects, so scoring and
    sparsifying are independent stages and a random control can reuse a real
    sparsifier unchanged.
    """

    scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    """`scores[dst][src]`. The keys are the participants, so there is no separate
    name list to keep in step with them."""
    reasons: dict[str, dict[str, str]] = Field(default_factory=dict)

    def reason(self, dst: str, src: str) -> str | None:
        return self.reasons.get(dst, {}).get(src)


class Intervention(BaseModel):
    """A correction applied to collaboration in flight.

    `message` and `participant` are separate fields on purpose. A single
    overloaded `target` meant the same string was an envelope id at one call site
    and a participant name at another, with nothing to catch the mix-up; here the
    validator does.
    """

    kind: Literal["inject", "reroute", "drop", "emit"]
    message: str | None = Field(
        default=None, description="Envelope id acted on, for inject/reroute/drop."
    )
    participant: str | None = Field(
        default=None, description="Participant the correction concerns."
    )
    content: str | None = Field(
        default=None, description="Text to inject, or the body of an emitted message."
    )
    recipients: tuple[str, ...] = Field(
        default=(), description="Where a reroute or emit should land."
    )
    reason: str = ""

    def model_post_init(self, _context: object) -> None:
        if self.kind in ("inject", "reroute", "drop") and not self.message:
            raise ValueError(f"Intervention '{self.kind}' requires a message id.")
        if self.kind == "emit" and not self.content:
            raise ValueError("Intervention 'emit' requires content.")

    def as_rewrites(self, *, at: int = 0, cause: str = "") -> tuple[Rewrite, ...]:
        """This correction, in the framework's operator vocabulary.

        The four kinds were already node and edge rewrites under other names, and
        saying so is what lets a repair appear in the same log as a rewiring
        rather than in a stream of its own:

        | Intervention | Operator |
        | --- | --- |
        | `inject` | `feature_update` on the message node |
        | `drop` | `delete` on the message node |
        | `reroute` | `rewire`, or `link` when no prior target is known |
        | `emit` | handled by `Heal`, which mints the message id |

        Routing edges here are `provenance`, not `communication`. Communication
        is reserved for agent-to-agent reachability — the thing `plan.links`
        projects — and a repair that moves one message is a trace of that
        message's handling, which is what provenance is for. Folding it into
        reachability would make a single redirected message look like a standing
        wire between two agents.

        `emit` returns nothing because its message does not exist yet; `Heal`
        creates the envelope and records the pair itself.
        """
        if not self.message:
            return ()
        stamp = {"at": at, "cause": cause, "reason": self.reason}
        if self.kind == "inject":
            return (
                Rewrite.feature_update(
                    "message", self.message, content=self.content or ""
                ).model_copy(update=stamp),
            )
        if self.kind == "drop":
            return (
                Rewrite.delete("message", self.message, cascade=True).model_copy(
                    update=stamp
                ),
            )
        if self.kind == "reroute":
            return tuple(
                (
                    Rewrite.rewire(
                        self.message,
                        self.participant,
                        to=recipient,
                        family="provenance",
                    )
                    if self.participant
                    else Rewrite.link(self.message, recipient, family="provenance")
                ).model_copy(update=stamp)
                for recipient in self.recipients
            )
        return ()


class Finding(BaseModel):
    """One detected failure: what, where, why, and what to do about it.

    The explanation travels with the correction rather than being logged beside
    it, because a healed run whose reasons live in a separate stream cannot be
    audited after the fact.
    """

    pattern: str = Field(description="Short code, e.g. 'ET' or 'CLA'.")
    detector: str = ""
    round: int = 0
    explanation: str = ""
    interventions: tuple[Intervention, ...] = ()
    cause: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Identifies the cascade this finding produced. Every rewrite "
        "it prescribes carries this id, which is what makes "
        "`RewriteLog.cascade(cause)` the paper's `C(c)` rather than a filter over "
        "a flat list, and what lets a cascade be recognised as cross-component.",
    )


class RoundPlan(BaseModel):
    """What the next round will look like. Every stage transforms this.

    One value rather than several return types, because the alternative was three
    separate hand-merges in the round loop — one for rewritten messages, one for
    links, one for created messages — and a function whose only reason to exist
    was that two seams emitted the same thing.

    Each field names the kind of stage that owns it. Stages return a **new** plan
    rather than mutating this one; that is what keeps a pipeline from becoming a
    set of stages quietly depending on each other's leftovers.
    """

    round: int = 0
    """The round this configures — the one *after* the turns it was built from."""
    turns: dict[str, Turn] = Field(default_factory=dict)
    """What participants just produced. Owner: the runtime; rewritten by `Heal`."""
    scores: ScoreMatrix | None = None
    """Owner: a scoring stage. Read by a sparsifying stage."""
    links: tuple[Link, ...] = ()
    """Who may reach whom. Owner: a sparsifying or shape stage; `Heal` may add to it."""
    notices: dict[str, tuple[Envelope, ...]] = Field(default_factory=dict)
    """Messages a stage created, by recipient. Owner: `Heal`."""
    findings: tuple[Finding, ...] = ()
    """Structural failures detected this round. Owner: `Heal`."""
    rewrites: tuple[Rewrite, ...] = ()
    """Every typed edit any stage made, in order. Owner: every stage.

    The single write channel, and the one field that is *appended to* rather than
    replaced. `links` is what the next round will be run under; this is what was
    done to arrive at it — plus everything a stage changed that reachability has
    no way to express, which is the whole of the graph outside the communication
    edges.

    The two are kept in step by construction: `with_links` and `with_rewrites`
    are the only writers, and each updates both. `fold_links(base_links,
    rewrites)` reproduces `links` as a set, which is what the invariant test
    checks. `links` stays a materialised tuple rather than becoming a property
    because its *order* is load-bearing — it is what `TopologyEvent` carries to
    consumers — and a fold over a dict would decide that order by accident."""
    base_links: tuple[Link, ...] = ()
    """The standing topology this round's rewrites were applied to."""

    def model_post_init(self, _context: object) -> None:
        # A plan constructed with links and no explicit base started from them:
        # the seeded round-0 topology, or a test stating a starting point. Set
        # here rather than asked of every caller, so `fold_links(base_links,
        # rewrites)` is meaningful on a plan nobody thought about it for.
        if self.links and not self.base_links:
            self.base_links = self.links

    def in_neighbours(self, dst: str) -> list[Link]:
        """Links pointing at *dst*, most relevant first.

        Incoming messages are aggregated in descending relevance, ties broken
        deterministically by source name so an ablation reproduces.
        """
        return sorted(
            (link for link in self.links if link.dst == dst),
            key=lambda link: (-link.weight, link.src),
        )

    def sources_for(self, dst: str) -> tuple[str, ...]:
        return tuple(link.src for link in self.in_neighbours(dst))

    def with_links(self, links: tuple[Link, ...]) -> RoundPlan:
        """Set reachability wholesale, recording the edits that get there.

        The signature and the result are exactly what they were — a sparsifying
        stage still hands back the graph it decided — but the difference from
        what was standing is now written down as `link`/`unlink`/
        `edge_feature_update` rewrites. That diff is what affected-scope analysis
        and rollback are computed from, and recomputing an identical graph from
        scratch every round records nothing, which is the honest answer.
        """
        edits = tuple(
            edit.model_copy(update={"at": self.round})
            for edit in diff_links(self.links, links)
        )
        return self.model_copy(
            update={"links": links, "rewrites": (*self.rewrites, *edits)}
        )

    def with_rewrites(self, *rewrites: Rewrite) -> RoundPlan:
        """Append typed edits, folding any that change reachability into `links`.

        The general channel, and the one a stage outside the edge-rewriting
        branch uses: inserting a memory node, updating a skill's attributes,
        linking a skill to the tool it needs. Rewrites are taken as given, `at`
        included — a repair prescribed in round *n* happened in round *n*, not in
        the round *n+1* this plan configures, and stamping it here would date
        every cascade one round late. `with_links` is the exception and stamps
        its own diff, because reachability takes effect in the round it names.
        Communication edges are folded so
        that a stage may equally decide reachability one edge at a time — which
        is what an incremental method like edge pruning actually does — instead
        of being made to rebuild the whole adjacency to drop one wire.
        """
        if not rewrites:
            return self
        merged = (*self.rewrites, *rewrites)
        links = self.links
        if any(r.touches_topology for r in rewrites):
            links = fold_links(links, rewrites)
        return self.model_copy(update={"links": links, "rewrites": merged})

    def cascade(self, cause: str) -> tuple[Rewrite, ...]:
        """The edits one cause produced — the paper's `C(c)`, before it is logged."""
        return tuple(r for r in self.rewrites if r.cause == cause)


@runtime_checkable
class Stage(Protocol):
    """One transform of the next round's plan.

    The single extension point for a routing or repair algorithm. A stage holds
    no participant handles, performs no I/O on the run and cannot deliver
    anything, so a published method can be replayed against a recorded trace with
    no agents running — and two methods compose by sitting next to each other in
    a list rather than by inheritance.

    A stage that reads a field only a structured turn can carry — `query`, `key`,
    `submitted`, a reaction, an addressed message — sets
    `needs_structured_turns = True` in its class body. Not a member of this
    protocol, which would make every stage without it fail an `isinstance` check;
    `Pipeline.needs_structured_turns` reads it with `getattr` so declaring it
    stays optional.
    """

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan: ...
