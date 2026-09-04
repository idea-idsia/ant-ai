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

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.participant import Envelope, ParticipantProfile, Turn

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
        return self.model_copy(update={"links": links})


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
