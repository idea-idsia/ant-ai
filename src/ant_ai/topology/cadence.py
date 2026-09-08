"""Running a stage on a slower clock than the round loop.

`BufferScheduler` already made the argument for one clock being a method choice
rather than a framework constant: under a synchronous barrier a stall is
unreachable by construction, so what counts as an observable failure depends on
what counts as a tick. The same argument has a second half, and this module is
it.

Cross-component co-evolution — the paper's A.4 — is defined as a fast loop
coupled to a slow one: expertise or memory updates every round, topology or
skill structure every few. With a single clock the two collapse, and a method
whose whole claim is that the slow loop is slow becomes a method that rewires on
every round. That is not an approximation of it; it is the ablation it was
compared against.

So: one combinator. A stage wrapped in `Every` runs on its own cadence and is
otherwise indistinguishable from the stage it wraps.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.topology.plan import RoundPlan, RunContext, Stage

__all__ = ["Every"]


class Every(BaseModel):
    """Run *stage* once every `k` rounds, and pass the plan through otherwise.

        Pipeline(stages=[Semantic(...), TopK(...), Every(Distil(...), k=3)])

    The wrapped stage's declarations are forwarded rather than redeclared:
    `needs_structured_turns` and `writes_links` are read off the wrapper by
    preflight and by `Pipeline`, and a wrapper that answered for itself would
    quietly disable the check that stopped a semantic matcher from scoring
    static profile text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage: Annotated[Stage, SkipValidation]
    k: int = Field(default=1, ge=1, description="Rounds between activations.")
    phase: int = Field(
        default=0,
        ge=0,
        description="Which round in each period it runs on. Two slow stages with "
        "the same `k` and different phases interleave instead of contending.",
    )

    @property
    def needs_structured_turns(self) -> bool:
        return bool(getattr(self.stage, "needs_structured_turns", False))

    @property
    def writes_links(self) -> bool:
        return bool(getattr(self.stage, "writes_links", False))

    @property
    def wrapped(self) -> Stage:
        """The stage underneath, so a report names the method and not the wrapper."""
        return self.stage

    def due(self, round: int) -> bool:
        return round % self.k == self.phase % self.k

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        if not self.due(ctx.round):
            return plan
        return await self.stage.apply(plan, ctx)
