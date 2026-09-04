"""DyTopo: per-round rewiring by semantic matching of self-declared descriptors.

arXiv:2602.06039. Two stages, exactly mirroring the paper's two steps:

    r_ij = q_i . k_j                      -> Semantic  (writes plan.scores)
    A_j->i = 1(r_ij > tau)(1 - delta_ij)  -> TopK      (reads them, writes plan.links)

Keeping scoring and sparsifying as separate stages is what makes the paper's own
random control honest: `RandomScores` paired with the *same* `TopK` holds sparsity
constant by construction rather than by careful reimplementation.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable
from typing import Annotated, ClassVar, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation

from ant_ai.embeddings.protocol import Embedder
from ant_ai.observer import obs
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.plan import RoundPlan, RunContext, ScoreMatrix
from ant_ai.topology.strategy import Pipeline, TopologyStrategy

__all__ = ["DyTopo", "RandomScores", "RandomTopology", "Semantic", "TopK"]


class Semantic(BaseModel):
    """Cosine between each participant's self-declared query and key.

    Both sides of every comparison are text the participants wrote about
    themselves. Nothing here reads the task and assigns roles.

    A participant that produced no descriptors falls back to its profile text, so
    matching degrades to capability-based routing rather than failing. That
    fallback is reported: profile text does not change between rounds, so a run
    where *everyone* falls back scores the same matrix every round and produces a
    topology that looks adaptive and never moves. `needs_structured_turns` is how
    `Colony.ensemble` avoids setting that up in the first place.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    needs_structured_turns: ClassVar[bool] = True
    """This stage matches on `Turn.query`/`Turn.key`, which only a participant
    invoked with a response schema can produce."""

    embedder: Annotated[Embedder, SkipValidation]

    _warned: bool = PrivateAttr(default=False)

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        names = list(ctx.names)
        if not names:
            return plan

        declared = _Descriptors.of(plan, ctx)
        if declared.mute:
            await self._report_fallback(declared.mute, names, ctx)

        q = _l2_normalise(await self.embedder.aembed(declared.queries))
        k = _l2_normalise(await self.embedder.aembed(declared.keys))

        def score(i: int, j: int) -> tuple[float, str]:
            r = _dot(q[i], k[j])
            return r, (
                f"'{_clip(declared.queries[i])}' <- "
                f"'{_clip(declared.keys[j])}' (sim={r:.2f})"
            )

        return plan.model_copy(update={"scores": _score_matrix(names, score)})

    async def _report_fallback(
        self, mute: list[str], names: list[str], ctx: RunContext
    ) -> None:
        """Say that descriptors were missing, and how badly it matters.

        Warned rather than only logged when it is total, because that case is not
        degraded matching but no matching at all: every round embeds the same
        static profile text, so the links never change and an ablation would be
        comparing a method against itself.
        """
        frozen = len(mute) == len(names)
        await obs.event(
            "topology.descriptors.missing",
            round=ctx.round,
            participants=sorted(mute),
            frozen=frozen,
        )
        if frozen and not self._warned:
            self._warned = True
            warnings.warn(
                "No participant declared query/key descriptors, so semantic "
                "matching is scoring static profile text and the topology cannot "
                "change between rounds. Participants must be invoked with a "
                "response schema — from a Colony, that is "
                "`ensemble(use_workflows=False)`.",
                RuntimeWarning,
                stacklevel=2,
            )


class RandomScores(BaseModel):
    """Random relevance, for the paper's random-topology control.

    Paired with `TopK(tau=None, k_in=k)` it produces exactly the same in-degree as
    any other scorer under the same sparsifier.
    """

    seed: int | None = None

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        rng = random.Random(None if self.seed is None else self.seed + ctx.round)
        matrix = _score_matrix(
            list(ctx.names), lambda i, j: (rng.uniform(-1.0, 1.0), "random baseline")
        )
        return plan.model_copy(update={"scores": matrix})


class TopK(BaseModel):
    """Threshold, then cap in-degree. Reads `plan.scores`, writes `plan.links`.

    Ties break deterministically on `(-score, src)`, as the paper requires, so
    ablations reproduce.

    Set `tau=None` to disable thresholding and keep pure top-k. That makes
    in-degree exactly `k_in` for every participant, which is what a sparsity-matched
    control needs.
    """

    tau: float | None = Field(default=0.35, ge=-1.0, le=1.0)
    k_in: int = Field(default=3, ge=1)

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        scores = plan.scores
        if scores is None:
            return plan

        links: list[Link] = []
        for dst in scores.scores:
            row = scores.scores.get(dst, {})
            candidates = [
                (score, src)
                for src, score in row.items()
                if src != dst and (self.tau is None or score > self.tau)
            ]
            candidates.sort(key=lambda pair: (-pair[0], pair[1]))
            for score, src in candidates[: self.k_in]:
                links.append(
                    Link(src=src, dst=dst, weight=score, reason=scores.reason(dst, src))
                )
        return plan.with_links(tuple(links))


class DyTopo(TopologyStrategy):
    """Defaults follow the paper: tau in 0.3-0.4, K_in = 3, T_max = 10."""

    name: ClassVar[str] = "dytopo"
    citation: ClassVar[str] = "arXiv:2602.06039"

    embedder: Annotated[Embedder, SkipValidation]
    tau: float = Field(default=0.35, ge=-1.0, le=1.0)
    k_in: int = Field(default=3, ge=1)

    def build(self) -> Pipeline:
        return Pipeline(
            stages=[
                Semantic(embedder=self.embedder),
                TopK(tau=self.tau, k_in=self.k_in),
            ],
            materialiser=DeliveryMaterialiser(),
        )


class RandomTopology(TopologyStrategy):
    """The paper's sparsity-matched control: random scores, the same sparsifier."""

    name: ClassVar[str] = "random"
    citation: ClassVar[str] = "arXiv:2602.06039 (baseline)"

    k_in: int = Field(default=3, ge=1)
    seed: int | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            stages=[RandomScores(seed=self.seed), TopK(tau=None, k_in=self.k_in)],
            materialiser=DeliveryMaterialiser(),
        )


class _Descriptors(NamedTuple):
    """What each participant said it needs and offers, one entry per name.

    A participant that declared neither falls back to its profile text and is
    listed in `mute`, which is what `Semantic._report_fallback` warns about.
    """

    queries: list[str]
    keys: list[str]
    mute: list[str]

    @classmethod
    def of(cls, plan: RoundPlan, ctx: RunContext) -> _Descriptors:
        declared = cls([], [], [])
        for profile in ctx.participants:
            turn = plan.turns.get(profile.name)
            fallback = profile.as_text()
            query = (turn.query if turn else "") or ""
            key = (turn.key if turn else "") or ""
            if not query and not key:
                declared.mute.append(profile.name)
            declared.queries.append(query or fallback)
            declared.keys.append(key or fallback)
        return declared


def _score_matrix(
    names: list[str], score: Callable[[int, int], tuple[float, str]]
) -> ScoreMatrix:
    """Dense pairwise scores, self-pairs excluded. `score(i, j)` is `r_ij` and why.

    Shared by every scorer so that a control differs from the method it controls
    for in the scoring function alone — the paper's random baseline is honest
    only if the matrix around it is built identically.
    """
    matrix = ScoreMatrix()
    for i, dst in enumerate(names):
        matrix.scores[dst] = {}
        matrix.reasons[dst] = {}
        for j, src in enumerate(names):
            if i == j:
                continue
            matrix.scores[dst][src], matrix.reasons[dst][src] = score(i, j)
    return matrix


def _l2_normalise(vectors: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        out.append([x / norm for x in vec] if norm else [0.0] * len(vec))
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
