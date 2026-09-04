from __future__ import annotations

import warnings

import pytest
from fakes import FakeEmbedder

from ant_ai.topology.builtins.dytopo import (
    DyTopo,
    RandomScores,
    RandomTopology,
    Semantic,
    TopK,
)
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.participant import ParticipantProfile, Turn
from ant_ai.topology.plan import RoundPlan, RunContext, ScoreMatrix

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _plan(scores: dict[str, dict[str, float]] | None = None, **kwargs) -> RoundPlan:
    if scores is not None:
        kwargs["scores"] = ScoreMatrix(names=tuple(scores), scores=scores)
    return RoundPlan(round=1, **kwargs)


def _ctx(*names: str, **kwargs) -> RunContext:
    kwargs.setdefault("participants", tuple(ParticipantProfile(name=n) for n in names))
    return RunContext(**kwargs)


# -- TopK: the sparsifier ---------------------------------------------------


async def test_topk_excludes_self_loops() -> None:
    plan = await TopK(tau=0.0).apply(
        _plan({"a": {"a": 0.99, "b": 0.5}, "b": {"a": 0.5, "b": 0.99}}), _ctx("a", "b")
    )

    assert all(link.src != link.dst for link in plan.links)


async def test_topk_caps_in_degree_and_breaks_ties_deterministically() -> None:
    """Ties break on `(-score, src)` so an ablation reproduces."""
    row = {"b": 0.9, "c": 0.9, "d": 0.5}
    plan = await TopK(tau=0.0, k_in=2).apply(_plan({"a": row}), _ctx("a"))

    assert [link.src for link in plan.links] == ["b", "c"]


async def test_topk_without_a_threshold_gives_every_node_exactly_k_edges() -> None:
    """What makes the paper's random control sparsity-matched by construction
    rather than by careful reimplementation."""
    scores = {
        "a": {"b": -0.9, "c": -0.8},
        "b": {"a": -0.7, "c": -0.6},
        "c": {"a": -0.5, "b": -0.4},
    }
    plan = await TopK(tau=None, k_in=2).apply(_plan(scores), _ctx("a", "b", "c"))

    assert len(plan.links) == 6


async def test_topk_is_a_no_op_without_scores() -> None:
    """A sparsifier with no scoring stage before it must not invent a topology."""
    plan = await TopK().apply(_plan(), _ctx("a", "b"))

    assert plan.links == ()


# -- Semantic: the scorer ---------------------------------------------------


async def test_semantic_matches_declared_need_against_declared_offer() -> None:
    embedder = FakeEmbedder(
        {
            "need the API surface": [1.0, 0.0],
            "module design and API surface": [1.0, 0.0],
            "need nothing": [0.0, 1.0],
            "implementation status": [1.0, 0.0],
        }
    )
    ctx = _ctx("dev", "arch")
    plan = _plan(
        turns={
            "dev": Turn(
                participant="dev",
                query="need the API surface",
                key="implementation status",
            ),
            "arch": Turn(
                participant="arch",
                query="need nothing",
                key="module design and API surface",
            ),
        }
    )

    plan = await Semantic(embedder=embedder).apply(plan, ctx)
    plan = await TopK(tau=0.5).apply(plan, ctx)

    assert [(link.src, link.dst) for link in plan.links] == [("arch", "dev")]
    assert plan.links[0].weight == pytest.approx(1.0)
    assert "sim=1.00" in (plan.links[0].reason or "")


async def test_semantic_falls_back_to_profile_text() -> None:
    """A participant that emitted no descriptors degrades to capability-based
    routing rather than failing."""
    embedder = FakeEmbedder({}, default=[1.0, 0.0])
    ctx = _ctx(
        "a",
        participants=(
            ParticipantProfile(name="a", description="writes code"),
            ParticipantProfile(name="b", description="reviews code"),
        ),
    )

    with pytest.warns(RuntimeWarning):
        plan = await Semantic(embedder=embedder).apply(_plan(), ctx)

    assert "writes code" in embedder.calls[0][0]
    assert plan.scores is not None
    assert plan.scores.scores["a"]["b"] == pytest.approx(1.0)


async def test_scoring_writes_only_scores_and_sparsifying_only_links() -> None:
    """The seam that lets a random control reuse a real sparsifier unchanged:
    each stage owns one field of the plan."""
    ctx = _ctx("a", "b")

    scored = await RandomScores(seed=1).apply(_plan(), ctx)
    assert scored.scores is not None and scored.links == ()

    sparse = await TopK(tau=None, k_in=1).apply(scored, ctx)
    assert sparse.scores is scored.scores and sparse.links


async def test_random_scores_are_reproducible_from_a_seed() -> None:
    ctx = _ctx("a", "b", "c")
    first = await RandomScores(seed=7).apply(_plan(), ctx)
    second = await RandomScores(seed=7).apply(_plan(), ctx)

    assert first.scores == second.scores


# -- the strategy -----------------------------------------------------------


def test_dytopo_assembles_the_papers_two_steps_in_order() -> None:
    pipeline = DyTopo(embedder=FakeEmbedder({}), tau=0.4, k_in=2).pipeline()

    assert [type(s).__name__ for s in pipeline.stages] == ["Semantic", "TopK"]
    assert pipeline.stages[1].tau == 0.4
    assert pipeline.stages[1].k_in == 2
    assert isinstance(pipeline.materialiser, DeliveryMaterialiser)


def test_random_topology_reuses_the_same_sparsifier() -> None:
    """The control differs from DyTopo in the scorer alone."""
    pipeline = RandomTopology(k_in=3).pipeline()

    assert [type(s).__name__ for s in pipeline.stages] == ["RandomScores", "TopK"]
    assert pipeline.stages[1].tau is None
    assert pipeline.stages[1].k_in == 3


@pytest.mark.parametrize("kwargs", [{"tau": 2.0}, {"k_in": 0}, {"max_rounds": 0}])
def test_hyperparameters_are_validated_at_construction(kwargs) -> None:
    """Fields, not function arguments — so a bad sweep value fails immediately."""
    with pytest.raises(ValueError):
        DyTopo(embedder=FakeEmbedder({}), **kwargs)


def test_provenance_records_method_and_hyperparameters() -> None:
    provenance = DyTopo(embedder=FakeEmbedder({}), tau=0.3).provenance()

    assert provenance["strategy"] == "dytopo"
    assert provenance["citation"] == "arXiv:2602.06039"
    assert provenance["tau"] == 0.3
    # A live embedder is not serialisable, so it is recorded by model id.
    assert provenance["embedder"] == "fake-embedder"


# -- descriptors: the difference between adaptive and static ----------------


async def test_a_run_where_nobody_declares_descriptors_is_flagged_as_frozen() -> None:
    """Profile text is the same every round, so a total fallback is not degraded
    matching but no matching at all. It must not pass silently."""
    embedder = FakeEmbedder({})
    ctx = _ctx("a", "b", round=1)

    with pytest.warns(RuntimeWarning, match="cannot change between rounds"):
        await Semantic(embedder=embedder).apply(
            _plan(turns={"a": Turn(participant="a"), "b": Turn(participant="b")}), ctx
        )

    # Fallen back to profile text for both, which is what makes it frozen.
    assert embedder.calls[0] == ["a.", "b."]


async def test_the_frozen_warning_is_raised_once_per_stage() -> None:
    """One line per run, not one per round: the condition cannot change without
    the participants themselves changing."""
    stage = Semantic(embedder=FakeEmbedder({}))
    plan = _plan(turns={"a": Turn(participant="a")})

    with pytest.warns(RuntimeWarning):
        await stage.apply(plan, _ctx("a", round=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await stage.apply(plan, _ctx("a", round=2))


async def test_declared_descriptors_are_used_and_raise_nothing() -> None:
    embedder = FakeEmbedder({})
    turns = {
        "a": Turn(participant="a", query="need tests", key="a parser"),
        "b": Turn(participant="b", query="need a parser", key="tests"),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await Semantic(embedder=embedder).apply(_plan(turns=turns), _ctx("a", "b"))

    assert embedder.calls[0] == ["need tests", "need a parser"]
    assert embedder.calls[1] == ["a parser", "tests"]


async def test_a_partial_fallback_still_matches_and_does_not_warn() -> None:
    """One quiet participant is capability-based routing for that participant,
    not a static run."""
    turns = {"a": Turn(participant="a", query="need tests", key="a parser")}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = await Semantic(embedder=FakeEmbedder({})).apply(
            _plan(turns=turns), _ctx("a", "b")
        )

    assert plan.scores is not None
    assert plan.scores.scores["a"]["b"] is not None


def test_semantic_declares_that_it_needs_structured_turns() -> None:
    """The flag `Colony.ensemble` reads to decide how a turn is taken."""
    assert Semantic.needs_structured_turns is True
    assert DyTopo(embedder=FakeEmbedder({})).pipeline().needs_structured_turns is True
    # The control scores noise, so it is the one matcher that needs nothing from
    # the participants at all.
    assert RandomTopology().pipeline().needs_structured_turns is False
