from __future__ import annotations

import pytest
from fakes import FakeEmbedder
from pydantic import TypeAdapter

from ant_ai.core.events import AnyEvent, HealingEvent, TopologyEvent, TopologyLink
from ant_ai.topology.builtins.dig import DigToHeal
from ant_ai.topology.builtins.dytopo import DyTopo
from ant_ai.topology.builtins.shapes import Baseline
from ant_ai.topology.heal import Heal
from ant_ai.topology.materialise import (
    DeliveryMaterialiser,
    VisibilityMaterialiser,
)
from ant_ai.topology.schedule import BufferScheduler, RoundScheduler
from ant_ai.topology.strategy import Pipeline, TopologyStrategy

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def test_a_strategy_declaring_nothing_is_the_framework_baseline() -> None:
    """`build()` is concrete on purpose. A method whose paper changes only
    supervision must not be forced to invent a routing choice it does not make,
    or its class body would read as a claim the paper never made."""

    class OnlyRounds(TopologyStrategy):
        pass

    pipeline = OnlyRounds(max_rounds=4).pipeline()

    assert pipeline.stages == []
    assert isinstance(pipeline.materialiser, VisibilityMaterialiser)
    assert isinstance(pipeline.scheduler, RoundScheduler)
    assert pipeline.max_rounds == 4


def test_a_strategy_overrides_only_what_its_paper_changes() -> None:
    """DIG's delta is supervision and timing, not routing — so it contributes a
    `Heal` stage and a scheduler and leaves the matcher alone."""
    pipeline = DigToHeal().pipeline()

    assert [type(s).__name__ for s in pipeline.stages] == ["Heal"]
    assert isinstance(pipeline.scheduler, BufferScheduler)
    assert isinstance(pipeline.materialiser, DeliveryMaterialiser)


# -- composition ------------------------------------------------------------


def test_composition_concatenates_stages_in_order() -> None:
    pipeline = (DyTopo(embedder=FakeEmbedder({})) | DigToHeal()).pipeline()

    assert [type(s).__name__ for s in pipeline.stages] == ["Semantic", "TopK", "Heal"]


def test_the_right_hand_side_wins_on_shared_components() -> None:
    pipeline = (DyTopo(embedder=FakeEmbedder({})) | DigToHeal()).pipeline()

    assert isinstance(pipeline.scheduler, BufferScheduler)
    assert isinstance(pipeline.materialiser, DeliveryMaterialiser)


def test_composing_does_not_revert_a_setting_to_a_default() -> None:
    """The subtle failure this guards: `Custom(max_rounds=4) | DigToHeal()` must
    not silently run for ten rounds because the right-hand side looks like it
    asked for its own default."""

    class Custom(TopologyStrategy):
        def build(self) -> Pipeline:
            return Pipeline()

    assert (Custom(max_rounds=4) | DigToHeal()).pipeline().max_rounds == 4
    assert (Custom(max_rounds=4) | DigToHeal(max_rounds=7)).pipeline().max_rounds == 7


def test_composition_chains_beyond_two() -> None:
    combined = Baseline() | DyTopo(embedder=FakeEmbedder({})) | DigToHeal()

    assert [type(s).__name__ for s in combined.pipeline().stages] == [
        "Semantic",
        "TopK",
        "Heal",
    ]


def test_composition_keeps_both_halves_in_provenance() -> None:
    """A `Composite` class rather than a folded pipeline exists for this: the
    assembled components alone would not say which papers produced them."""
    provenance = (DyTopo(embedder=FakeEmbedder({}), tau=0.3) | DigToHeal()).provenance()

    assert provenance["strategy"] == "dytopo|dig"
    assert [layer["strategy"] for layer in provenance["layers"]] == ["dytopo", "dig"]
    assert provenance["layers"][0]["tau"] == 0.3


# -- registry ---------------------------------------------------------------


def test_strategies_are_constructible_by_name() -> None:
    strategy = TopologyStrategy.create("dytopo", embedder=FakeEmbedder({}), tau=0.42)

    assert isinstance(strategy, DyTopo)
    assert strategy.tau == 0.42


def test_unknown_names_report_the_known_ones() -> None:
    with pytest.raises(KeyError, match="dytopo"):
        TopologyStrategy.get("nope")


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):

        class Clashing(TopologyStrategy):
            name = "dytopo"


# -- pipeline mechanics -----------------------------------------------------


def test_a_heal_stage_can_be_appended_without_subclassing() -> None:
    """The one-off case: an extra detector on an existing strategy."""
    base = DyTopo(embedder=FakeEmbedder({})).pipeline()
    extended = base | Pipeline(stages=[Heal()])

    assert [type(s).__name__ for s in extended.stages] == ["Semantic", "TopK", "Heal"]
    assert isinstance(extended.materialiser, DeliveryMaterialiser)


def test_topology_event_round_trips_through_the_union() -> None:
    """A missing `AnyEvent` union entry fails here, not at runtime."""
    event = TopologyEvent(
        round=2, links=(TopologyLink(src="a", dst="b", weight=0.6, reason="why"),)
    )

    restored = TypeAdapter(AnyEvent).validate_python(event.model_dump())

    assert isinstance(restored, TopologyEvent)
    assert restored.round == 2
    assert restored.links[0].reason == "why"


def test_healing_event_round_trips_through_the_union() -> None:
    event = HealingEvent(round=1, pattern="ET", interventions=("inject", "reroute"))

    restored = TypeAdapter(AnyEvent).validate_python(event.model_dump())

    assert isinstance(restored, HealingEvent)
    assert restored.pattern == "ET"
    assert restored.interventions == ("inject", "reroute")
