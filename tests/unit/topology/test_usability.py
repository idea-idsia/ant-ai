"""The entry surface: naming a strategy, routing over declared edges, reading a run."""

from __future__ import annotations

import pytest
from fakes import FakeEmbedder, FakeParticipant

from ant_ai.topology.builtins.dig import DigToHeal
from ant_ai.topology.builtins.dytopo import DyTopo
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.strategy import EvolutionStrategy, Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


# -- naming a strategy -------------------------------------------------------


def test_a_strategy_is_constructible_by_name() -> None:
    assert EvolutionStrategy.parse("dig").name == "dig"


def test_names_compose_with_a_pipe() -> None:
    """`"dytopo|dig"` is the composed pair, matching what `|` does to objects."""
    strategy = EvolutionStrategy.parse("dytopo|dig")

    assert [type(s).__name__ for s in strategy.pipeline().stages] == [
        "Semantic",
        "TopK",
        "Heal",
    ]


def test_composition_by_name_keeps_provenance_of_both_members() -> None:
    """A folded pipeline would record which components ran but lose which
    published methods produced them."""
    provenance = EvolutionStrategy.parse("dytopo|dig").provenance()

    assert provenance["strategy"] == "dytopo|dig"
    assert [layer["strategy"] for layer in provenance["layers"]] == ["dytopo", "dig"]


def test_hyperparameters_reach_a_named_strategy() -> None:
    assert EvolutionStrategy.parse("dytopo", k_in=7).k_in == 7


def test_hyperparameters_on_a_composed_name_are_refused() -> None:
    """They would silently belong to whichever member happened to accept them."""
    with pytest.raises(ValueError, match="single strategy"):
        EvolutionStrategy.parse("dytopo|dig", k_in=7)


def test_an_unknown_name_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="dytopo"):
        EvolutionStrategy.parse("nope")


def test_naming_a_strategy_does_not_require_importing_it() -> None:
    """A strategy registers itself when its class body runs, so a lookup by name
    used to depend on an import the caller had no reason to make — which is the
    whole thing naming one is meant to avoid. Verified in a subprocess because
    this process has already imported the module."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ant_ai.topology import EvolutionStrategy as T;"
            "print(T.parse('dytopo|dig').provenance()['strategy']);"
            "print(sorted(T.known()))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "dytopo|dig" in result.stdout
    assert "'dig'" in result.stdout and "'baseline'" in result.stdout


def test_an_empty_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="Empty topology strategy"):
        EvolutionStrategy.parse("  ")


# -- the default embedder ----------------------------------------------------


def test_dytopo_resolves_an_embedder_when_none_is_given() -> None:
    """`DyTopo()` is the common case; requiring an encoder object plus knowing
    about an optional extra was the first thing anyone hit."""
    assert DyTopo().embedder is not None


def test_a_given_embedder_is_kept() -> None:
    embedder = FakeEmbedder({})

    assert DyTopo(embedder=embedder).embedder is embedder


def test_provenance_records_the_encoder_that_actually_ran() -> None:
    """Resolved at construction rather than in `build()`, so the run record
    names the encoder rather than the fact that none was chosen."""
    assert DyTopo().provenance()["embedder"] != {}
    assert DyTopo(embedder=FakeEmbedder({})).provenance()["embedder"] == "fake-embedder"


# -- declared edges as the standing topology ---------------------------------


async def test_declared_edges_route_when_no_stage_writes_links() -> None:
    """A pure repair strategy decides no reachability, and used to leave the
    plan empty: unaddressed messages went nowhere and every round reported an
    orphan. The seed now stands wherever nothing overwrites it."""
    participants = {n: FakeParticipant(n) for n in ("a", "b")}
    ensemble = Ensemble(
        participants=participants,
        pipeline=Pipeline(materialiser=DeliveryMaterialiser(), max_rounds=2),
        seed=(Link(src="a", dst="b"),),
    )

    await ensemble.ainvoke("task")

    # The direct statement of the claim: b's second brief carries what a said.
    inbox = participants["b"].briefs[1].inbox
    assert [e.sender for e in inbox] == ["a"]
    assert [(link.src, link.dst) for link in ensemble.graph.links(1)] == [("a", "b")]


async def test_a_stage_that_writes_links_overrides_the_seed() -> None:
    """Otherwise a matcher's decision would be merged with wiring it replaced."""
    from ant_ai.topology.builtins.shapes import Static

    participants = {n: FakeParticipant(n) for n in ("a", "b")}
    ensemble = Ensemble(
        participants=participants,
        pipeline=Pipeline(
            stages=[Static(links=(Link(src="b", dst="a"),))],
            materialiser=DeliveryMaterialiser(),
            max_rounds=2,
        ),
        seed=(Link(src="a", dst="b"),),
    )

    await ensemble.ainvoke("task")

    assert [(x.src, x.dst) for x in ensemble.graph.links(1)] == [("b", "a")]


# -- reading a run -----------------------------------------------------------


async def test_a_report_summarises_what_the_run_did() -> None:
    participants = {n: FakeParticipant(n) for n in ("a", "b")}
    ensemble = Ensemble(
        participants=participants,
        pipeline=Pipeline(materialiser=DeliveryMaterialiser(), max_rounds=2),
        seed=(Link(src="a", dst="b"),),
        provenance={"strategy": "test"},
    )
    await ensemble.ainvoke("task")

    report = ensemble.report()

    assert report.strategy == "test"
    assert report.rounds == 2
    assert report.participants == ["a", "b"]
    assert report.halt_reason == "round budget exhausted"
    assert report.messages > 0


async def test_a_report_flags_a_topology_that_never_moved() -> None:
    """The failure the layer is most likely to hide: a matcher wired the same
    graph every round, and the answer looks no different for it."""
    participants = {n: FakeParticipant(n) for n in ("a", "b")}
    ensemble = Ensemble(
        participants=participants,
        pipeline=Pipeline(materialiser=DeliveryMaterialiser(), max_rounds=3),
        seed=(Link(src="a", dst="b"),),
    )
    await ensemble.ainvoke("task")

    report = ensemble.report()

    assert report.topology_changed is False
    assert "never changed" in report.render()


async def test_a_report_renders_without_a_strategy() -> None:
    """`Baseline` decides nothing, so there is no link count to show and the
    report has to say that rather than print an empty row."""
    ensemble = Ensemble(
        participants={"a": FakeParticipant("a")},
        pipeline=Pipeline(max_rounds=1),
    )
    await ensemble.ainvoke("task")

    assert "none decided" in ensemble.report().render()


def test_halt_reason_is_empty_before_a_run() -> None:
    assert Ensemble(participants={}).report().halt_reason == ""


async def test_findings_are_counted_by_pattern() -> None:
    """Which pathology fired and how often is the first question of a healed
    run, and it used to require walking the stage list by hand."""
    from ant_ai.topology.builtins.shapes import Static

    participants = {n: FakeParticipant(n) for n in ("a", "b")}
    ensemble = Ensemble(
        participants=participants,
        pipeline=DigToHeal().pipeline().model_copy(update={"max_rounds": 3}),
        seed=(Link(src="a", dst="b"), Link(src="b", dst="a")),
    )
    await ensemble.ainvoke("task")

    report = ensemble.report()

    assert set(report.findings) <= {"ET", "MC", "OE", "DL", "ER", "CLA", "RSP"}
    assert all(count > 0 for count in report.findings.values())
    assert isinstance(Static().links, tuple)
