"""Every check here corresponds to a run that used to succeed and mean nothing."""

from __future__ import annotations

import pytest
from fakes import FakeEmbedder

from ant_ai.topology.builtins.dig import DigToHeal
from ant_ai.topology.builtins.dytopo import DyTopo, RandomTopology
from ant_ai.topology.builtins.shapes import Baseline, Static, mesh
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import DeliveryMaterialiser, VisibilityMaterialiser
from ant_ai.topology.preflight import check
from ant_ai.topology.problem import Problem, TopologyConfigurationError
from ant_ai.topology.strategy import Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def codes(problems: list[Problem]) -> list[str]:
    return [p.code for p in problems]


def dytopo() -> Pipeline:
    return DyTopo(embedder=FakeEmbedder({})).pipeline()


# -- E001: components reading fields nothing declares ------------------------


def test_a_matcher_without_structured_turns_is_an_error() -> None:
    """It would score static profile text every round: a fixed topology that
    reports itself as adaptive."""
    problems = check(dytopo(), structured_turns=False, participants=3)

    assert "E001" in codes(problems)
    assert "Semantic" in problems[0].message


def test_a_repair_strategy_without_structured_turns_is_an_error() -> None:
    """Every symptom DIG looks for is declared in a structured turn."""
    problems = check(DigToHeal().pipeline(), structured_turns=False, participants=3)

    assert "E001" in codes(problems)
    assert "Heal" in problems[0].message


def test_the_message_names_the_cause_it_can_be_acted_on() -> None:
    """'Cannot be invoked with a response schema' is true of both causes and
    actionable for neither; which one it is determines the fix."""
    remote = check(dytopo(), structured_turns=False, local=False, participants=3)
    workflow = check(dytopo(), structured_turns=False, local=True, participants=3)

    assert "remote (A2A)" in remote[0].message
    assert "workflow-driven" in workflow[0].message


def test_structured_turns_satisfy_the_check() -> None:
    assert codes(check(dytopo(), structured_turns=True, participants=3)) == []


def test_a_pipeline_reading_nothing_declared_is_unaffected() -> None:
    """`Baseline` and the fixed shapes read no declared field, so a
    workflow-driven run of one is a legitimate configuration."""
    assert (
        codes(check(Baseline().pipeline(), structured_turns=False, participants=3))
        == []
    )


# -- E002: nothing can reach anyone ------------------------------------------


def test_delivery_with_no_links_and_no_addressing_is_an_error() -> None:
    """The case that produced an orphaned-event storm: unaddressed messages are
    routed by links, and there were none, so every message reached nobody."""
    pipeline = Pipeline(materialiser=DeliveryMaterialiser())

    assert "E002" in codes(check(pipeline, structured_turns=False, participants=3))


def test_declared_edges_are_a_route() -> None:
    """`collab()` edges stand as the topology in every round no stage overwrites,
    so a pure repair strategy routes over the colony's own wiring."""
    pipeline = Pipeline(materialiser=DeliveryMaterialiser())

    problems = check(pipeline, structured_turns=False, seeded=True, participants=3)

    assert "E002" not in codes(problems)


def test_a_link_writing_stage_is_a_route() -> None:
    pipeline = Pipeline(
        stages=[Static(links=mesh(["a", "b"]).links)],
        materialiser=DeliveryMaterialiser(),
    )

    assert "E002" not in codes(check(pipeline, structured_turns=False, participants=2))


def test_structured_turns_are_a_route_because_agents_can_address() -> None:
    """A method whose agents name their own correspondents needs no matcher."""
    pipeline = Pipeline(materialiser=DeliveryMaterialiser())

    assert "E002" not in codes(check(pipeline, structured_turns=True, participants=3))


def test_visibility_mode_is_not_subject_to_this_check() -> None:
    """Reachability is the peer tool set there; there is no inbox to fill."""
    pipeline = Pipeline(materialiser=VisibilityMaterialiser())

    assert "E002" not in codes(check(pipeline, structured_turns=False, participants=3))


# -- E003: a topology that cannot bind --------------------------------------


def test_remote_participants_under_visibility_are_an_error() -> None:
    pipeline = Pipeline(stages=[Static()], materialiser=VisibilityMaterialiser())

    assert "E003" in codes(
        check(pipeline, structured_turns=True, local=False, participants=3)
    )


def test_a_colony_with_no_strategy_is_not_an_error() -> None:
    """Nothing decides a topology, so remote agents stay wired as their servers
    wired them — the pre-topology behaviour, not a failure."""
    pipeline = Pipeline(materialiser=VisibilityMaterialiser())

    assert (
        codes(check(pipeline, structured_turns=True, local=False, participants=3)) == []
    )


# -- warnings ----------------------------------------------------------------


def test_a_run_that_cannot_halt_early_warns() -> None:
    problems = check(
        RandomTopology().pipeline(), structured_turns=False, participants=3
    )

    assert codes(problems) == ["W001"]
    assert problems[0].level == "warning"


def test_a_topology_over_one_participant_warns() -> None:
    assert "W002" in codes(check(dytopo(), structured_turns=True, participants=1))


def test_no_strategy_over_one_participant_is_silent() -> None:
    """Nothing was configured to rewire, so there is nothing to point out."""
    assert (
        codes(check(Baseline().pipeline(), structured_turns=True, participants=1)) == []
    )


# -- ordering and reporting --------------------------------------------------


def test_errors_sort_before_warnings() -> None:
    problems = check(dytopo(), structured_turns=False, participants=1)

    assert codes(problems) == ["E001", "W001", "W002"]


def test_the_error_carries_its_problems_for_programmatic_use() -> None:
    """An ablation sweep branches on `code`, not on message text."""
    error = TopologyConfigurationError(
        [Problem(code="E001", level="error", message="m", hint="h")]
    )

    assert [p.code for p in error.problems] == ["E001"]
    assert "m" in str(error) and "h" in str(error)
    assert "Ensemble(...)" in str(error), "the escape hatch has to be discoverable"


def test_a_problem_renders_its_fix() -> None:
    rendered = Problem(code="E001", level="error", message="what", hint="how").render()

    assert rendered == "[E001] what\n  Fix: how"


# -- the marker the checks read ----------------------------------------------


def test_pipelines_report_whether_any_stage_decides_reachability() -> None:
    assert Pipeline(stages=[Static()]).writes_links is True
    assert dytopo().writes_links is True, "TopK decides reachability"
    assert DigToHeal().pipeline().writes_links is False, "repair moves messages only"
    assert Pipeline().writes_links is False


def test_a_stage_that_never_heard_of_the_marker_decides_nothing() -> None:
    """Read with `getattr`, as `needs_structured_turns` is, so a duck-typed
    stage is simply one that writes no links."""

    class Bare:
        async def apply(self, plan, ctx):  # pragma: no cover - never run
            return plan

    assert Pipeline(stages=[Bare()]).writes_links is False


def test_a_link_written_by_a_seed_is_the_same_kind_of_route() -> None:
    """Sanity: the seed is links, not a separate concept the checks must know."""
    assert Link(src="a", dst="b").weight == 1.0


# -- E004: a workflow that cannot run ----------------------------------------


def test_an_unrunnable_workflow_is_an_error() -> None:
    """`Ensemble._act` never raises, so every activation failing looks like a
    quiet run: no messages, and the answer 'Ensemble completed'."""
    problems = check(
        Baseline().pipeline(),
        structured_turns=False,
        participants=2,
        unrunnable_workflows=("architect", "developer"),
    )

    assert codes(problems) == ["E004"]
    assert "architect, developer" in problems[0].message


def test_no_unrunnable_workflows_is_silent() -> None:
    assert (
        codes(check(Baseline().pipeline(), structured_turns=True, participants=2)) == []
    )


# -- W003: a schema that will be coerced rather than constrained -------------


def test_peer_tools_make_a_schema_coerce() -> None:
    """Measured: 1 LLM call for an agent with no tools, 2 with. The second is a
    repair pass that invents `submitted`, `query` and the reactions."""
    pipeline = DyTopo(embedder=FakeEmbedder({})).pipeline()
    pipeline = pipeline.model_copy(update={"materialiser": VisibilityMaterialiser()})

    problems = check(pipeline, structured_turns=True, participants=3)

    assert "W003" in codes(problems)
    assert "peer tools" in problems[0].message


def test_agents_with_their_own_tools_also_coerce() -> None:
    """`Agent` builds a tool step whenever its registry is non-empty, whoever
    filled it — so a real colony's agents hit this under delivery mode too."""
    problems = check(
        dytopo(), structured_turns=True, participants=3, agents_have_tools=True
    )

    assert "W003" in codes(problems)
    assert "tools of their own" in problems[0].message


def test_the_shipped_delivery_path_does_not_coerce() -> None:
    """`DyTopo` and `DigToHeal` both default to `DeliveryMaterialiser`, which
    binds no peer tools — one constrained call per turn."""
    assert "W003" not in codes(check(dytopo(), structured_turns=True, participants=3))
    assert "W003" not in codes(
        check(DigToHeal().pipeline(), structured_turns=True, participants=3)
    )


def test_coercion_is_not_reported_when_nothing_reads_declarations() -> None:
    """No schema is requested, so there is nothing to coerce."""
    pipeline = (
        Baseline()
        .pipeline()
        .model_copy(update={"materialiser": VisibilityMaterialiser()})
    )

    assert "W003" not in codes(check(pipeline, structured_turns=True, participants=3))
