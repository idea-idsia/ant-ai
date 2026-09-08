"""The layer as a whole, once every change is a typed edit.

The unit tests either side of this one check the operators and the graph they
apply to. This one checks the claim that motivates both: that a run records what
was *done*, that a method outside the edge-rewriting branch has somewhere to put
its output, and that reachability is still exactly what it was.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from fakes import FakeParticipant
from pydantic import BaseModel

from ant_ai.topology.builtins.shapes import Static, mesh
from ant_ai.topology.graph import Link
from ant_ai.topology.heal import Heal
from ant_ai.topology.plan import Finding, Intervention, RoundPlan, RunContext
from ant_ai.topology.rewrite import Rewrite, fold_links
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.strategy import Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


class Distil(BaseModel):
    """A node-evolving stage: the shape a memory or skill method takes here."""

    writes_nodes: ClassVar[bool] = True

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        return plan.with_rewrites(
            Rewrite.insert("skill", f"skill-r{ctx.round}", label="distilled").caused_by(
                "distil", at=ctx.round
            ),
            Rewrite.insert(
                "memory", f"note-r{ctx.round}", content="what worked"
            ).caused_by("distil", at=ctx.round),
        )


def _ensemble(names: list[str], **kwargs) -> Ensemble:
    people = {n: FakeParticipant(n) for n in names}
    fields = set(Pipeline.model_fields)
    pipeline = Pipeline(**{k: v for k, v in kwargs.items() if k in fields})
    if "stages" not in kwargs:
        pipeline = pipeline.model_copy(update={"stages": [mesh(names)]})
    rest = {k: v for k, v in kwargs.items() if k not in fields}
    return Ensemble(participants=people, pipeline=pipeline, **rest)


# -- the plan --------------------------------------------------------------


def test_with_links_still_replaces_and_now_says_what_it_changed() -> None:
    plan = RoundPlan(round=1, links=(Link(src="a", dst="b"), Link(src="b", dst="c")))

    updated = plan.with_links((Link(src="a", dst="b"), Link(src="c", dst="a")))

    assert [(x.src, x.dst) for x in updated.links] == [("a", "b"), ("c", "a")]
    assert sorted(r.op for r in updated.rewrites) == ["link", "unlink"]


def test_the_trail_folds_back_to_the_topology_it_produced() -> None:
    """The invariant that lets `links` stay a materialised tuple — its order is
    what `TopologyEvent` carries — without the two drifting apart."""
    plan = RoundPlan(round=1, links=(Link(src="a", dst="b"),))

    updated = plan.with_links((Link(src="b", dst="c", weight=0.5),))

    folded = fold_links(updated.base_links, updated.rewrites)
    assert {(x.src, x.dst, x.weight) for x in folded} == {
        (x.src, x.dst, x.weight) for x in updated.links
    }


def test_an_edge_rewrite_folds_into_reachability_one_wire_at_a_time() -> None:
    """What an incremental method like edge pruning actually does, instead of
    rebuilding the whole adjacency to drop one wire."""
    plan = RoundPlan(round=1, links=(Link(src="a", dst="b"), Link(src="b", dst="c")))

    updated = plan.with_rewrites(Rewrite.unlink("b", "c"))

    assert [(x.src, x.dst) for x in updated.links] == [("a", "b")]


def test_a_provenance_rewrite_does_not_touch_reachability() -> None:
    plan = RoundPlan(round=1, links=(Link(src="a", dst="b"),))

    updated = plan.with_rewrites(Rewrite.link("msg1", "b", family="provenance"))

    assert updated.links == plan.links
    assert len(updated.rewrites) == 1


def test_with_links_stamps_its_diff_with_the_round_it_configures() -> None:
    plan = RoundPlan(round=4)

    updated = plan.with_links((Link(src="a", dst="b"),))

    assert {r.at for r in updated.rewrites} == {4}


# -- repair as rewrites ----------------------------------------------------


class _Inject:
    """A detector that always rewrites the first message it can see."""

    pattern = "I"

    async def detect(self, graph, ctx) -> list[Finding]:
        if not graph.messages:
            return []
        message = next(iter(graph.messages))
        return [
            Finding(
                pattern="I",
                round=ctx.round,
                explanation="always",
                interventions=(
                    Intervention(
                        kind="inject",
                        message=message,
                        content="note",
                        reason="test",
                    ),
                ),
            )
        ]


class _Reroute:
    """A detector that always prescribes the same correction."""

    pattern = "X"

    async def detect(self, graph, ctx) -> list[Finding]:
        message = next(iter(graph.messages))
        return [
            Finding(
                pattern="X",
                round=ctx.round,
                explanation="always",
                interventions=(
                    Intervention(
                        kind="reroute",
                        message=message,
                        recipients=("b",),
                        reason="test",
                    ),
                ),
            )
        ]


async def test_a_repair_carries_the_id_of_the_finding_that_prescribed_it() -> None:
    """Without it the log says a message was rewritten and cannot say which
    detector decided it should be."""
    healer = Heal(detectors=[_Reroute()])
    ensemble = _ensemble(["a", "b"], stages=[mesh(["a", "b"]), healer], max_rounds=2)

    await ensemble.ainvoke("task")

    causes = {f.cause for f in healer.history}
    logged = {
        e.rewrite.cause for e in ensemble.log.entries if e.rewrite.reason == "test"
    }
    assert logged and logged <= causes


async def test_a_repair_does_not_leak_into_reachability() -> None:
    """Repair moves messages; deciding who may reach whom is another stage's job."""
    ensemble = _ensemble(
        ["a", "b"],
        stages=[Static(links=(Link(src="a", dst="b"),)), Heal(detectors=[_Reroute()])],
        max_rounds=2,
    )

    await ensemble.ainvoke("task")

    for rnd in ensemble.graph.rounds():
        assert all(
            (link.src, link.dst) == ("a", "b") for link in ensemble.graph.links(rnd)
        )


# -- the run ---------------------------------------------------------------


async def test_the_seed_is_in_the_graph_before_the_first_diff() -> None:
    ensemble = _ensemble(["a", "b"], seed=(Link(src="a", dst="b"),), max_rounds=1)

    await ensemble.ainvoke("task")

    assert ensemble.log.cascade("seed")[0].rewrite.op == "link"


async def test_a_node_evolving_stage_shows_up_as_a_component_edit() -> None:
    """A run that only ever shows `agent` evolved its wiring and nothing else."""
    ensemble = _ensemble(["a", "b"], stages=[mesh(["a", "b"]), Distil()], max_rounds=2)

    await ensemble.ainvoke("task")
    report = ensemble.report()

    assert report.components == {"skill": 2, "memory": 2}
    assert report.rewrites["insert"] == 4


async def test_a_cascade_touching_two_kinds_is_reported_as_cross_component() -> None:
    ensemble = _ensemble(["a", "b"], stages=[mesh(["a", "b"]), Distil()], max_rounds=2)

    await ensemble.ainvoke("task")

    assert ensemble.report().cross_component == 1
    assert ensemble.log.cross_component() == ("distil",)


async def test_bound_peers_are_recorded_as_read_only_activations() -> None:
    """Binding peers *is* subgraph activation, and writing it down is the
    difference between a topology that was decided and one that can be scored."""
    ensemble = _ensemble(["a", "b"], max_rounds=2)

    await ensemble.ainvoke("task")

    supports = [s for s in ensemble.state.supports if s.kind == "agent"]
    assert supports and all(s.reason == "peers bound for the round" for s in supports)
    assert {s.query for s in supports} == {"a", "b"}


async def test_reachability_recording_can_be_turned_off() -> None:
    ensemble = _ensemble(["a", "b"], max_rounds=2, record_reachability=False)

    await ensemble.ainvoke("task")

    assert ensemble.state.supports == []


async def test_a_run_with_no_evolution_reports_none() -> None:
    """The counters are only interesting because their absence is a real result."""
    ensemble = _ensemble(["a"], stages=[], max_rounds=1, record_reachability=False)

    await ensemble.ainvoke("task")
    report = ensemble.report()

    assert report.rewrites == {} and report.components == {}
    assert "rewrites" not in report.render()


async def test_the_run_can_be_rolled_back_to_where_it_started() -> None:
    ensemble = _ensemble(["a", "b"], stages=[mesh(["a", "b"]), Distil()], max_rounds=3)

    await ensemble.ainvoke("task")
    assert ensemble.state.of_kind("skill")

    ensemble.log.rollback(ensemble.state, to=0)

    assert all(n.valid_to is not None for n in ensemble.state.of_kind("skill"))


async def test_a_rewritten_message_is_a_real_edit_on_the_state_graph() -> None:
    """Messages are mirrored into the state graph as nodes precisely so that a
    repair is an invertible edit rather than a silent no-op that still counts."""
    healer = Heal(detectors=[_Inject()])
    ensemble = _ensemble(["a", "b"], stages=[mesh(["a", "b"]), healer], max_rounds=2)

    await ensemble.ainvoke("task")

    edited = [
        e
        for e in ensemble.log.entries
        if e.rewrite.op == "feature_update" and e.rewrite.reason == "test"
    ]
    assert edited and all(e.inverse is not None for e in edited)

    target = edited[0].rewrite.target.id
    assert "note" in ensemble.state.nodes[target].attrs["content"]

    ensemble.log.rollback(ensemble.state, to=0)
    assert "note" not in ensemble.state.nodes[target].attrs.get("content", "")


async def test_messages_are_mirrored_without_flooding_the_log() -> None:
    """A message an agent produced is not an edit a strategy made."""
    ensemble = _ensemble(["a", "b"], max_rounds=2, record_reachability=False)

    await ensemble.ainvoke("task")

    assert ensemble.state.of_kind("message")
    assert all(e.rewrite.op != "insert" for e in ensemble.log.entries)
