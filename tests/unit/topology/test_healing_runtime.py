"""Healing as it behaves inside the round loop, end to end.

The unit tests for detectors run against hand-built graphs; these check that a
real run produces the graph those detectors need — lineage attributed,
consumption labelled, `e_inf` present — and that a correction changes what
happens next rather than only being reported.
"""

from __future__ import annotations

import pytest
from fakes import FakeParticipant, ScriptedParticipant

from ant_ai.core.events import HealingEvent
from ant_ai.topology.builtins.dig import DigToHeal, dig_detectors
from ant_ai.topology.builtins.shapes import Baseline, Static, mesh
from ant_ai.topology.heal import Heal
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.strategy import Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _people(*names: str, submitter: str | None = None) -> dict:
    return {
        name: FakeParticipant(name, submitted=(name == submitter)) for name in names
    }


def _ensemble(people: dict, *, heal: bool = False, **kwargs) -> Ensemble:
    """Meshed delivery, optionally with DIG's detectors as a `Heal` stage."""
    stages = [mesh(list(people))]
    if heal:
        stages.append(Heal(detectors=dig_detectors()))
    return Ensemble(
        participants=people,
        pipeline=Pipeline(stages=stages, materialiser=DeliveryMaterialiser(), **kwargs),
    )


# -- the graph a run leaves behind ------------------------------------------


async def test_a_run_records_consumption_so_reachable_work_can_empty() -> None:
    """`R(t)` is *unconsumed* messages. An unlabelled delivery edge would leave
    every message outstanding forever and four of the seven detectors on."""
    ensemble = _ensemble(_people("a", "b"), max_rounds=2)

    await ensemble.ainvoke("task")

    assert ensemble.graph.consumed()
    assert all(
        e.action == "consume" for e in ensemble.graph.edges if e.kind == "delivers"
    )


async def test_lineage_is_attributed_from_what_a_turn_consumed() -> None:
    """Without this `Envelope.parents` stays empty and both lineage detectors
    are uncomputable — which is what they were before."""
    ensemble = _ensemble(_people("a", "b"), max_rounds=2)

    await ensemble.ainvoke("task")

    later = [m for m in ensemble.graph.messages.values() if m.round == 1]
    assert later and all(m.parents for m in later)
    assert all(p in ensemble.graph.messages for m in later for p in m.parents)


async def test_submission_is_the_single_source_of_the_terminal_flag() -> None:
    """A `Participant` written by hand sets `submitted` and knows nothing about
    `Envelope.terminal`. Deriving one from the other centrally is what stops a
    custom participant silently disabling every detector."""
    people = _people("a", "b", submitter="a")

    ensemble = _ensemble(people, max_rounds=1)
    await ensemble.ainvoke("task")

    terminal = ensemble.graph.terminal_message()
    assert terminal is not None and terminal.sender == "a"


# -- healing changes what happens next --------------------------------------


async def test_early_termination_healing_keeps_a_premature_run_going() -> None:
    """The headline behaviour: one agent declaring victory while others are
    still working ends an unhealed run at round 0 and does not end a healed one."""
    unhealed = _ensemble(_people("a", "b", "c", submitter="a"), max_rounds=4)
    await unhealed.ainvoke("task")

    healed = _ensemble(_people("a", "b", "c", submitter="a"), max_rounds=4, heal=True)
    await healed.ainvoke("task")

    assert unhealed.graph.rounds() == [0]
    assert healed.graph.rounds() == [0, 1, 2, 3]
    assert "ET" in {f.pattern for f in healed.findings}


async def test_a_unanimous_submission_is_left_alone() -> None:
    """Healing must not block legitimate termination, only premature ones."""
    people = {name: FakeParticipant(name, submitted=True) for name in ("a", "b")}
    ensemble = _ensemble(people, max_rounds=4, heal=True)

    await ensemble.ainvoke("task")

    assert ensemble.graph.rounds() == [0]
    assert ensemble.findings == []


async def test_a_deadlock_is_broken_by_an_emitted_broadcast() -> None:
    """Reachable work with nobody activating. Only observable at all because
    `BufferScheduler` lets an activation set be empty."""
    people = {
        name: ScriptedParticipant(name, [("hello", False)]) for name in ("a", "b")
    }
    strategy = DigToHeal(max_rounds=3)
    ensemble = Ensemble(
        participants=people,
        pipeline=Pipeline(stages=[Static()]) | strategy.pipeline(),
    )

    await ensemble.ainvoke("task")

    assert "DL" in {f.pattern for f in ensemble.findings}
    # The broadcast landed, so the agents that had stalled took another turn.
    assert len(people["a"].briefs) > 1


async def test_missing_completion_fires_when_the_work_runs_out() -> None:
    people = {
        name: ScriptedParticipant(name, [("hi", False), None]) for name in ("a", "b")
    }
    ensemble = Ensemble(
        participants=people,
        pipeline=Pipeline(stages=[mesh(list(people))])
        | DigToHeal(max_rounds=4).pipeline(),
    )

    await ensemble.ainvoke("task")

    assert "MC" in {f.pattern for f in ensemble.findings}


async def test_findings_reach_the_caller_as_events() -> None:
    """Healing that leaves no trace on the stream is indistinguishable from a
    run that never needed it."""
    ensemble = _ensemble(_people("a", "b", "c", submitter="a"), max_rounds=2, heal=True)

    events = [
        event
        async for event in ensemble.stream("task")
        if isinstance(event, HealingEvent)
    ]

    assert events
    assert events[0].pattern == "ET"
    assert events[0].content  # the explanation travels with the correction
    assert "reroute" in events[0].interventions


async def test_no_supervisor_is_the_unhealed_control_and_costs_nothing() -> None:
    ensemble = Ensemble(
        participants=_people("a", "b"), pipeline=Baseline(max_rounds=2).pipeline()
    )

    await ensemble.ainvoke("task")

    assert ensemble.pipeline.stages == []
    assert ensemble.findings == []
