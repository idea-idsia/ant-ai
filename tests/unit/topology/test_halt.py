from __future__ import annotations

import pytest
from fakes import ScriptedParticipant

from ant_ai.topology.builtins.shapes import mesh
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.participant import Turn
from ant_ai.topology.plan import RoundPlan, RunContext
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.strategy import Halt, Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _halt(policy: Halt, round: int = 0, **submitted: bool) -> str | None:
    plan = RoundPlan(
        round=round + 1,
        turns={
            name: Turn(participant=name, submitted=flag)
            for name, flag in submitted.items()
        },
    )
    return policy.halt(plan, RunContext(round=round))


def test_the_default_stops_on_the_first_claim() -> None:
    assert _halt(Halt(), a=False, b=True) == "b submitted"
    assert _halt(Halt(), a=False, b=False) is None


def test_unanimous_waits_for_consensus() -> None:
    assert _halt(Halt(unanimous=True), a=True, b=False) is None
    assert _halt(Halt(unanimous=True), a=True, b=True) == "all deciders submitted"


def test_deciders_exclude_everyone_else() -> None:
    """Completion belongs to a manager judging global state, not to whichever
    contributor finishes its own piece first."""
    policy = Halt(deciders={"integrator"})

    assert _halt(policy, solver=True, integrator=False) is None
    assert _halt(policy, solver=True, integrator=True) == "integrator submitted"


def test_never_runs_the_full_budget() -> None:
    assert _halt(Halt.never(), a=True) is None


def test_min_rounds_floors_the_whole_rule() -> None:
    policy = Halt(min_rounds=2)

    assert _halt(policy, 0, a=True) is None
    assert _halt(policy, 1, a=True) is None
    assert _halt(policy, 2, a=True) == "a submitted"


def test_two_constraints_combine_without_nesting() -> None:
    """This was `MinRounds(inner=Designated("x"), rounds=3)` — five classes for
    one predicate, and the combined case needed two of them stacked."""
    policy = Halt(deciders={"reviewer"}, min_rounds=1)

    assert _halt(policy, 0, reviewer=True) is None
    assert _halt(policy, 1, solver=True) is None
    assert _halt(policy, 1, reviewer=True) == "reviewer submitted"


async def test_ensemble_honours_the_halt_policy() -> None:
    """The regression this seam exists for: without it, the first agent to finish
    its own piece ends everyone's run, and two conditions stop being comparable
    because they ran different numbers of rounds."""
    people = {
        "solver": ScriptedParticipant("solver", [("done", True)]),
        "reviewer": ScriptedParticipant("reviewer", [("still checking", False)]),
    }
    ensemble = Ensemble(
        participants=people,
        pipeline=Pipeline(
            stages=[mesh(list(people))],
            materialiser=DeliveryMaterialiser(),
            halt=Halt(deciders={"reviewer"}),
            max_rounds=3,
        ),
    )

    await ensemble.ainvoke("task")

    assert len(people["solver"].briefs) == 3
