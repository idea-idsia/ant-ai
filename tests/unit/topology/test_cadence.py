from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel

from ant_ai.topology.cadence import Every
from ant_ai.topology.plan import RoundPlan, RunContext
from ant_ai.topology.preflight import check
from ant_ai.topology.strategy import Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


class _Counting(BaseModel):
    needs_structured_turns: ClassVar[bool] = True
    writes_links: ClassVar[bool] = True
    calls: list[int] = []

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        self.calls.append(ctx.round)
        return plan


async def test_a_wrapped_stage_runs_on_its_own_clock() -> None:
    inner = _Counting(calls=[])
    slow = Every(stage=inner, k=3)

    for rnd in range(7):
        await slow.apply(RoundPlan(), RunContext(round=rnd))

    assert inner.calls == [0, 3, 6]


async def test_phase_interleaves_two_stages_on_the_same_period() -> None:
    first, second = _Counting(calls=[]), _Counting(calls=[])

    for rnd in range(4):
        await Every(stage=first, k=2).apply(RoundPlan(), RunContext(round=rnd))
        await Every(stage=second, k=2, phase=1).apply(
            RoundPlan(), RunContext(round=rnd)
        )

    assert first.calls == [0, 2]
    assert second.calls == [1, 3]


def test_declarations_are_forwarded_not_redeclared() -> None:
    """A wrapper answering for itself would quietly disable the check that stops
    a semantic matcher from scoring static profile text."""
    pipeline = Pipeline(stages=[Every(stage=_Counting(calls=[]), k=2)])

    assert pipeline.needs_structured_turns
    assert pipeline.writes_links


def test_a_cadence_that_cannot_fit_the_run_is_reported() -> None:
    pipeline = Pipeline(stages=[Every(stage=_Counting(calls=[]), k=9)], max_rounds=4)

    problems = [
        p
        for p in check(pipeline, structured_turns=True, participants=2, seeded=True)
        if p.code == "W004"
    ]

    assert len(problems) == 1
    assert "_Counting" in problems[0].message
    assert "fires 1 time(s)" in problems[0].message


def test_a_cadence_that_fits_is_not_reported() -> None:
    pipeline = Pipeline(stages=[Every(stage=_Counting(calls=[]), k=2)], max_rounds=6)

    problems = check(pipeline, structured_turns=True, participants=2, seeded=True)

    assert [p.code for p in problems if p.code == "W004"] == []


def test_a_stage_on_a_cadence_is_named_by_the_stage() -> None:
    """Being told that `Every` needs structured turns says nothing about which
    method to change."""
    pipeline = Pipeline(stages=[Every(stage=_Counting(calls=[]), k=1)], max_rounds=4)

    problems = check(pipeline, structured_turns=False, local=False, participants=2)

    assert "_Counting" in next(p for p in problems if p.code == "E001").message
