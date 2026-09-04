from __future__ import annotations

import pytest

from ant_ai.topology.builtins.shapes import Baseline, Static, chain, mesh, star
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import VisibilityMaterialiser
from ant_ai.topology.plan import RoundPlan, RunContext

pytestmark = [pytest.mark.unit, pytest.mark.topology]

NAMES = ["a", "b", "c"]


async def _links(stage) -> set[tuple[str, str]]:
    plan = await stage.apply(RoundPlan(round=1), RunContext())
    return {(link.src, link.dst) for link in plan.links}


async def test_chain_informs_the_next_participant_only() -> None:
    assert await _links(chain(NAMES)) == {("a", "b"), ("b", "c")}


async def test_star_connects_every_spoke_to_the_hub_and_nothing_else() -> None:
    assert await _links(star("a", NAMES)) == {
        ("a", "b"),
        ("b", "a"),
        ("a", "c"),
        ("c", "a"),
    }


async def test_mesh_connects_everyone() -> None:
    assert await _links(mesh(NAMES)) == {(a, b) for a in NAMES for b in NAMES if a != b}


async def test_static_replaces_whatever_links_a_plan_already_had() -> None:
    """A shape is a statement about reachability, not an addition to one."""
    stage = Static(links=(Link(src="a", dst="b"),))
    plan = await stage.apply(
        RoundPlan(round=1, links=(Link(src="x", dst="y"),)), RunContext()
    )

    assert [(link.src, link.dst) for link in plan.links] == [("a", "b")]


def test_baseline_is_every_default_and_nothing_else() -> None:
    pipeline = Baseline().pipeline()

    assert pipeline.stages == []
    assert isinstance(pipeline.materialiser, VisibilityMaterialiser)
