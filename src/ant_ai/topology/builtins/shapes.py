"""Fixed topologies: the shapes a comparison uses as controls.

Not published methods — one-liners over `Static` — but they live here rather than
in the core because the core ships no concrete routing at all. What reachability
should be is always somebody's choice, never a framework default.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from ant_ai.topology.graph import Link
from ant_ai.topology.plan import RoundPlan, RunContext
from ant_ai.topology.strategy import TopologyStrategy

__all__ = ["Baseline", "Static", "chain", "mesh", "star"]


class Static(BaseModel):
    """A fixed reachability graph, reused every round.

    The compatibility anchor: built from a colony's declared `collab()` edges, it
    reproduces exactly what a colony wires today.
    """

    links: tuple[Link, ...] = ()

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        return plan.with_links(self.links)


def chain(names: list[str]) -> Static:
    """`a -> b -> c`: each participant informs the next."""
    return Static(
        links=tuple(
            Link(src=a, dst=b, reason="chain")
            for a, b in zip(names, names[1:], strict=False)
        )
    )


def star(hub: str, spokes: list[str]) -> Static:
    """Every spoke exchanges with a central hub, and only with the hub."""
    links: list[Link] = []
    for spoke in spokes:
        if spoke == hub:
            continue
        links.append(Link(src=hub, dst=spoke, reason="star"))
        links.append(Link(src=spoke, dst=hub, reason="star"))
    return Static(links=tuple(links))


def mesh(names: list[str]) -> Static:
    """Everyone reaches everyone — also the usual debate baseline.

    There was a separate `debate()` that returned the identical edge set under a
    different `reason` string; a second name for the same graph is not a second
    baseline.
    """
    return Static(
        links=tuple(
            Link(src=a, dst=b, reason="mesh") for a in names for b in names if a != b
        )
    )


class Baseline(TopologyStrategy):
    """The framework's own behaviour, named so it can be a row in a table.

    Every default, nothing overridden. Exists because "no strategy" has to be
    something you can select by name alongside the others, not a special case the
    harness has to know about.
    """

    name: ClassVar[str] = "baseline"
