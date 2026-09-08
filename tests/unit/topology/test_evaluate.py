from __future__ import annotations

import pytest

from ant_ai.topology.evaluate import locality
from ant_ai.topology.log import RewriteLog
from ant_ai.topology.rewrite import Rewrite
from ant_ai.topology.state import StateGraph

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _coupled() -> StateGraph:
    """A skill that depends on a tool, and an unrelated skill that does not."""
    graph = StateGraph()
    graph.apply(Rewrite.insert("tool", "grep"))
    graph.apply(Rewrite.insert("skill", "search"))
    graph.apply(Rewrite.insert("skill", "unrelated"))
    graph.apply(Rewrite.link("search", "grep", family="dependency"))
    return graph


def test_an_edit_is_local_with_respect_to_what_it_cannot_reach() -> None:
    graph = _coupled()
    log = RewriteLog()
    log.record(Rewrite.feature_update("tool", "grep", version=2), graph)

    assert locality(log, probes=["unrelated"]) == 1.0
    assert locality(log, probes=["search"]) == 0.0
    assert locality(log, probes=["search", "unrelated"]) == 0.5


def test_locality_is_scoped_to_edits_since_a_point() -> None:
    graph = _coupled()
    log = RewriteLog()
    log.record(
        Rewrite.feature_update("tool", "grep", version=2).model_copy(update={"at": 1}),
        graph,
    )

    assert locality(log, since=2, probes=["search"]) == 1.0


def test_with_nothing_watched_there_is_nothing_to_report() -> None:
    assert locality(RewriteLog()) == 1.0
