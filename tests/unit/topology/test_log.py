from __future__ import annotations

import pytest

from ant_ai.topology.log import RewriteLog
from ant_ai.topology.rewrite import Rewrite
from ant_ai.topology.state import StateGraph

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _graph() -> StateGraph:
    graph = StateGraph()
    graph.ensure("agent", "arch", "dev")
    return graph


def test_an_entry_carries_the_inverse_and_the_scope() -> None:
    graph = _graph()
    graph.apply(Rewrite.insert("tool", "grep"))
    graph.apply(Rewrite.insert("skill", "search"))
    graph.apply(Rewrite.link("search", "grep", family="dependency"))
    log = RewriteLog()

    entry = log.record(Rewrite.feature_update("tool", "grep", version=2), graph)

    assert entry.applied and entry.inverse is not None
    assert entry.affected == ("search",)


def test_a_rejected_edit_is_kept_rather_than_raised() -> None:
    """A strategy emitting edits the schema forbids is a finding about the
    strategy; killing the run would throw away the trace showing which edit."""
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m"))
    log = RewriteLog()

    entry = log.record(Rewrite.link("m", "arch"), graph)

    assert not entry.applied and "E101" in entry.problem
    assert log.rejected() == (entry,)
    assert log.by_op() == {}


def test_a_cascade_is_the_edits_sharing_one_cause() -> None:
    graph = _graph()
    log = RewriteLog()
    log.extend(
        [
            Rewrite.insert("skill", "s").caused_by("c1"),
            Rewrite.insert("memory", "m").caused_by("c1"),
            Rewrite.insert("tool", "t").caused_by("c2"),
        ],
        graph,
    )

    assert len(log.cascade("c1")) == 2
    assert set(log.cascades()) == {"c1", "c2"}


def test_cross_component_is_a_cascade_touching_two_kinds() -> None:
    """The discriminator for co-evolution, and the only way to tell it from a
    run that did several unrelated things."""
    graph = _graph()
    log = RewriteLog()
    log.extend(
        [
            Rewrite.insert("skill", "s").caused_by("coupled"),
            Rewrite.insert("memory", "m").caused_by("coupled"),
            Rewrite.insert("memory", "m2").caused_by("alone"),
        ],
        graph,
    )

    assert log.cross_component() == ("coupled",)


def test_by_kind_says_whether_anything_beyond_the_wiring_evolved() -> None:
    graph = _graph()
    log = RewriteLog()
    log.extend([Rewrite.link("arch", "dev"), Rewrite.insert("skill", "s")], graph)

    assert log.by_kind() == {"skill": 1}
    assert log.by_op() == {"link": 1, "insert": 1}


def test_rollback_undoes_newest_first() -> None:
    """Applying inverses in order would restore attributes a later edit had
    already replaced."""
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m", note="first"))
    log = RewriteLog()
    log.record(
        Rewrite.feature_update("memory", "m", note="second").model_copy(
            update={"at": 1}
        ),
        graph,
    )
    log.record(
        Rewrite.feature_update("memory", "m", note="third").model_copy(
            update={"at": 2}
        ),
        graph,
    )

    log.rollback(graph, to=1)

    assert graph.nodes["m"].attrs["note"] == "first"


def test_rollback_leaves_the_log_alone() -> None:
    """A rollback that appended to the log would make the next one undo the undo."""
    graph = _graph()
    log = RewriteLog()
    log.record(Rewrite.insert("skill", "s"), graph)

    log.rollback(graph, to=0)

    assert len(log) == 1


def test_rollback_stops_at_the_boundary() -> None:
    graph = _graph()
    log = RewriteLog()
    log.record(Rewrite.insert("skill", "kept").model_copy(update={"at": 1}), graph)
    log.record(Rewrite.insert("skill", "undone").model_copy(update={"at": 2}), graph)

    log.rollback(graph, to=2)

    assert graph.nodes["kept"].valid_to is None
    assert graph.nodes["undone"].valid_to is not None
