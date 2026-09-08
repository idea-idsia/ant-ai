from __future__ import annotations

import pytest

from ant_ai.topology.rewrite import Rewrite
from ant_ai.topology.state import SchemaViolation, StateGraph, SupportSubgraph

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _graph() -> StateGraph:
    graph = StateGraph()
    graph.ensure("agent", "arch", "dev")
    return graph


def test_insert_then_delete_round_trips_through_the_inverse() -> None:
    graph = _graph()
    inverse = graph.apply(Rewrite.insert("skill", "refactor", label="Refactor"))

    assert inverse is not None and inverse.op == "delete"
    graph.apply(inverse)
    assert graph.nodes["refactor"].valid_to is not None


def test_a_feature_update_inverts_to_the_attributes_it_replaced() -> None:
    """The inverse comes from the graph and not from the edit, because only the
    graph knew the before-state — which is what makes rollback exact."""
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m", note="first"))

    inverse = graph.apply(Rewrite.feature_update("memory", "m", note="second"))

    assert graph.nodes["m"].attrs["note"] == "second"
    graph.apply(inverse)
    assert graph.nodes["m"].attrs["note"] == "first"


def test_an_update_merges_but_its_inverse_replaces() -> None:
    """Restoring three attributes by merging them back leaves a fourth one the
    edit added still sitting there — a rollback that does not roll back."""
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m", kept="yes"))

    inverse = graph.apply(Rewrite.feature_update("memory", "m", added="new"))

    assert graph.nodes["m"].attrs == {"kept": "yes", "added": "new"}
    assert inverse is not None and inverse.mode == "replace"
    graph.apply(inverse)
    assert graph.nodes["m"].attrs == {"kept": "yes"}


def test_an_edge_update_inverts_the_same_way() -> None:
    graph = _graph()
    graph.apply(Rewrite.link("arch", "dev", weight=0.4))

    inverse = graph.apply(
        Rewrite.edge_feature_update("arch", "dev", weight=0.9, note="raised")
    )

    assert graph.live_edges()[0].attrs["weight"] == 0.9
    graph.apply(inverse)
    assert graph.live_edges()[0].attrs == {"weight": 0.4, "reason": ""}


def test_the_schema_refuses_an_edge_between_the_wrong_kinds() -> None:
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m"))

    with pytest.raises(SchemaViolation, match="E101"):
        graph.apply(Rewrite.link("m", "arch"))


def test_provenance_may_relate_anything_to_anything() -> None:
    """Which is why it is the family every audit question is asked of."""
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m"))
    graph.apply(Rewrite.link("m", "arch", family="provenance"))

    assert len(graph.live_edges()) == 1


def test_deleting_a_node_with_live_edges_is_the_dangling_condition() -> None:
    graph = _graph()
    graph.apply(Rewrite.link("arch", "dev"))

    with pytest.raises(SchemaViolation, match="E102"):
        graph.apply(Rewrite.delete("agent", "arch"))


def test_cascade_closes_the_edges_along_with_the_node() -> None:
    graph = _graph()
    graph.apply(Rewrite.link("arch", "dev"))

    graph.apply(
        Rewrite.delete("agent", "arch", cascade=True).model_copy(update={"at": 3})
    )

    assert graph.nodes["arch"].valid_to == 3
    assert graph.live_edges() == ()


def test_at_hides_what_did_not_exist_yet() -> None:
    """The leakage-free protocol in one method: evidence written later is absent,
    not merely down-ranked."""
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "later").model_copy(update={"at": 5}))

    assert "later" not in graph.at(2).nodes
    assert "later" in graph.at(5).nodes


def test_merge_archives_the_original_and_keeps_provenance() -> None:
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "m1", content="a"))

    graph.apply(Rewrite.merge("memory", "m1", into="m*", label="consolidated"))

    assert graph.nodes["m1"].attrs["archived"] is True
    assert graph.nodes["m*"].label == "consolidated"
    assert [(e.src, e.dst, e.family) for e in graph.live_edges()] == [
        ("m*", "m1", "provenance")
    ]


def test_affected_follows_dependency_and_provenance_but_not_communication() -> None:
    """Who an agent may talk to is decided fresh every round, so counting it
    would make every local edit look like it affects the whole colony."""
    graph = _graph()
    graph.apply(Rewrite.insert("tool", "grep"))
    graph.apply(Rewrite.insert("skill", "search"))
    graph.apply(Rewrite.link("search", "grep", family="dependency"))
    graph.apply(Rewrite.link("arch", "dev"))

    assert graph.affected("grep") == {"search"}
    assert graph.affected("arch") == set()


def test_purge_removes_the_record_and_reports_what_depended_on_it() -> None:
    graph = _graph()
    graph.apply(Rewrite.insert("memory", "secret", content="pii"))
    graph.apply(Rewrite.insert("skill", "derived"))
    graph.apply(Rewrite.link("derived", "secret", family="provenance"))
    graph.supports.append(SupportSubgraph(id="s", nodes=("secret", "other")))

    downstream = graph.purge("secret")

    assert downstream == {"derived"}
    assert "secret" not in graph.nodes
    assert graph.edges == []
    assert graph.supports[0].nodes == ("other",)


def test_dice_rewards_compact_evidence() -> None:
    support = SupportSubgraph(id="s", nodes=("a", "b"))

    assert support.dice({"a", "b"}) == 1.0
    assert support.dice({"a", "b", "c", "d"}) == pytest.approx(2 / 3)
    assert support.dice(set()) == 0.0
