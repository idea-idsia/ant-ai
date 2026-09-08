from __future__ import annotations

import pytest

from ant_ai.topology.graph import Link
from ant_ai.topology.rewrite import EdgeRef, NodeRef, Rewrite, diff_links, fold_links

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def test_an_operator_and_its_target_have_to_agree() -> None:
    """The validator is what makes the classmethods the only sane way in: an
    operator whose target is the wrong shape is not an edit with a bad field,
    it is not an edit."""
    with pytest.raises(ValueError, match="needs a NodeRef"):
        Rewrite(op="insert", target=EdgeRef(src="a", dst="b"))
    with pytest.raises(ValueError, match="needs an EdgeRef"):
        Rewrite(op="link", target=NodeRef(kind="agent", id="a"))


def test_rewire_without_a_destination_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires `to`"):
        Rewrite(op="rewire", target=EdgeRef(src="a", dst="b"))


def test_only_communication_edges_are_reachability() -> None:
    """A repair that moves one message writes provenance, and folding that into
    reachability would turn it into a standing wire between two agents."""
    assert Rewrite.link("a", "b").touches_topology
    assert not Rewrite.link("m1", "b", family="provenance").touches_topology
    assert not Rewrite.insert("memory", "m").touches_topology


def test_kinds_reports_the_component_a_node_edit_touches() -> None:
    assert Rewrite.insert("skill", "s").kinds == frozenset({"skill"})
    assert Rewrite.link("a", "b").kinds == frozenset()


def test_folding_a_trail_reproduces_the_topology() -> None:
    base = (Link(src="a", dst="b"), Link(src="b", dst="c"))
    target = (Link(src="b", dst="c", weight=0.5), Link(src="c", dst="a"))

    folded = fold_links(base, diff_links(base, target))

    assert {(x.src, x.dst) for x in folded} == {("b", "c"), ("c", "a")}
    assert next(x for x in folded if x.src == "b").weight == 0.5


def test_a_diff_is_a_diff_and_not_a_rebuild() -> None:
    """The difference between dropping one edge and rebuilding an identical
    graph is the whole reason the trail is kept: affected-scope reads the first
    as one neighbourhood and the second as everybody's."""
    base = (Link(src="a", dst="b"), Link(src="b", dst="c"))
    target = (Link(src="a", dst="b"),)

    edits = diff_links(base, target)

    assert [e.op for e in edits] == ["unlink"]
    assert edits[0].target.key == ("b", "c")


def test_an_unchanged_topology_produces_no_edits() -> None:
    base = (Link(src="a", dst="b", weight=0.3, reason="why"),)
    assert diff_links(base, base) == ()


def test_a_changed_weight_is_an_edge_feature_update() -> None:
    base = (Link(src="a", dst="b", weight=0.3),)
    target = (Link(src="a", dst="b", weight=0.9),)

    edits = diff_links(base, target)

    assert [e.op for e in edits] == ["edge_feature_update"]
    assert fold_links(base, edits)[0].weight == 0.9


def test_rewire_moves_an_edge_and_carries_its_attributes() -> None:
    base = (Link(src="a", dst="b", weight=0.7, reason="kept"),)

    moved = fold_links(base, (Rewrite.rewire("a", "b", to="c"),))

    assert [(x.src, x.dst) for x in moved] == [("a", "c")]
    assert moved[0].weight == 0.7 and moved[0].reason == "kept"


def test_caused_by_stamps_a_cascade_without_rebuilding_the_edit() -> None:
    stamped = Rewrite.insert("memory", "m").caused_by("finding-1", at=4)
    assert (stamped.cause, stamped.at) == ("finding-1", 4)


def test_an_activation_is_transient_by_construction() -> None:
    """A read-only operator that could be marked persistent would let a method
    claim it changed nothing while the record said otherwise."""
    assert Rewrite.activate("memory", "sel", nodes=("m1",)).scope == "transient"
