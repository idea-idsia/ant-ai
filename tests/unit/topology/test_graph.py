from __future__ import annotations

import pytest

from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.participant import Envelope

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def test_records_bipartite_structure() -> None:
    graph = InteractionGraph()
    act = graph.record_activation("architect", round=0)
    envelope = Envelope(sender="architect", content="API design", round=0)
    graph.record_message(envelope, activation_id=act)

    receiver = graph.record_activation("developer", round=1)
    graph.record_delivery(
        envelope.id, receiver, round=1, weight=0.62, reason="needs API"
    )

    kinds = [e.kind for e in graph.edges]
    assert kinds == ["generates", "delivers"]
    assert graph.edges[1].reason == "needs API"
    assert graph.activations[act].participant == "architect"


def test_end_activation_records_error() -> None:
    graph = InteractionGraph()
    act = graph.record_activation("dev", round=0)
    graph.end_activation(act, error="boom")

    assert graph.activations[act].error == "boom"
    assert graph.activations[act].ended_at is not None


def test_links_projects_view_per_round() -> None:
    graph = InteractionGraph()
    graph.record_links((Link(src="a", dst="b", weight=0.6, reason="r"),), round=0)
    graph.record_links((Link(src="b", dst="c"),), round=1)

    assert [(link.src, link.dst) for link in graph.links(0)] == [("a", "b")]
    assert [(link.src, link.dst) for link in graph.links(1)] == [("b", "c")]
    assert graph.links(0)[0].reason == "r"
    assert graph.in_neighbours("b", 0) == ["a"]


def test_unused_visibility_finds_granted_but_uncalled() -> None:
    """Reachability the matcher granted that the agent never exercised."""
    graph = InteractionGraph()
    graph.record_links(
        (Link(src="reviewer", dst="dev"), Link(src="architect", dst="dev")),
        round=0,
    )
    graph.record_invocation("dev", "reviewer", round=0)

    unused = graph.unused_visibility(0)
    assert [(link.src, link.dst) for link in unused] == [("architect", "dev")]


def test_snapshot_truncates_to_round() -> None:
    graph = InteractionGraph()
    graph.record_activation("a", round=0)
    graph.record_activation("a", round=5)
    graph.record_links((Link(src="a", dst="b"),), round=5)

    snap = graph.snapshot(0)
    assert len(snap.activations) == 1
    assert snap.edges == []


def test_round_trips_through_json() -> None:
    """The property the deferred detector suite depends on: develop and test
    detectors against recorded traces, with no agents running."""
    graph = InteractionGraph()
    act = graph.record_activation("a", round=0)
    graph.record_message(Envelope(sender="a", content="hi", round=0), activation_id=act)
    graph.record_links((Link(src="a", dst="b", weight=0.5),), round=0)

    restored = InteractionGraph.model_validate_json(graph.model_dump_json())

    assert restored.edges == graph.edges
    assert restored.messages == graph.messages
    assert list(restored.activations) == list(graph.activations)


def test_mermaid_renders_each_round() -> None:
    graph = InteractionGraph()
    graph.record_activation("a", round=0)
    graph.record_links((Link(src="a", dst="b", weight=0.62),), round=0)

    out = graph.to_mermaid()
    assert out.startswith("flowchart LR")
    assert "0.62" in out
