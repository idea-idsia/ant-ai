"""Detectors as pure queries over a recorded graph.

Every test here builds an `InteractionGraph` by hand and runs a detector
against it, with no participants, no ensemble and no event loop doing anything
interesting. That is the property the seam was designed for: a detection
algorithm can be developed and falsified against a trace.
"""

from __future__ import annotations

import pytest

from ant_ai.topology.builtins.dig import (
    CrossLineageAggregation,
    Deadlock,
    EarlyTermination,
    ExcessiveRerouting,
    MissingCompletion,
    OrphanedEvent,
    RepeatedSubproblem,
    dig_detectors,
)
from ant_ai.topology.graph import InteractionGraph
from ant_ai.topology.heal import Detector
from ant_ai.topology.participant import Envelope, ParticipantProfile
from ant_ai.topology.plan import SUPERVISOR, Intervention, RunContext

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _say(
    graph: InteractionGraph,
    sender: str,
    *,
    round: int = 0,
    terminal: bool = False,
    parents: tuple[str, ...] = (),
    consumed_by: str | None = None,
) -> Envelope:
    """One participant generating one message, optionally consumed by another."""
    activation = graph.record_activation(sender, round=round)
    envelope = Envelope(
        sender=sender,
        content=f"{sender}@{round}",
        round=round,
        terminal=terminal,
        parents=parents,
    )
    graph.record_message(envelope, activation_id=activation)
    if consumed_by is not None:
        reader = graph.record_activation(consumed_by, round=round + 1)
        graph.record_delivery(envelope.id, reader, round=round + 1, action="consume")
    return envelope


def _ctx(**kwargs) -> RunContext:
    kwargs.setdefault(
        "participants", tuple(ParticipantProfile(name=n) for n in ("a", "b", "c"))
    )
    kwargs.setdefault("active", frozenset({"a", "b", "c"}))
    return RunContext(**kwargs)


def test_every_dig_detector_satisfies_the_protocol() -> None:
    """A structural check, so a detector that drifts out of the seam fails here
    rather than at the first run that needed it."""
    detectors = dig_detectors()
    assert len(detectors) == 7
    assert all(isinstance(d, Detector) for d in detectors)
    assert {d.pattern for d in detectors} == {
        "ET",
        "MC",
        "OE",
        "DL",
        "ER",
        "CLA",
        "RSP",
    }


# -- ET ---------------------------------------------------------------------


async def test_early_termination_fires_on_a_submit_with_work_outstanding() -> None:
    graph = InteractionGraph()
    _say(graph, "a", terminal=True)
    _say(graph, "b")

    findings = await EarlyTermination().detect(graph, _ctx())

    assert [f.pattern for f in findings] == ["ET"]
    assert {i.kind for i in findings[0].interventions} == {"inject", "reroute"}
    assert "b" in findings[0].explanation


async def test_early_termination_ignores_work_from_agents_that_also_finished() -> None:
    """Otherwise the detector fires whenever anyone else speaks in the final
    round — which is every real run — and healing would block every legitimate
    termination rather than only the premature ones."""
    graph = InteractionGraph()
    _say(graph, "a", terminal=True)
    _say(graph, "b", terminal=True)

    assert await EarlyTermination().detect(graph, _ctx()) == []


async def test_early_termination_is_silent_without_a_submit() -> None:
    graph = InteractionGraph()
    _say(graph, "a")

    assert await EarlyTermination().detect(graph, _ctx()) == []


# -- MC ---------------------------------------------------------------------


async def test_missing_completion_fires_once_work_is_exhausted() -> None:
    graph = InteractionGraph()
    _say(graph, "a", consumed_by="b")

    findings = await MissingCompletion().detect(graph, _ctx(round=1))

    assert [f.pattern for f in findings] == ["MC"]
    assert findings[0].interventions[0].kind == "emit"
    assert findings[0].interventions[0].recipients == ("a", "b", "c")


async def test_missing_completion_respects_its_window() -> None:
    graph = InteractionGraph()
    _say(graph, "a", consumed_by="b")

    assert await MissingCompletion(window=3).detect(graph, _ctx(round=1)) == []


async def test_missing_completion_is_silent_while_work_remains() -> None:
    graph = InteractionGraph()
    _say(graph, "a")

    assert await MissingCompletion().detect(graph, _ctx(round=1)) == []


# -- OE ---------------------------------------------------------------------


async def test_orphaned_event_fires_on_a_message_nobody_was_routed() -> None:
    graph = InteractionGraph()
    _say(graph, "a")

    findings = await OrphanedEvent().detect(graph, _ctx(round=1))

    assert [f.pattern for f in findings] == ["OE"]
    assert findings[0].interventions[1].recipients == ("a",)


async def test_a_contribution_heard_through_one_copy_is_not_orphaned() -> None:
    """A turn emits its contribution publicly and privately, and a delivering
    materialiser routes one of them. Two visibilities of one event must settle
    together, or every public message in delivery mode looks orphaned."""
    graph = InteractionGraph()
    activation = graph.record_activation("a", round=0)
    public = Envelope(sender="a", content="public", visibility="public")
    private = Envelope(sender="a", content="private", visibility="private")
    graph.record_message(public, activation_id=activation)
    graph.record_message(private, activation_id=activation)
    reader = graph.record_activation("b", round=1)
    graph.record_delivery(private.id, reader, round=1, action="consume")

    assert await OrphanedEvent().detect(graph, _ctx(round=1)) == []


# -- DL ---------------------------------------------------------------------


async def test_deadlock_fires_when_nobody_activates_with_work_pending() -> None:
    graph = InteractionGraph()
    _say(graph, "a")

    findings = await Deadlock().detect(graph, _ctx(round=1, active=frozenset()))

    assert [f.pattern for f in findings] == ["DL"]
    assert findings[0].interventions[0].kind == "emit"


async def test_deadlock_is_silent_while_anyone_is_working() -> None:
    graph = InteractionGraph()
    _say(graph, "a")

    assert await Deadlock().detect(graph, _ctx(round=1)) == []


# -- ER ---------------------------------------------------------------------


async def test_excessive_rerouting_counts_the_graphs_own_interventions() -> None:
    """A supervisor that is thrashing detects itself, because every rewrite is
    recorded as an edge rather than only logged."""
    graph = InteractionGraph()
    envelope = _say(graph, "a")
    for target in ("b", "c", "b"):
        graph.record_intervention(
            envelope.id, target, action="reroute", round=1, reason="test"
        )

    findings = await ExcessiveRerouting(threshold=2).detect(graph, _ctx(round=1))

    assert [f.pattern for f in findings] == ["ER"]
    assert "3 times" in findings[0].explanation


async def test_excessive_rerouting_ignores_a_message_that_landed() -> None:
    graph = InteractionGraph()
    envelope = _say(graph, "a", consumed_by="b")
    for target in ("b", "c", "b"):
        graph.record_intervention(
            envelope.id, target, action="reroute", round=1, reason="test"
        )

    assert await ExcessiveRerouting(threshold=2).detect(graph, _ctx(round=1)) == []


# -- CLA --------------------------------------------------------------------


async def test_cross_lineage_aggregation_fires_on_disjoint_ancestry() -> None:
    graph = InteractionGraph()
    left = _say(graph, "a")
    right = _say(graph, "b")
    child_l = _say(graph, "a", round=1, parents=(left.id,))
    child_r = _say(graph, "b", round=1, parents=(right.id,))

    reader = graph.record_activation("c", round=2)
    for message in (child_l, child_r):
        graph.record_delivery(message.id, reader, round=2, action="consume")

    findings = await CrossLineageAggregation().detect(graph, _ctx(round=2))

    assert [f.pattern for f in findings] == ["CLA"]
    assert {i.kind for i in findings[0].interventions} == {"inject"}


async def test_shared_ancestry_is_not_cross_lineage() -> None:
    graph = InteractionGraph()
    root = _say(graph, "a")
    child_a = _say(graph, "a", round=1, parents=(root.id,))
    child_b = _say(graph, "b", round=1, parents=(root.id,))

    reader = graph.record_activation("c", round=2)
    for message in (child_a, child_b):
        graph.record_delivery(message.id, reader, round=2, action="consume")

    assert await CrossLineageAggregation().detect(graph, _ctx(round=2)) == []


async def test_two_messages_with_no_ancestry_are_not_cross_lineage() -> None:
    """At the start of a run every message is a root. Comparing two roots would
    report the entire first exchange as cross-lineage: absence of lineage is not
    evidence of different lineage."""
    graph = InteractionGraph()
    first = _say(graph, "a")
    second = _say(graph, "b")

    reader = graph.record_activation("c", round=1)
    for message in (first, second):
        graph.record_delivery(message.id, reader, round=1, action="consume")

    assert await CrossLineageAggregation().detect(graph, _ctx(round=1)) == []


# -- RSP --------------------------------------------------------------------


async def test_repeated_subproblem_fires_when_two_reducers_share_an_input() -> None:
    graph = InteractionGraph()
    upstream = _say(graph, "a")

    for name in ("b", "c"):
        activation = graph.record_activation(name, round=1)
        graph.record_delivery(upstream.id, activation, round=1, action="consume")
        graph.record_message(
            Envelope(sender=name, content="solved", round=1), activation_id=activation
        )

    findings = await RepeatedSubproblem().detect(graph, _ctx(round=1))

    assert [f.pattern for f in findings] == ["RSP"]
    assert {i.recipients[0] for i in findings[0].interventions} == {"b", "c"}


async def test_problem_generating_activations_are_not_duplicated_work() -> None:
    """Two agents *expanding* on the same input is collaboration; two agents
    *solving* it is waste. `|O_v| <= |I_v|` is what separates them."""
    graph = InteractionGraph()
    upstream = _say(graph, "a")

    for name in ("b", "c"):
        activation = graph.record_activation(name, round=1)
        graph.record_delivery(upstream.id, activation, round=1, action="consume")
        for part in range(2):
            graph.record_message(
                Envelope(sender=name, content=f"subtask {part}", round=1),
                activation_id=activation,
            )

    assert await RepeatedSubproblem().detect(graph, _ctx(round=1)) == []


async def test_a_supervisor_broadcast_is_not_duplicated_work() -> None:
    """A broadcast is addressed to everyone by design. Counting it would make
    healing manufacture the next round's finding."""
    graph = InteractionGraph()
    notice = Envelope(sender=SUPERVISOR, content="status", round=0)
    graph.record_emission(notice)

    for name in ("b", "c"):
        activation = graph.record_activation(name, round=1)
        graph.record_delivery(notice.id, activation, round=1, action="consume")
        graph.record_message(
            Envelope(sender=name, content="ack", round=1), activation_id=activation
        )

    assert await RepeatedSubproblem().detect(graph, _ctx(round=1)) == []


# -- intervention validation ------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "inject", "content": "x"},
        {"kind": "reroute", "recipients": ("a",)},
        {"kind": "drop"},
        {"kind": "emit"},
    ],
)
def test_an_intervention_missing_its_target_is_rejected(kwargs) -> None:
    """`message` and `participant` are separate fields because a single
    overloaded `target` meant the same string was an envelope id at one call
    site and a participant name at another, with nothing to catch the mix-up."""
    with pytest.raises(ValueError):
        Intervention(**kwargs)
