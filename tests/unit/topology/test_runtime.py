from __future__ import annotations

import pytest
from fakes import FakeParticipant

from ant_ai.core.events import (
    CompletedEvent,
    FinalAnswerEvent,
    StartEvent,
    TopologyEvent,
)
from ant_ai.topology.builtins.shapes import Static, mesh
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.strategy import Pipeline

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _ensemble(participants: list[FakeParticipant], **kwargs) -> Ensemble:
    """An ensemble over *participants*, meshed unless a test says otherwise.

    Pipeline settings are accepted as flat keywords so the tests read as
    statements about behaviour rather than about assembly.
    """
    people = {p.name: p for p in participants}
    seed = kwargs.pop("seed", None)
    fields = set(Pipeline.model_fields)
    pipeline = Pipeline(
        **{k: v for k, v in kwargs.items() if k in fields},
    )
    if "stages" not in kwargs:
        pipeline = pipeline.model_copy(update={"stages": [mesh(list(people))]})
    rest = {k: v for k, v in kwargs.items() if k not in fields}
    return Ensemble(participants=people, pipeline=pipeline, seed=seed or (), **rest)


async def test_halts_when_a_participant_submits() -> None:
    a = FakeParticipant("a", submitted=True)
    b = FakeParticipant("b")
    ensemble = _ensemble([a, b], max_rounds=5)

    await ensemble.ainvoke("task")

    assert len(a.briefs) == 1


async def test_halts_at_max_rounds() -> None:
    a = FakeParticipant("a")
    ensemble = _ensemble([a], max_rounds=3)

    await ensemble.ainvoke("task")

    assert len(a.briefs) == 3


async def test_final_answer_comes_from_the_last_round() -> None:
    a = FakeParticipant("a", message="the answer", submitted=True)
    assert await _ensemble([a]).ainvoke("task") == "the answer"


async def test_seed_binds_before_the_first_round() -> None:
    """Round 0 uses the colony's declared edges, so the first turn behaves
    exactly as a colony does today."""
    a, b = FakeParticipant("a", submitted=True), FakeParticipant("b")
    ensemble = _ensemble([a, b], seed=(Link(src="a", dst="b"),))

    await ensemble.ainvoke("task")

    assert b.bound[0] == frozenset({"a"})
    assert ensemble.graph.links(0) == [Link(src="a", dst="b")]


async def test_a_raising_participant_does_not_kill_the_ensemble() -> None:
    """A failed activation is structural signal, recorded rather than swallowed."""
    boom = FakeParticipant("boom", raises="exploded")
    ok = FakeParticipant("ok", message="still here")
    ensemble = _ensemble([boom, ok], max_rounds=2)

    result = await ensemble.ainvoke("task")

    assert "still here" in result
    errors = [a.error for a in ensemble.graph.activations.values() if a.error]
    assert errors and all("exploded" in e for e in errors)


async def test_emits_one_topology_event_per_transition() -> None:
    a, b = FakeParticipant("a"), FakeParticipant("b")
    events = [e async for e in _ensemble([a, b], max_rounds=3).stream("task")]

    kinds = [e.kind for e in events]
    topology = [e for e in events if isinstance(e, TopologyEvent)]

    assert kinds[0] == "start" and kinds[-1] == "completed"
    assert isinstance(events[0], StartEvent) and isinstance(events[-1], CompletedEvent)
    # Three rounds run, two of which hand a topology to the next round.
    assert [e.round for e in topology] == [1, 2]


async def test_participant_events_stream_through_live() -> None:
    a = FakeParticipant("a", message="hello", submitted=True)
    events = [e async for e in _ensemble([a]).stream("task")]

    assert any(isinstance(e, FinalAnswerEvent) and e.content == "hello" for e in events)


async def test_topology_event_carries_reasons() -> None:
    a, b = FakeParticipant("a"), FakeParticipant("b")
    shape = Static(links=(Link(src="a", dst="b", weight=0.7, reason="why"),))
    events = [
        e
        async for e in Ensemble(
            participants={"a": a, "b": b},
            pipeline=Pipeline(stages=[shape], max_rounds=2),
        ).stream("task")
    ]

    link = next(e for e in events if isinstance(e, TopologyEvent)).links[0]
    assert (link.src, link.dst, link.reason) == ("a", "b", "why")


async def test_delivery_mode_fills_the_next_inbox() -> None:
    a, b = FakeParticipant("a"), FakeParticipant("b")
    ensemble = Ensemble(
        participants={"a": a, "b": b},
        pipeline=Pipeline(
            stages=[Static(links=(Link(src="a", dst="b"),))],
            materialiser=DeliveryMaterialiser(),
            max_rounds=2,
        ),
    )

    await ensemble.ainvoke("task")

    assert [e.sender for e in b.briefs[1].inbox] == ["a"]
    assert a.briefs[1].inbox == ()


async def test_invocations_are_recorded_separately_from_visibility() -> None:
    """Granted vs exercised: the gap between them is signal."""
    a = FakeParticipant("a", invoked=("b",))
    b = FakeParticipant("b")
    ensemble = _ensemble(
        [a, b],
        seed=(Link(src="b", dst="a"), Link(src="a", dst="b")),
    )

    await ensemble.ainvoke("task")

    assert [(link.src, link.dst) for link in ensemble.graph.unused_visibility(0)] == [
        ("a", "b")
    ]


async def test_final_answer_is_deterministic_not_completion_ordered() -> None:
    """`turns` is filled concurrently, so picking "the last public message"
    would make the result depend on which agent happened to finish first."""
    people = [
        FakeParticipant("zeta", message="Z"),
        FakeParticipant("alpha", message="A"),
    ]

    results = {await _ensemble(people, max_rounds=1).ainvoke("task") for _ in range(5)}

    assert len(results) == 1
    assert results == {"[alpha] A\n\n[zeta] Z"}


async def test_a_submitting_participant_owns_the_answer() -> None:
    people = [
        FakeParticipant("reviewer", message="looks wrong to me"),
        FakeParticipant("developer", message="here is the code", submitted=True),
    ]

    assert await _ensemble(people, max_rounds=2).ainvoke("task") == "here is the code"


class Reacting(FakeParticipant):
    """A participant that says what it did with what it was handed."""

    def __init__(self, name: str, *, reaction: str = "consume", to: tuple = ()) -> None:
        super().__init__(name)
        self.reaction = reaction
        self.to = to

    async def act(self, brief, *, ctx=None):
        self.briefs.append(brief)
        turn = self._make_turn(brief.round)
        yield turn.model_copy(
            update={
                "reactions": {e.id: self.reaction for e in brief.inbox},
                "rerouted": (
                    {e.id: self.to for e in brief.inbox}
                    if self.reaction == "reroute"
                    else {}
                ),
            }
        )


async def test_a_waited_message_stays_in_the_buffer() -> None:
    """DIG's wait: declining to act on something is not the same as losing it,
    so the message is still there next round and still counts as outstanding."""
    a, b = FakeParticipant("a"), Reacting("b", reaction="wait")
    ensemble = _ensemble([a, b], max_rounds=3, materialiser=DeliveryMaterialiser())

    await ensemble.ainvoke("task")

    first = b.briefs[1].inbox[0]
    assert first.id in {e.id for e in b.briefs[2].inbox}
    assert len(b.briefs[2].inbox) == 2
    assert any(
        e.kind == "delivers" and e.action == "wait" for e in ensemble.graph.edges
    )
    assert first.id not in ensemble.graph.consumed()
    assert first.id in {m.id for m in ensemble.graph.unsettled()}


async def test_a_waited_message_is_not_yet_an_ancestor() -> None:
    """Lineage is what a turn consumed. Attributing what it explicitly put off
    would make every waited message an ancestor of work that has not read it."""
    a, b = FakeParticipant("a"), Reacting("b", reaction="wait")
    ensemble = _ensemble([a, b], max_rounds=2, materialiser=DeliveryMaterialiser())

    await ensemble.ainvoke("task")

    produced = [m for m in ensemble.graph.messages.values() if m.sender == "b"]
    assert produced and all(m.parents == () for m in produced)


async def test_an_agent_can_hand_a_message_to_someone_else() -> None:
    """Reroute as a *recipient's* decision, which is where the paper puts it:
    the message moves without the topology changing, and the count that
    `ExcessiveRerouting` reads sees it."""
    a = FakeParticipant("a")
    b = Reacting("b", reaction="reroute", to=("c",))
    c = FakeParticipant("c")
    ensemble = _ensemble([a, b, c], max_rounds=3, materialiser=DeliveryMaterialiser())

    await ensemble.ainvoke("task")

    handed = b.briefs[1].inbox[0]
    assert handed.id in {e.id for e in c.briefs[2].inbox}
    assert ensemble.graph.reroutes(handed.id) == 1


async def test_a_discarded_message_settles_without_being_consumed() -> None:
    a, b = FakeParticipant("a"), Reacting("b", reaction="discard")
    ensemble = _ensemble([a, b], max_rounds=2, materialiser=DeliveryMaterialiser())

    await ensemble.ainvoke("task")

    dropped = b.briefs[1].inbox[0]
    assert dropped.id in ensemble.graph.discarded()
    assert dropped.id not in {m.id for m in ensemble.graph.unsettled()}
    assert dropped.id not in ensemble.graph.inputs_of(
        next(
            a.id
            for a in ensemble.graph.activations.values()
            if a.participant == "b" and a.round == 1
        )
    )
