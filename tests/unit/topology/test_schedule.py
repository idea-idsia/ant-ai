from __future__ import annotations

import pytest
from fakes import FakeParticipant

from ant_ai.topology.participant import Envelope
from ant_ai.topology.schedule import BufferScheduler, RoundScheduler

pytestmark = [pytest.mark.unit, pytest.mark.topology]

PEOPLE = {name: FakeParticipant(name) for name in ("a", "b")}


def _envelope(sender: str) -> Envelope:
    return Envelope(sender=sender, content="hi")


def test_round_scheduler_activates_everyone() -> None:
    scheduler = RoundScheduler()
    assert scheduler.activations(round=3, participants=PEOPLE, inboxes={}) == frozenset(
        PEOPLE
    )


def test_buffer_scheduler_starts_everyone_then_follows_the_buffers() -> None:
    scheduler = BufferScheduler()

    assert scheduler.activations(round=0, participants=PEOPLE, inboxes={}) == frozenset(
        PEOPLE
    )

    active = scheduler.activations(
        round=1, participants=PEOPLE, inboxes={"a": (_envelope("b"),), "b": ()}
    )
    assert active == frozenset({"a"})


def test_an_unchanged_buffer_does_not_reactivate() -> None:
    """DIG's rule is `B_a(t) != B_a(t-1)`. Re-running an agent on a buffer it
    already read is the busy-wait the rule exists to prevent."""
    scheduler = BufferScheduler()
    inboxes = {"a": (_envelope("b"),), "b": ()}

    scheduler.activations(round=0, participants=PEOPLE, inboxes={})
    assert scheduler.activations(round=1, participants=PEOPLE, inboxes=inboxes)
    assert not scheduler.activations(round=2, participants=PEOPLE, inboxes=inboxes)


def test_an_empty_activation_set_is_reachable() -> None:
    """The whole reason this scheduler exists: under a barrier `V_A(t)` is never
    empty, so DIG's Deadlock detector is unreachable by construction."""
    scheduler = BufferScheduler()
    scheduler.activations(round=0, participants=PEOPLE, inboxes={})

    assert (
        scheduler.activations(round=1, participants=PEOPLE, inboxes={}) == frozenset()
    )
