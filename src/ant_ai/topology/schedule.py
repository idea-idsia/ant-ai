from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, PrivateAttr

from ant_ai.topology.participant import Envelope, Participant

__all__ = ["BufferScheduler", "RoundScheduler", "Scheduler"]


@runtime_checkable
class Scheduler(Protocol):
    """Decides who activates on a given tick.

    A seam because published methods disagree about time, and the disagreement
    is not cosmetic: under a hard barrier where everyone acts every round,
    "work remains but nobody acted" is unreachable by construction, so a
    detector looking for a stall can never fire. What counts as an observable
    failure depends on what counts as a tick.
    """

    def activations(
        self,
        *,
        round: int,
        participants: Mapping[str, Participant],
        inboxes: Mapping[str, tuple[Envelope, ...]],
    ) -> frozenset[str]: ...


class RoundScheduler(BaseModel):
    """A synchronous barrier: everyone activates, every round. The default."""

    def activations(
        self,
        *,
        round: int,
        participants: Mapping[str, Participant],
        inboxes: Mapping[str, tuple[Envelope, ...]],
    ) -> frozenset[str]:
        return frozenset(participants)


class BufferScheduler(BaseModel):
    """Event-driven activation: act when your buffer changed and you are idle.

    The rule a buffer-based method states as
    `activate(a, t) = 1 iff B_a(t) != B_a(t-1) and activate(a, t-1) = 0`.

    The second conjunct guards against re-entering an agent that is still
    running, and is dropped here: under a round barrier an activation always
    completes before the next tick, so every participant is idle at the point
    this is asked. Keeping it would make agents act on alternate rounds, which
    is an artefact of the clock rather than the paper's rule.

    This is what makes stall-shaped failures observable at all. Under a synchronous
    barrier every agent acts every round, so an empty activation set never
    happens and a deadlock detector is dead code; an agent that nobody routed to
    still burns a turn, so an orphaned message never manifests either. Here an
    empty activation set is a legitimate state, and a detector can see it.

    The clock is still the round loop's — this is event-driven *activation*, not
    a fully asynchronous runtime — so an agent activates at the next barrier
    rather than the instant its buffer changes. That difference costs latency
    realism, not detectability.
    """

    activate_all_at_start: bool = Field(
        default=True,
        description="Whether every participant acts in round 0. False starts only "
        "those the seed topology routes to, which is the stricter reading of the "
        "paper but needs a seeded inbox to start at all.",
    )

    _last_seen: dict[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)

    def activations(
        self,
        *,
        round: int,
        participants: Mapping[str, Participant],
        inboxes: Mapping[str, tuple[Envelope, ...]],
    ) -> frozenset[str]:
        if round == 0 and self.activate_all_at_start:
            self._last_seen = {
                name: tuple(e.id for e in inboxes.get(name, ()))
                for name in participants
            }
            return frozenset(participants)

        active: set[str] = set()
        for name in participants:
            buffer = tuple(e.id for e in inboxes.get(name, ()))
            if buffer and buffer != self._last_seen.get(name, ()):
                active.add(name)
            self._last_seen[name] = buffer
        return frozenset(active)
