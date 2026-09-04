from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel

from ant_ai.observer import obs
from ant_ai.topology.participant import Envelope, Participant, Turn
from ant_ai.topology.plan import RoundPlan

__all__ = [
    "DeliveryMaterialiser",
    "TopologyMaterialiser",
    "VisibilityMaterialiser",
]

type Inboxes = dict[str, tuple[Envelope, ...]]


@runtime_checkable
class TopologyMaterialiser(Protocol):
    """Turns a decided plan into something that actually constrains a round.

    Takes the whole `RoundPlan` rather than a view plus the turns plus a bag of
    stage-created messages, and returns just the inboxes: delivery is the
    component that owns "what lands in an inbox", so folding those together here
    is what removed the last of the round loop's hand-merges.
    """

    async def apply(
        self, plan: RoundPlan, participants: Mapping[str, Participant]
    ) -> Inboxes: ...


class VisibilityMaterialiser(BaseModel):
    """Reachability as the peer tool set. The default.

    Each participant is bound to exactly the peers it can reach, so its address
    book *is* the topology. Descriptions come from AgentCards as they always have,
    nothing is injected into prompts, and the agent still decides whom to call —
    the topology layer only decides whom it *could* call.

    Note the reversal: `Link(src, dst)` is information flow, so `dst` is the one
    that gets a tool calling `src`.
    """

    async def apply(
        self, plan: RoundPlan, participants: Mapping[str, Participant]
    ) -> Inboxes:
        unbindable: list[str] = []

        for name, participant in participants.items():
            peers = {
                src: participants[src]
                for src in plan.sources_for(name)
                if src in participants
            }
            if not await participant.bind_peers(peers):
                unbindable.append(name)

        if unbindable:
            # Reported rather than returned: a participant whose reachability
            # cannot change is a silent lie about what the topology did, and the
            # only caller of the old return value was this log line.
            await obs.event(
                "topology.unbindable", round=plan.round, participants=unbindable
            )

        # Notices still have to arrive: there are no peer tools for a message a
        # stage invented, so they ride the brief's inbox even in visibility mode.
        return {n: plan.notices.get(n, ()) for n in participants}


class _Delivery(NamedTuple):
    """One message about to land in one inbox, with the rank it lands at."""

    recipient: str
    weight: float
    sender: str
    envelope: Envelope


@dataclass(frozen=True, slots=True)
class _Reachability:
    """Who a round's links let reach whom, as a lookup.

    Built once per round rather than re-scanned per message: the rule below is
    asked about every (sender, recipient) pair a turn produces.
    """

    weights: Mapping[tuple[str, str], float]
    known: frozenset[str]

    @classmethod
    def of(cls, plan: RoundPlan, participants: Iterable[str]) -> _Reachability:
        return cls(
            weights={(link.src, link.dst): link.weight for link in plan.links},
            known=frozenset(participants),
        )

    @property
    def constrained(self) -> bool:
        """False when no stage wrote links, i.e. nobody has an opinion yet."""
        return bool(self.weights)

    def delivers(self, sender: str, recipient: str) -> bool:
        if recipient not in self.known or recipient == sender:
            return False
        return not self.constrained or (sender, recipient) in self.weights

    def weight(self, sender: str, recipient: str) -> float:
        return self.weights.get((sender, recipient), 1.0)

    def audience(self, sender: str) -> tuple[str, ...]:
        """Whom *sender* may reach — the routing for a message it did not address."""
        return tuple(
            recipient
            for (src, recipient) in self.weights
            if src == sender and self.delivers(sender, recipient)
        )


class DeliveryMaterialiser(BaseModel):
    """Reachability as routed messages, addressing as the sender's own choice.

    One rule covers both halves of the split the topology layer is built on:

        delivery = selection ∩ reachability

    *Selection* is `Envelope.recipients` — whom the sender addressed. *Reachability*
    is `plan.links` — whom a stage decided it may reach. Each side has a default,
    and the defaults are what make two very different published methods fit the
    same materialiser:

    - A sender that addressed **nobody** is routed by the links, as a
      matcher-driven run has always been.
    - A round in which **no stage wrote links** has no opinion on reachability,
      so an addressed message goes where it was addressed. That is a method
      whose agents name their own correspondents, running with no matcher under
      it at all.
    - A message addressed to somebody unreachable is *not* delivered. It stays a
      generated event that reached no one, which is exactly the failure
      `OrphanedEvent` looks for — silently widening reachability to whoever was
      named would erase the pathology instead of reporting it.

    Inboxes are ordered by descending relevance, the ordering coming from the
    plan (`RoundPlan.in_neighbours`); this class only applies it. Notices a stage
    created — an emitted broadcast, a rerouted message — are appended last and
    are not subject to either constraint, because a repair that the topology
    could veto would not be a repair.
    """

    include_public: bool = True
    """Fall back to a sender's public message when it produced no private one."""

    async def apply(
        self, plan: RoundPlan, participants: Mapping[str, Participant]
    ) -> Inboxes:
        reach = _Reachability.of(plan, participants)
        picked: dict[str, list[_Delivery]] = {name: [] for name in participants}
        for sender, turn in plan.turns.items():
            for delivery in self._route(turn, sender=sender, reach=reach):
                picked[delivery.recipient].append(delivery)

        return {
            name: _inbox(picked[name], plan.notices.get(name, ()))
            for name in participants
        }

    def _route(
        self, turn: Turn, *, sender: str, reach: _Reachability
    ) -> Iterator[_Delivery]:
        """Where one turn's messages go — selection first, reachability always."""
        addressed = [e for e in turn.outputs if e.recipients]
        if addressed:
            # The sender said who this was for. Its public message is its
            # contribution to the record, not a second copy for everyone it
            # happens to be wired to.
            for envelope in addressed:
                yield from self._addressed_to(
                    envelope.recipients, envelope, sender, reach
                )
            return

        fallback = turn.private or (turn.public if self.include_public else None)
        if fallback is not None:
            yield from self._addressed_to(
                reach.audience(sender), fallback, sender, reach
            )

    @staticmethod
    def _addressed_to(
        recipients: Iterable[str],
        envelope: Envelope,
        sender: str,
        reach: _Reachability,
    ) -> Iterator[_Delivery]:
        for recipient in recipients:
            if reach.delivers(sender, recipient):
                yield _Delivery(
                    recipient, reach.weight(sender, recipient), sender, envelope
                )


def _inbox(
    deliveries: list[_Delivery], notices: tuple[Envelope, ...]
) -> tuple[Envelope, ...]:
    """One participant's inbox: ranked deliveries, then notices, deduplicated.

    Ties break on sender name so an ablation reproduces, and a message routed
    twice — addressed *and* rerouted, say — is still delivered once.
    """
    ranked = sorted(deliveries, key=lambda d: (-d.weight, d.sender))
    seen: set[str] = set()
    inbox: list[Envelope] = []
    for envelope in chain((d.envelope for d in ranked), notices):
        if envelope.id not in seen:
            seen.add(envelope.id)
            inbox.append(envelope)
    return tuple(inbox)
