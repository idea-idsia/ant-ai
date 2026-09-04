from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ant_ai.observer import obs
from ant_ai.topology.participant import Envelope, Participant
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
        ranked = {(link.src, link.dst): link.weight for link in plan.links}
        constrained = bool(plan.links)
        picked: dict[str, list[tuple[float, str, Envelope]]] = {
            name: [] for name in participants
        }

        for sender, turn in plan.turns.items():
            addressed = [e for e in turn.outputs if e.recipients]
            for envelope in addressed:
                for name in envelope.recipients:
                    if name not in picked or name == sender:
                        continue
                    if constrained and (sender, name) not in ranked:
                        continue
                    picked[name].append(
                        (ranked.get((sender, name), 1.0), sender, envelope)
                    )
            if addressed:
                # The sender said who this was for. Its public message is its
                # contribution to the record, not a second copy for everyone
                # it happens to be wired to.
                continue
            fallback = turn.private or (turn.public if self.include_public else None)
            if fallback is None:
                continue
            for name in picked:
                if (sender, name) in ranked:
                    picked[name].append((ranked[(sender, name)], sender, fallback))

        inboxes: Inboxes = {}
        for name in participants:
            seen: set[str] = set()
            delivered: list[Envelope] = []
            for _, _, envelope in sorted(
                picked[name], key=lambda item: (-item[0], item[1])
            ):
                if envelope.id not in seen:
                    seen.add(envelope.id)
                    delivered.append(envelope)
            for envelope in plan.notices.get(name, ()):
                if envelope.id not in seen:
                    seen.add(envelope.id)
                    delivered.append(envelope)
            inboxes[name] = tuple(delivered)

        return inboxes
