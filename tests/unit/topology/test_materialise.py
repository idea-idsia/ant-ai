from __future__ import annotations

from unittest.mock import patch

import pytest
from fakes import FakeParticipant

from ant_ai.observer import obs
from ant_ai.topology.graph import Link
from ant_ai.topology.materialise import DeliveryMaterialiser, VisibilityMaterialiser
from ant_ai.topology.participant import Envelope, Turn
from ant_ai.topology.plan import RoundPlan

pytestmark = [pytest.mark.unit, pytest.mark.topology]


async def test_visibility_binds_the_needer_to_the_offerer() -> None:
    """The reversal footgun, pinned by a test rather than a comment.

    `Link(src, dst)` is information flow, so `dst` is the one that gets a tool
    calling `src`. Getting this backwards produces a plausible-looking graph
    that routes exactly wrong.
    """
    arch, dev = FakeParticipant("arch"), FakeParticipant("dev")
    view = RoundPlan(round=1, links=(Link(src="arch", dst="dev"),))

    await VisibilityMaterialiser().apply(view, {"arch": arch, "dev": dev})

    assert dev.bound == [frozenset({"arch"})]
    assert arch.bound == [frozenset()]


async def test_visibility_detaches_peers_that_dropped_out() -> None:
    dev = FakeParticipant("dev")
    participants = {"arch": FakeParticipant("arch"), "dev": dev}

    await VisibilityMaterialiser().apply(
        RoundPlan(round=1, links=(Link(src="arch", dst="dev"),)), participants
    )
    await VisibilityMaterialiser().apply(RoundPlan(round=2), participants)

    assert dev.bound == [frozenset({"arch"}), frozenset()]


async def test_visibility_reports_unbindable_participants() -> None:
    """A remote A2A peer cannot have tools attached from another process. The
    materialiser reports that rather than pretending — as an observability event,
    since a return value nobody read was the only thing carrying it."""
    seen: list[dict] = []

    class Remote(FakeParticipant):
        async def bind_peers(self, peers) -> bool:
            await super().bind_peers(peers)
            return False

    with patch.object(obs, "event", new=_capture(seen)):
        await VisibilityMaterialiser().apply(
            RoundPlan(round=1, links=(Link(src="a", dst="b"),)),
            {"a": FakeParticipant("a"), "b": Remote("b")},
        )

    assert seen == [{"name": "topology.unbindable", "round": 1, "participants": ["b"]}]


async def test_delivery_orders_inbox_by_descending_relevance() -> None:
    """DyTopo's Sigma_sigma(t): aggregation in descending relevance."""
    turns = {
        "low": Turn(participant="low"),
        "high": Turn(participant="high"),
    }
    for name in turns:
        turns[name] = FakeParticipant(name, message=f"from {name}")._make_turn()

    view = RoundPlan(
        round=1,
        links=(
            Link(src="low", dst="dev", weight=0.2),
            Link(src="high", dst="dev", weight=0.9),
        ),
    )
    participants = {n: FakeParticipant(n) for n in ("low", "high", "dev")}

    result = await DeliveryMaterialiser().apply(
        view.model_copy(update={"turns": turns}), participants
    )

    assert [e.sender for e in result["dev"]] == ["high", "low"]
    assert result["low"] == ()


def _capture(sink: list[dict]):
    async def _event(name: str, **fields) -> None:
        sink.append({"name": name, **fields})

    return _event


async def test_delivery_honours_an_address_the_sender_wrote() -> None:
    """Selection is the sender's, reachability the topology's, and this is the
    intersection: `arch` addressed `dev` and only `dev` gets it, even though the
    links reach further."""
    turn = Turn(
        participant="arch",
        outputs=(
            Envelope(sender="arch", content="for the record", visibility="public"),
            Envelope(sender="arch", content="your chunk", recipients=("dev",)),
        ),
    )
    plan = RoundPlan(
        round=1,
        turns={"arch": turn},
        links=(Link(src="arch", dst="dev"), Link(src="arch", dst="ops")),
    )
    participants = {n: FakeParticipant(n) for n in ("arch", "dev", "ops")}

    result = await DeliveryMaterialiser().apply(plan, participants)

    assert [e.content for e in result["dev"]] == ["your chunk"]
    assert result["ops"] == ()


async def test_an_address_the_topology_does_not_reach_is_not_delivered() -> None:
    """The orphan, preserved rather than papered over: quietly widening
    reachability to whoever was named would erase the very failure
    `OrphanedEvent` reports."""
    turn = Turn(
        participant="arch",
        outputs=(Envelope(sender="arch", content="psst", recipients=("ops",)),),
    )
    plan = RoundPlan(
        round=1, turns={"arch": turn}, links=(Link(src="arch", dst="dev"),)
    )

    result = await DeliveryMaterialiser().apply(
        plan, {n: FakeParticipant(n) for n in ("arch", "dev", "ops")}
    )

    assert result["ops"] == ()
    assert result["dev"] == ()


async def test_with_no_links_at_all_an_address_still_arrives() -> None:
    """A round no stage wrote links for has no opinion on reachability, so a
    method whose agents name their own correspondents runs with no matcher
    under it — which is what the DIG paper's own runtime does."""
    turn = Turn(
        participant="arch",
        outputs=(Envelope(sender="arch", content="your chunk", recipients=("dev",)),),
    )
    plan = RoundPlan(round=1, turns={"arch": turn})

    result = await DeliveryMaterialiser().apply(
        plan, {n: FakeParticipant(n) for n in ("arch", "dev")}
    )

    assert [e.content for e in result["dev"]] == ["your chunk"]


async def test_notices_arrive_whatever_the_topology_says() -> None:
    """A repair the topology could veto would not be a repair."""
    moved = Envelope(sender="arch", content="handle this")
    plan = RoundPlan(round=1, notices={"dev": (moved,)})

    result = await DeliveryMaterialiser().apply(
        plan, {n: FakeParticipant(n) for n in ("arch", "dev")}
    )

    assert [e.id for e in result["dev"]] == [moved.id]
