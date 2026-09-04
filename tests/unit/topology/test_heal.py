from __future__ import annotations

import pytest

from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.heal import Detector, Heal, apply_interventions
from ant_ai.topology.participant import Envelope, ParticipantProfile, Turn
from ant_ai.topology.plan import (
    SUPERVISOR,
    Finding,
    Intervention,
    RoundPlan,
    RunContext,
)

pytestmark = [pytest.mark.unit, pytest.mark.topology]


def _recorded(graph: InteractionGraph, name: str, **kwargs) -> Turn:
    """A turn already written into the graph, as `Ensemble` would leave it."""
    activation = graph.record_activation(name, round=0)
    turn = Turn(
        participant=name,
        outputs=(
            Envelope(
                sender=name, content=f"{name} says", visibility="public", **kwargs
            ),
        ),
        submitted=kwargs.get("terminal", False),
    )
    graph.record_message(turn.public, activation_id=activation)
    return turn


def _fixture(*names: str, **kwargs):
    graph = InteractionGraph()
    turns = {n: _recorded(graph, n, **kwargs) for n in names}
    ctx = RunContext(
        graph=graph,
        participants=tuple(ParticipantProfile(name=n) for n in names),
    )
    return graph, RoundPlan(round=1, turns=turns), ctx


def test_inject_appends_to_the_message_and_records_an_edge() -> None:
    graph, plan, ctx = _fixture("a")
    target = plan.turns["a"].public

    healed = apply_interventions(
        plan,
        ctx,
        [
            Intervention(
                kind="inject",
                message=target.id,
                content="you missed something",
                reason="early termination",
            )
        ],
    )

    assert "you missed something" in healed.turns["a"].public.content
    assert graph.messages[target.id].content == healed.turns["a"].public.content
    assert any(e.kind == "intervenes" and e.action == "inject" for e in graph.edges)


def test_rerouting_a_submit_un_terminates_it() -> None:
    """ "Reroute the submit back to the issuing agent" is only a correction if it
    also stops the run ending — an un-terminated submit is what buys the agent
    another round."""
    graph, plan, ctx = _fixture("a", terminal=True)
    target = plan.turns["a"].public

    healed = apply_interventions(
        plan,
        ctx,
        [
            Intervention(
                kind="reroute",
                message=target.id,
                recipients=("a",),
                reason="early termination",
            )
        ],
    )

    assert healed.turns["a"].submitted is False
    assert graph.terminal_message() is None
    assert [e.id for e in healed.notices["a"]] == [target.id]
    assert graph.intervention_count(target.id, action="reroute") == 1


def test_reroute_moves_the_message_itself_and_leaves_reachability_alone() -> None:
    """A repair moves *this* message, and does not rewire the run to do it.

    A forced link would say "whatever `a` says next goes to `b`", which
    delivers the wrong message once `a` has moved on and nothing at all once it
    has fallen silent — the state every stall-shaped repair exists for.
    """
    graph, plan, ctx = _fixture("a")
    plan = plan.with_links((Link(src="a", dst="b", weight=0.9), Link(src="c", dst="d")))
    target = plan.turns["a"].public

    healed = apply_interventions(
        plan,
        ctx,
        [
            Intervention(
                kind="reroute", message=target.id, recipients=("b",), reason="fix"
            )
        ],
    )

    assert [e.id for e in healed.notices["b"]] == [target.id]
    assert [(link.src, link.dst) for link in healed.links] == [("a", "b"), ("c", "d")]


def test_reroute_reaches_a_message_older_than_this_rounds_turns() -> None:
    """The orphan case: the message to repair was said rounds ago, and its sender
    has not spoken since. Locating it in the graph rather than in `plan.turns` is
    what lets `OrphanedEvent` actually clear the orphan it reported."""
    graph, plan, ctx = _fixture("a")
    stale = Envelope(sender="ghost", content="nobody heard this", round=0)
    graph.record_emission(stale)

    healed = apply_interventions(
        RoundPlan(round=1, turns=plan.turns),
        ctx,
        [
            Intervention(
                kind="reroute",
                message=stale.id,
                recipients=("ghost",),
                reason="orphaned event",
            )
        ],
    )

    assert [e.id for e in healed.notices["ghost"]] == [stale.id]


def test_drop_removes_the_message_from_delivery_and_from_outstanding_work() -> None:
    graph, plan, ctx = _fixture("a")
    target = plan.turns["a"].public

    healed = apply_interventions(
        plan, ctx, [Intervention(kind="drop", message=target.id)]
    )

    assert healed.turns["a"].public is None
    assert graph.unsettled() == ()


def test_emit_creates_a_message_addressed_to_its_recipients() -> None:
    graph, plan, ctx = _fixture("a")

    healed = apply_interventions(
        plan,
        ctx,
        [
            Intervention(
                kind="emit", content="everyone stop", recipients=("a", "b"), reason="dl"
            )
        ],
    )

    assert set(healed.notices) == {"a", "b"}
    envelope = healed.notices["a"][0]
    assert envelope.sender == SUPERVISOR
    assert envelope.id in graph.messages


def test_emit_without_recipients_reaches_every_participant() -> None:
    graph, plan, ctx = _fixture("a", "b", "c")

    healed = apply_interventions(plan, ctx, [Intervention(kind="emit", content="hi")])

    assert set(healed.notices) == {"a", "b", "c"}


def test_an_intervention_naming_an_unknown_message_is_skipped() -> None:
    graph, plan, ctx = _fixture("a")

    healed = apply_interventions(plan, ctx, [Intervention(kind="drop", message="nope")])

    assert healed.turns["a"].public is not None


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
    overloaded `target` meant the same string was an envelope id at one call site
    and a participant name at another, with nothing to catch the mix-up."""
    with pytest.raises(ValueError):
        Intervention(**kwargs)


class _Boom(Detector):
    pattern: str = "BOOM"

    async def detect(self, graph, ctx):
        raise RuntimeError("detector is broken")


class _Always(Detector):
    pattern: str = "X"

    async def detect(self, graph, ctx):
        return [self.finding(ctx, "always")]


async def test_a_broken_detector_does_not_abort_the_run() -> None:
    """Diagnostic machinery aborting the run it was meant to protect would be
    exactly backwards."""
    findings = await Heal(detectors=[_Boom(), _Always()]).inspect(
        InteractionGraph(), RunContext()
    )

    assert [f.pattern for f in findings] == ["X"]


async def test_the_finding_helper_attributes_detector_and_round() -> None:
    """Public on the base class so anyone writing a detector can reach it; the
    private base this replaced kept it out of reach of its own subclasses."""
    findings = await _Always().detect(InteractionGraph(), RunContext(round=3))

    assert findings[0] == Finding(
        pattern="X", detector="_Always", round=3, explanation="always"
    )


async def test_heal_is_a_stage_and_accumulates_history() -> None:
    healer = Heal(detectors=[_Always()])
    ctx = RunContext(round=1)

    plan = await healer.apply(RoundPlan(round=2), ctx)
    await healer.apply(RoundPlan(round=3), ctx)

    assert [f.pattern for f in plan.findings] == ["X"]
    assert len(healer.history) == 2


async def test_a_duck_typed_detector_still_works() -> None:
    """`Heal.detectors` skips validation on purpose, so a detector need not
    inherit — only have `pattern` and `detect`."""

    class Loose:
        pattern = "LOOSE"

        async def detect(self, graph, ctx):
            return [Finding(pattern="LOOSE", explanation="fine")]

    findings = await Heal(detectors=[Loose()]).inspect(InteractionGraph(), RunContext())

    assert [f.pattern for f in findings] == ["LOOSE"]
