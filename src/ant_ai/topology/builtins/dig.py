"""DIG to Heal: detect coordination pathologies structurally and correct them.

arXiv:2603.00309. Seven detectors, which the paper's section 5 groups two ways:
*reachability and termination* (ET, MC, OE, DL) and *progress* (ER, CLA, RSP).
Each is a query over `InteractionGraph` and each prescribes its own correction.

The paper's delta is supervision, not routing, so `DigToHeal` contributes a single
`Heal` stage and composes with whatever routing strategy it is layered on:

    DyTopo(embedder=e) | DigToHeal()

Two of its other choices are load-bearing rather than cosmetic. `BufferScheduler`,
because a synchronous barrier makes the Deadlock detector unreachable by
construction. `DeliveryMaterialiser`, because detection is defined over messages
that exist, are delivered and are consumed — under visibility mode a peer call
collapses all three into one tool call and there is nothing to inspect between
them.
"""

from __future__ import annotations

from itertools import combinations
from typing import Annotated, Any, ClassVar

from pydantic import ConfigDict, Field, SkipValidation

from ant_ai.topology.graph import InteractionGraph
from ant_ai.topology.heal import Detector, Heal
from ant_ai.topology.materialise import DeliveryMaterialiser
from ant_ai.topology.participant import Envelope
from ant_ai.topology.plan import SUPERVISOR, Finding, Intervention, RunContext
from ant_ai.topology.schedule import BufferScheduler
from ant_ai.topology.strategy import Pipeline, TopologyStrategy

__all__ = [
    "CrossLineageAggregation",
    "Deadlock",
    "DigToHeal",
    "EarlyTermination",
    "ExcessiveRerouting",
    "JudgeHealing",
    "LLMJudge",
    "MissingCompletion",
    "OrphanedEvent",
    "RepeatedSubproblem",
    "dig_detectors",
]


# ---------------------------------------------------------------------------
# The paper's reading of the record
#
# `InteractionGraph` answers what happened; these three say what DIG takes it to
# mean. They live here, with the paper making the claim, so that the record stays
# a neutral account a different method can read differently.
# ---------------------------------------------------------------------------


def reachable_work(graph: InteractionGraph) -> tuple[Envelope, ...]:
    """DIG's `R(t)`: work that exists and has not been consumed."""
    return graph.unsettled()


def orphans(
    graph: InteractionGraph, *, round: int, window: int
) -> tuple[Envelope, ...]:
    """Undelivered messages old enough to count, per DIG's time window."""
    return tuple(m for m in graph.undelivered() if round - m.round >= window)


def is_problem_reducing(graph: InteractionGraph, activation_id: str) -> bool:
    """`|O_v| <= |I_v|`: the activation narrowed the frontier rather than widening it.

    DIG's discriminator, not a property of the record. An activation with no
    inputs is problem-*generating* by definition — it can only have added work —
    which is what keeps every round-0 turn out of the Repeated Subproblem
    detector.
    """
    inputs = graph.inputs_of(activation_id)
    return bool(inputs) and len(graph.outputs_of(activation_id)) <= len(inputs)


class EarlyTermination(Detector):
    """ET — a Submit was generated while reachable work remains unconsumed.

    The paper's condition is "no directed path from some `e` in `R(t)` to
    `e_inf`". Unconsumed work has no outgoing consume edge and therefore no
    descendants at all, so the path test reduces to `R(t)` being non-empty —
    the same predicate, one set lookup instead of a traversal.

    Healing reroutes the submit back to its issuer, which in this framework
    also *un-terminates* it: the run does not stop, and the agent that claimed
    completion gets told what it left behind.
    """

    pattern: str = "ET"

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        terminal = graph.terminal_message()
        if terminal is None:
            return []
        # Work belonging to a participant that also declared completion is not
        # work left behind — it is that participant agreeing the task is done.
        # Without this the detector fires whenever anyone else speaks in the
        # final round, which is every real run, and healing would block every
        # legitimate termination rather than only the premature ones.
        finished = {m.sender for m in graph.messages.values() if m.terminal}
        pending = [m for m in reachable_work(graph) if m.sender not in finished]
        if not pending:
            return []

        summary = ", ".join(f"{m.sender}: {_clip(m.content)}" for m in pending[:5])
        return [
            self.finding(
                ctx,
                f"{terminal.sender} submitted while {len(pending)} message(s) "
                f"remain unconsumed ({summary}).",
                Intervention(
                    kind="inject",
                    message=terminal.id,
                    content=f"Unresolved work remains: {summary}",
                    reason="early termination",
                ),
                Intervention(
                    kind="reroute",
                    message=terminal.id,
                    recipients=(terminal.sender,),
                    reason="early termination",
                ),
            )
        ]


class MissingCompletion(Detector):
    """MC — all reachable work is consumed but nobody has submitted.

    Healing emits the signal the agents are missing: the work is exhausted, so
    somebody has to call it done.
    """

    pattern: str = "MC"

    window: int = Field(
        default=1,
        ge=0,
        description="Rounds of exhausted work to tolerate before firing. The "
        "paper says 'a reasonable time window' and gives no value; this is the "
        "smallest one that cannot fire on the round the work was consumed.",
    )

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        if ctx.round < self.window:
            return []
        if graph.terminal_message() is not None or reachable_work(graph):
            return []
        return [
            self.finding(
                ctx,
                "All reachable work is consumed and no participant has submitted.",
                Intervention(
                    kind="emit",
                    content=(
                        "All outstanding work has been consumed and no answer has been "
                        "submitted. Consolidate what you have and submit."
                    ),
                    recipients=ctx.names,
                    reason="missing completion",
                ),
            )
        ]


class OrphanedEvent(Detector):
    """OE — a message was generated that nothing was ever routed to.

    Only observable because reachability and delivery are recorded separately:
    a matcher that leaves a participant with no out-edges produces work that
    exists and is unreadable, and without this the run simply looks quiet.
    """

    pattern: str = "OE"

    window: int = Field(default=1, ge=0, description="Rounds to wait before firing.")

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        findings: list[Finding] = []
        for message in orphans(graph, round=ctx.round, window=self.window):
            findings.append(
                self.finding(
                    ctx,
                    f"{message.sender}'s message from round {message.round} was "
                    "never routed to anyone.",
                    Intervention(
                        kind="inject",
                        message=message.id,
                        content="Nobody received this. Restate it for whoever needs it.",
                        reason="orphaned event",
                    ),
                    Intervention(
                        kind="reroute",
                        message=message.id,
                        recipients=(message.sender,),
                        reason="orphaned event",
                    ),
                )
            )
        return findings


class Deadlock(Detector):
    """DL — reachable work remains but nobody activated.

    Unreachable under a synchronous barrier, which is why `Ensemble` takes a
    `Scheduler`: with `BufferScheduler`, an activation set can legitimately be
    empty, and this is the detector that notices.
    """

    pattern: str = "DL"

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        if ctx.active or not reachable_work(graph):
            return []
        return [
            self.finding(
                ctx,
                f"{len(reachable_work(graph))} message(s) are outstanding but no "
                f"participant activated in round {ctx.round}.",
                Intervention(
                    kind="emit",
                    content=(
                        "Collaboration has stalled with work outstanding. "
                        "Report your status and act on what you have."
                    ),
                    recipients=ctx.names,
                    reason="deadlock",
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Progress warnings
# ---------------------------------------------------------------------------


class ExcessiveRerouting(Detector):
    """ER — a message keeps being rerouted and never gets consumed.

    Fires on the graph's own record of past interventions, so a supervisor that
    is thrashing detects itself.
    """

    pattern: str = "ER"

    threshold: int = Field(
        default=2,
        ge=1,
        description="Reroutes of one message tolerated before firing. The paper "
        "says 'more than a reasonable number' and gives no value.",
    )

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        consumed = graph.consumed()
        findings: list[Finding] = []
        for message in graph.messages.values():
            if message.id in consumed:
                continue
            count = graph.reroutes(message.id)
            if count <= self.threshold:
                continue
            findings.append(
                self.finding(
                    ctx,
                    f"{message.sender}'s message has been rerouted {count} times "
                    "without being consumed.",
                    Intervention(
                        kind="inject",
                        message=message.id,
                        content=(
                            f"This message has been rerouted {count} times without "
                            "anyone acting on it. Handle it or say why you cannot."
                        ),
                        reason="excessive rerouting",
                    ),
                )
            )
        return findings


class CrossLineageAggregation(Detector):
    """CLA — one activation consumed messages from unrelated lineages.

    Two inputs whose ancestor sets are disjoint came from different
    problem-generating branches, and an agent silently merging them produces an
    answer to a question nobody asked. Healing labels each input with where it
    came from rather than removing it, because the aggregation may well be
    intentional.
    """

    pattern: str = "CLA"

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        findings: list[Finding] = []
        for activation in graph.activations_in(ctx.round):
            inputs = graph.inputs_of(activation.id)
            if len(inputs) < 2:
                continue
            # A message with no parents is the root of its own lineage, and two
            # roots are not evidence of anything: at the start of a run every
            # message is a root, so comparing them would report the whole first
            # exchange as cross-lineage. Only messages that have ancestry can be
            # shown to have *different* ancestry.
            lineages = {
                mid: graph.ancestors(mid)
                for mid in inputs
                if mid in graph.messages
                and graph.messages[mid].parents
                and graph.messages[mid].sender != SUPERVISOR
            }
            unrelated = [
                (a, b)
                for a, b in combinations(sorted(lineages), 2)
                if not (lineages[a] & lineages[b])
            ]
            if not unrelated:
                continue

            senders = sorted(
                {
                    graph.messages[mid].sender
                    for pair in unrelated
                    for mid in pair
                    if mid in graph.messages
                }
            )
            findings.append(
                self.finding(
                    ctx,
                    f"{activation.participant} is aggregating messages from "
                    f"unrelated lineages ({', '.join(senders)}).",
                    *[
                        Intervention(
                            kind="inject",
                            message=mid,
                            content=(
                                "This message comes from a different line of work than "
                                f"the others {activation.participant} received; do not "
                                "assume they share a premise."
                            ),
                            reason="cross-lineage aggregation",
                        )
                        for mid in sorted({m for pair in unrelated for m in pair})
                    ],
                )
            )
        return findings


class RepeatedSubproblem(Detector):
    """RSP — two problem-reducing activations consumed the same upstream message.

    Duplicated work, and the reason `is_problem_reducing` exists: two agents
    *expanding* on the same input is collaboration, two agents *solving* it is
    waste.
    """

    pattern: str = "RSP"

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        reducing = [
            a
            for a in graph.activations_in(ctx.round)
            if is_problem_reducing(graph, a.id)
        ]
        by_input: dict[str, list[str]] = {}
        for activation in reducing:
            for message_id in graph.inputs_of(activation.id):
                # A supervisor broadcast is addressed to everyone by design, so
                # everyone receiving it is not duplicated work. Counting it
                # would make healing manufacture the next round's finding.
                message = graph.messages.get(message_id)
                if message is None or message.sender == SUPERVISOR:
                    continue
                by_input.setdefault(message_id, []).append(activation.participant)

        findings: list[Finding] = []
        for _message_id, workers in sorted(by_input.items()):
            if len(set(workers)) < 2:
                continue
            names = sorted(set(workers))
            findings.append(
                self.finding(
                    ctx,
                    f"{' and '.join(names)} both solved the same upstream message.",
                    *[
                        Intervention(
                            kind="emit",
                            participant=name,
                            content=(
                                f"{' and '.join(names)} worked on the same input this "
                                "round, so part of your result may be duplicated. "
                                "Reconcile before continuing."
                            ),
                            recipients=(name,),
                            reason="repeated subproblem solving",
                        )
                        for name in names
                    ],
                )
            )
        return findings


def dig_detectors(*, window: int = 1, threshold: int = 2) -> list[Detector]:
    """All seven of the paper's detectors, in its own order.

    The two hyperparameters the paper leaves as "a reasonable window" and "a
    reasonable number" are call arguments, so a threshold sweep is a loop rather
    than a rebuilt list.
    """
    return [
        EarlyTermination(),
        MissingCompletion(window=window),
        OrphanedEvent(window=window),
        Deadlock(),
        ExcessiveRerouting(threshold=threshold),
        CrossLineageAggregation(),
        RepeatedSubproblem(),
    ]


class LLMJudge(Detector):
    """The paper's second baseline: a model asked, periodically, what is wrong.

    Included because it is the comparison that matters — structural detection
    is only interesting if it beats asking a model — and because it is the
    honest test of whether `Detector` is a real seam: a judge holds a model, not
    a graph query, and still fits without widening anything.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern: str = "JUDGE"
    llm: Annotated[Any, SkipValidation] = None
    every: int = Field(default=1, ge=1, description="Invoke once every N rounds.")

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        if self.llm is None or ctx.round % self.every:
            return []

        from ant_ai.core.message import Message

        transcript = "\n".join(
            f"[r{m.round}] {m.sender}: {_clip(m.content, 200)}"
            for m in graph.messages.values()
        )
        prompt = (
            "You are auditing a multi-agent collaboration for coordination "
            "failures: premature completion, stalling, duplicated work, ignored "
            f"messages.\n\nTask: {ctx.task}\n\nTranscript:\n{transcript}\n\n"
            "Reply with one short sentence naming the problem, or exactly NONE."
        )
        reply = await self.llm.ainvoke([Message(role="user", content=prompt)])
        verdict = str(getattr(reply, "content", reply) or "").strip()
        if not verdict or verdict.upper().startswith("NONE"):
            return []

        return [
            self.finding(
                ctx,
                verdict,
                Intervention(
                    kind="emit",
                    content=f"Coordination review: {verdict}",
                    recipients=ctx.names,
                    reason="llm judge",
                ),
            )
        ]


class DigToHeal(TopologyStrategy):
    """The paper's strategy: one `Heal` stage, plus the timing it needs.

    `Halt` is deliberately left at the framework default: repairing an early
    termination already un-terminates a premature submit, which is the paper's own
    mechanism for the same problem and a strictly more informative one than
    forbidding most participants from finishing.
    """

    name: ClassVar[str] = "dig"
    citation: ClassVar[str] = "arXiv:2603.00309"

    window: int = Field(
        default=1,
        ge=0,
        description="Rounds a symptom must persist before it counts. The paper "
        "specifies 'a reasonable time window' without giving a value.",
    )
    threshold: int = Field(
        default=2,
        ge=1,
        description="Reroutes of one message tolerated before Excessive Rerouting fires.",
    )

    def build(self) -> Pipeline:
        return Pipeline(
            stages=[
                Heal(
                    detectors=dig_detectors(
                        window=self.window, threshold=self.threshold
                    )
                )
            ],
            scheduler=BufferScheduler(),
            materialiser=DeliveryMaterialiser(),
        )


class JudgeHealing(TopologyStrategy):
    """The paper's second baseline: a periodically invoked LLM judge.

    The comparison that matters — structural detection is only interesting if it
    beats asking a model — and the honest test of whether `Detector` is a real
    seam, since a judge holds a model rather than a graph query and still needs
    nothing widened to fit.
    """

    name: ClassVar[str] = "judge"
    citation: ClassVar[str] = "arXiv:2603.00309 (baseline)"

    llm: Annotated[Any, SkipValidation] = None
    every: int = Field(default=2, ge=1, description="Invoke once every N rounds.")

    def build(self) -> Pipeline:
        return Pipeline(
            stages=[Heal(detectors=[LLMJudge(llm=self.llm, every=self.every)])],
            scheduler=BufferScheduler(),
            materialiser=DeliveryMaterialiser(),
        )


def _clip(text: str, limit: int = 80) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
