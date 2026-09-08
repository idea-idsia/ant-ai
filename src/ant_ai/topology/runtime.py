from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.core.events import (
    CompletedEvent,
    Event,
    EventOrigin,
    HealingEvent,
    StartEvent,
    TopologyEvent,
)
from ant_ai.core.types import InvocationContext
from ant_ai.observer import obs
from ant_ai.topology.activate import record_support
from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.heal import Heal
from ant_ai.topology.log import RewriteLog
from ant_ai.topology.participant import Brief, Envelope, Participant, Turn
from ant_ai.topology.plan import Finding, RoundPlan, RunContext
from ant_ai.topology.rewrite import Rewrite
from ant_ai.topology.state import StateGraph
from ant_ai.topology.strategy import Pipeline

_SENTINEL = object()


class Ensemble(BaseModel):
    """Runs a multi-agent task, rewiring who can reach whom between rounds.

    One loop order serves every strategy, because the pipeline always configures
    the *next* round:

    1. **Act** — the scheduler names who activates; they run concurrently with
       whatever tools or inbox were bound at the end of the previous round.
    2. **Record** — attribute message lineage and mark what each turn consumed.
    3. **Plan** — run the pipeline's stages in order over a `RoundPlan`. A
       matching stage writes links; a repair stage rewrites messages and may add
       links of its own. There is nothing to reconcile afterwards because both
       edit the same value.
    4. **Halt?** — asked *after* the stages, since repairing an early termination
       un-terminates a submit and must be able to keep the run alive.
    5. **Materialise** — rebind peer tools and/or fill inboxes for round t+1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    participants: dict[str, Annotated[Participant, SkipValidation]]
    pipeline: Pipeline = Field(default_factory=Pipeline)
    seed: tuple[Link, ...] = Field(
        default=(),
        description="Topology for round 0, normally the colony's declared collab "
        "edges, so the very first turn behaves exactly as a colony does today.",
    )
    graph: InteractionGraph = Field(default_factory=InteractionGraph)
    state: StateGraph = Field(
        default_factory=StateGraph,
        description="The typed agent-state graph every rewrite is applied to. "
        "Passed in to carry memories, tools and skills that outlive this run; "
        "left at its default it starts with the participants and nothing else.",
    )
    log: RewriteLog = Field(
        default_factory=RewriteLog,
        description="Every edit this run made, in order, with its inverse.",
    )
    record_reachability: bool = Field(
        default=True,
        description="Record each round's bound peers as a support subgraph. On "
        "by default because it is what makes reachability measurable rather than "
        "merely decided; off for a long run where one entry per participant per "
        "round is more record than the question being asked needs.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Which strategy and hyperparameters produced this run.",
    )
    halt_reason: str = Field(
        default="",
        description="Why the run stopped. Empty until it has.",
    )

    async def stream(
        self, task: str, *, ctx: InvocationContext | None = None
    ) -> AsyncIterator[Event]:
        inboxes: dict[str, tuple[Envelope, ...]] = dict.fromkeys(self.participants, ())
        final = ""

        with obs.bind(session_id=ctx.session_id if ctx else ""):
            await self._announce(task, ctx)
            yield StartEvent(
                origin=EventOrigin(layer="workflow", run_step=0),
                content="Ensemble started",
            )
            # The participants have to be nodes before any edge between them can
            # be checked against the schema, and the seed has to be in the graph
            # before the first round's diff can be honest about what changed.
            self.state.ensure("agent", *sorted(self.participants))
            await self._seed_round_zero()

            for rnd in range(self.pipeline.max_rounds):
                active = self.pipeline.scheduler.activations(
                    round=rnd, participants=self.participants, inboxes=inboxes
                )
                await obs.event(
                    "topology.round.start",
                    node=f"round {rnd}",
                    run_step=rnd,
                    round=rnd,
                    active=sorted(active),
                )

                outcome = RoundOutcome()
                async for item in self._run_round(
                    rnd, active, inboxes, task, ctx, outcome
                ):
                    yield item

                run_ctx = RunContext(
                    round=rnd,
                    task=task,
                    participants=tuple(p.profile for p in self.participants.values()),
                    active=active,
                    graph=self.graph,
                    state=self.state,
                )
                plan = await self._plan(rnd, outcome.turns, run_ctx)
                # Before the halt check, not after: a repair applied in the final
                # round is an edit that happened, and a log that drops it makes
                # the run that was corrected and the run that ended cleanly look
                # the same afterwards.
                self._evolve(plan)

                for finding in plan.findings:
                    yield HealingEvent(
                        origin=EventOrigin(layer="workflow", run_step=rnd),
                        content=finding.explanation,
                        round=rnd,
                        pattern=finding.pattern,
                        detector=finding.detector,
                        interventions=tuple(i.kind for i in finding.interventions),
                    )

                final = _final_answer(plan.turns) or final

                # Asked *after* the stages, since repairing an early termination
                # un-terminates a submit and must be able to keep the run alive.
                stop = self.pipeline.halt.halt(plan, run_ctx)
                if rnd == self.pipeline.max_rounds - 1:
                    stop = stop or "round budget exhausted"
                if stop:
                    self.halt_reason = stop
                    await self._round_end(rnd, stop)
                    break

                self.graph.record_links(plan.links, round=plan.round)
                self._record_reachability(plan)
                yield TopologyEvent(
                    origin=EventOrigin(layer="workflow", run_step=plan.round),
                    content=f"Topology for round {plan.round}",
                    round=plan.round,
                    links=plan.links,
                )
                inboxes = await self._deliver(plan, outcome)
                await self._round_end(rnd, f"{len(plan.links)} links")

            await obs.event("topology.end", output=final)
            yield CompletedEvent(
                origin=EventOrigin(layer="workflow", run_step=self.pipeline.max_rounds),
                content=final or "Ensemble completed",
            )

    async def ainvoke(self, task: str, *, ctx: InvocationContext | None = None) -> str:
        final = ""
        async for event in self.stream(task, ctx=ctx):
            if isinstance(event, CompletedEvent):
                final = event.content
        return final

    @property
    def findings(self) -> list[Finding]:
        """Every structural failure detected across the run."""
        return [
            finding
            for stage in self.pipeline.stages
            if isinstance(stage, Heal)
            for finding in stage.history
        ]

    def report(self) -> RunReport:
        """What this run actually did, in the terms the layer is configured in.

        The layer's failure mode is a run that completes and looks fine: a
        topology that never moved, messages that reached nobody, detectors
        firing every round on defaults. All of that is visible in the graph and
        none of it is visible in the answer string, so reading the answer alone
        is how a broken configuration gets believed. This is the two-line check.
        """
        graph = self.graph
        per_round = {r: len(graph.links(r)) for r in sorted(_rounds_with_links(graph))}
        shapes = {
            frozenset((link.src, link.dst) for link in graph.links(r))
            for r in per_round
        }
        findings: dict[str, int] = {}
        for finding in self.findings:
            findings[finding.pattern] = findings.get(finding.pattern, 0) + 1

        return RunReport(
            strategy=str(self.provenance.get("strategy", "") or "none"),
            rounds=len(graph.rounds()),
            participants=sorted(self.participants),
            halt_reason=self.halt_reason,
            messages=len(graph.messages),
            delivered=len(graph.delivered()),
            consumed=len(graph.consumed()),
            outstanding=len(graph.unsettled()),
            links_per_round=per_round,
            topology_changed=len(shapes) > 1,
            findings=findings,
            unused_visibility=sum(
                len(graph.unused_visibility(r)) for r in graph.rounds()
            ),
            rewrites=self.log.by_op(),
            components=self.log.by_kind(),
            cascades=len(self.log.cascades()),
            cross_component=len(self.log.cross_component()),
            activations=len(self.state.supports),
            rejected=len(self.log.rejected()),
        )

    async def _announce(self, task: str, ctx: InvocationContext | None) -> None:
        # Field names match the workflow lifecycle events so existing sinks
        # (LangfuseSink keys spans on node/run_step) pick these up unchanged.
        await obs.event(
            "topology.start",
            agent_name="ensemble",
            session_id=ctx.session_id if ctx else None,
            input=task,
            max_steps=self.pipeline.max_rounds,
            participants=list(self.participants),
            **self.provenance,
        )

    async def _round_end(self, rnd: int, output: str) -> None:
        await obs.event(
            "topology.round.end",
            node=f"round {rnd}",
            run_step=rnd,
            round=rnd,
            output=output,
        )

    async def _seed_round_zero(self) -> None:
        """Materialise the declared topology, so round 0 behaves as a colony does."""
        if not self.seed:
            return
        await self.pipeline.materialiser.apply(
            RoundPlan(round=0, links=self.seed), self.participants
        )
        self.graph.record_links(self.seed, round=0)
        self.log.extend(
            (
                Rewrite.link(
                    link.src,
                    link.dst,
                    weight=link.weight,
                    reason=link.reason or "declared topology",
                ).caused_by("seed", at=0)
                for link in self.seed
            ),
            self.state,
        )

    def _evolve(self, plan: RoundPlan) -> None:
        """Apply the round's edits to the persistent graph, and log them.

        The one place a `Rewrite` becomes a fact. Stages decide, this applies —
        which is what keeps a stage replayable against a recorded trace with
        nothing running, and what makes an edit the schema rejects a logged
        finding about the strategy rather than a crash in the middle of a run.
        """
        if not plan.rewrites:
            return
        self.log.extend(plan.rewrites, self.state)

    def _record_reachability(self, plan: RoundPlan) -> None:
        """Record what each participant may reach as a read-only activation.

        Binding peers *is* subgraph activation — a query (this participant, this
        round) selecting a subset of the agent graph that the next decision is
        made from — and writing it down is the difference between a topology
        that was decided and one that can be scored. It is also the only place
        `Activate` appears in a run with no memory in it, which is why it is on
        by default.
        """
        if not self.record_reachability:
            return
        for name in sorted(self.participants):
            record_support(
                self.state,
                kind="agent",
                id=f"reach:{plan.round}:{name}",
                nodes=plan.sources_for(name),
                at=plan.round,
                query=name,
                reason="peers bound for the round",
                log=self.log,
            )

    async def _deliver(
        self, plan: RoundPlan, outcome: RoundOutcome
    ) -> dict[str, tuple[Envelope, ...]]:
        """Phase 5: the inboxes the next round starts with.

        What a turn declined to settle is still in front of it, and what one
        handed on is in front of somebody else. Both survive the round boundary,
        which is what makes `wait` a decision rather than a way to lose a message.
        """
        delivered = await self.pipeline.materialiser.apply(plan, self.participants)
        return {
            name: _merge(outcome.carried.get(name, ()), delivered.get(name, ()))
            for name in self.participants
        }

    async def _run_round(
        self,
        rnd: int,
        active: frozenset[str],
        inboxes: Mapping[str, tuple[Envelope, ...]],
        task: str,
        ctx: InvocationContext | None,
        outcome: RoundOutcome,
    ) -> AsyncIterator[Event]:
        """Phase 1: run the active participants concurrently, streaming their events.

        What the round produced is written to *outcome* rather than returned,
        because this is an async generator: the caller consumes the participants'
        events as they happen and reads the turns once the barrier is passed.
        """
        events: asyncio.Queue[Any] = asyncio.Queue()

        async def take_turn(name: str) -> None:
            brief = Brief(round=rnd, task=task, inbox=inboxes.get(name, ()))
            turn = await self._act(name, brief, ctx=ctx, events=events)
            outcome.record(name, brief.inbox, turn, participants=self.participants)

        async for event in _live([take_turn(name) for name in sorted(active)], events):
            yield event

    async def _act(
        self,
        name: str,
        brief: Brief,
        *,
        ctx: InvocationContext | None,
        events: asyncio.Queue[Any],
    ) -> Turn:
        """One participant's turn, recorded in the graph. Never raises.

        A participant that raises is recorded as a failed activation and the round
        continues — letting one agent's exception kill the ensemble would also
        throw away structural signal a detector wants to see.
        """
        activation = self.graph.record_activation(name, round=brief.round)
        declared: Turn | None = None
        error: str | None = None
        try:
            with obs.bind(agent_name=name):
                async for item in self.participants[name].act(brief, ctx=ctx):
                    if isinstance(item, Turn):
                        declared = item
                    else:
                        await events.put(item)
        except Exception as exc:
            error = str(exc)
            await obs.exception("topology.participant.error", exc, participant=name)

        self.graph.end_activation(activation, error=error)
        # Normalised centrally: without attributed lineage `Envelope.parents`
        # stays empty and the two lineage detectors cannot run at all, and
        # without a derived terminal flag `e_inf` never appears in the graph.
        # Lineage is what the turn *consumed*: a message it left waiting is not
        # an ancestor of anything yet.
        turn = declared or Turn(participant=name, error=error)
        turn = turn.recorded(
            tuple(e.id for e in brief.inbox if turn.reaction_for(e.id) == "consume")
        )
        self._record(name, activation, brief, turn)
        return turn

    def _record(self, name: str, activation: str, brief: Brief, turn: Turn) -> None:
        """Write one finished turn into the interaction graph."""
        for envelope in brief.inbox:
            # Labelled with what the participant said it did, defaulting to
            # `consume`. A delivery edge with no action would leave every message
            # looking outstanding forever and any detector reading that
            # permanently on; a delivery edge that says `consume` when the agent
            # said `wait` is the same lie in the other direction.
            self.graph.record_delivery(
                envelope.id,
                activation,
                round=brief.round,
                action=turn.reaction_for(envelope.id),
            )
        for envelope in turn.outputs:
            self.graph.record_message(envelope, activation_id=activation)
            # Mirrored into the state graph as a node, not through the log: a
            # message an agent produced is not an edit a strategy made, and
            # logging it would bury the strategy's own edits under one entry per
            # message. The node has to exist all the same — a repair that
            # rewrites or drops a message is a `feature_update` or `delete` on
            # it, and against a graph that has never heard of the message both
            # are silent no-ops that still count as edits in the report.
            self.state.apply(
                Rewrite.insert(
                    "message",
                    envelope.id,
                    label=envelope.content[:80],
                    content=envelope.content,
                    sender=envelope.sender,
                    visibility=envelope.visibility,
                ).model_copy(update={"at": envelope.round})
            )
        for callee in turn.invoked:
            self.graph.record_invocation(name, callee, round=brief.round)

    async def _plan(
        self, rnd: int, turns: dict[str, Turn], ctx: RunContext
    ) -> RoundPlan:
        """Phase 3: run every stage in order over one plan.

        A plan governs the *next* round: it is what the participants will act
        under, so it is numbered and recorded against that round rather than the
        one that produced it.
        """
        # Seeded with the declared edges rather than empty. A stage that writes
        # links overwrites them; one that does not — a pure repair strategy —
        # leaves the colony's own `collab()` topology standing, which is what
        # makes `DigToHeal` alone route anything at all.
        plan = RoundPlan(
            round=rnd + 1, turns=turns, links=self.seed, base_links=self.seed
        )
        for stage in self.pipeline.stages:
            plan = await stage.apply(plan, ctx)
        await obs.event("topology.match", round=plan.round, links=len(plan.links))
        return plan


class RunReport(BaseModel):
    """A run summarised in the terms its configuration was written in.

    Every field is here because its absence hid a real failure: a topology that
    never changed, messages that reached nobody, a detector firing on every
    round, reachability granted to peers nobody called.
    """

    strategy: str
    rounds: int
    participants: list[str]
    halt_reason: str = ""
    messages: int = 0
    delivered: int = 0
    consumed: int = 0
    outstanding: int = 0
    links_per_round: dict[int, int] = Field(default_factory=dict)
    topology_changed: bool = False
    """False with a routing strategy configured means the matcher decided the
    same graph every round — usually descriptors that never varied."""
    findings: dict[str, int] = Field(default_factory=dict)
    unused_visibility: int = 0
    """Reachability granted that nobody called. Persistently high means the
    matcher is wiring peers the agents have no use for."""
    rewrites: dict[str, int] = Field(default_factory=dict)
    """Applied edits by operator. Empty with a strategy configured means every
    stage decided the same graph it was given — the same failure
    `topology_changed` reports, seen from the other side."""
    components: dict[str, int] = Field(default_factory=dict)
    """Applied edits by component type. A run that only ever shows `agent` here
    evolved its wiring and nothing else, which is worth knowing when the strategy
    claimed otherwise."""
    cascades: int = 0
    cross_component: int = 0
    """Cascades that touched two or more component types. Zero with a
    co-evolution strategy configured means the coupling never fired."""
    activations: int = 0
    """Read-only subgraph selections recorded — retrievals, bound peers."""
    rejected: int = 0
    """Edits the schema refused. Never zero by accident: a strategy emitting
    them is emitting edits that did nothing, every round, in silence."""

    def render(self) -> str:
        lines = [
            f"strategy       : {self.strategy}",
            f"rounds         : {self.rounds}  ({self.halt_reason or 'still running'})",
            f"participants   : {', '.join(self.participants)}",
            f"messages       : {self.messages} generated, {self.delivered} delivered, "
            f"{self.consumed} consumed, {self.outstanding} outstanding",
            "links/round    : "
            + (
                ", ".join(f"r{r}={n}" for r, n in self.links_per_round.items())
                or "none decided"
            ),
        ]
        if self.links_per_round and not self.topology_changed:
            lines.append("  ! the topology never changed — check that descriptors vary")
        if self.unused_visibility:
            lines.append(
                f"unused links   : {self.unused_visibility} granted, never called"
            )
        if self.rewrites:
            detail = ", ".join(f"{k}x{v}" for k, v in sorted(self.rewrites.items()))
            lines.append(f"rewrites       : {detail}")
        if self.components:
            detail = ", ".join(f"{k}x{v}" for k, v in sorted(self.components.items()))
            lines.append(f"components     : {detail}")
        if self.cascades:
            lines.append(
                f"cascades       : {self.cascades} "
                f"({self.cross_component} cross-component)"
            )
        if self.activations:
            lines.append(f"activations    : {self.activations} recorded")
        if self.rejected:
            lines.append(
                f"  ! {self.rejected} edit(s) were refused by the schema and did "
                "nothing — check what the strategy is emitting"
            )
        if self.findings:
            detail = ", ".join(f"{k}x{v}" for k, v in sorted(self.findings.items()))
            lines.append(f"findings       : {detail}")
            if max(self.findings.values()) >= self.rounds and self.rounds > 1:
                lines.append(
                    "  ! a detector fired every round — likely reading defaults, "
                    "not the collaboration"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _rounds_with_links(graph: InteractionGraph) -> set[int]:
    return {e.round for e in graph.edges if e.kind == "visible"}


@dataclass(slots=True)
class RoundOutcome:
    """What one round produced, collected as its participants finish.

    `carried` is the round boundary's whole job: what a turn declined to settle
    is still in front of it, and what it handed on is in front of somebody else.
    Both survive into the next round, which is what makes `wait` a decision
    rather than a way to lose a message.
    """

    turns: dict[str, Turn] = field(default_factory=dict)
    carried: dict[str, list[Envelope]] = field(default_factory=dict)

    def record(
        self,
        name: str,
        inbox: tuple[Envelope, ...],
        turn: Turn,
        *,
        participants: Mapping[str, Participant],
    ) -> None:
        self.turns[name] = turn
        self._hold(name, [e for e in inbox if turn.reaction_for(e.id) == "wait"])
        for handed, target in _handovers(inbox, turn):
            if target in participants and target != name:
                self._hold(target, [handed])

    def _hold(self, name: str, envelopes: list[Envelope]) -> None:
        if envelopes:
            self.carried.setdefault(name, []).extend(envelopes)


def _handovers(
    inbox: tuple[Envelope, ...], turn: Turn
) -> Iterator[tuple[Envelope, str]]:
    """The messages a turn handed on, paired with whom it handed each to.

    A reroute names an envelope this turn was actually delivered; one naming
    anything else is ignored rather than invented.
    """
    by_id = {e.id: e for e in inbox}
    for message_id, targets in turn.rerouted.items():
        handed = by_id.get(message_id)
        if handed is not None:
            yield from ((handed, target) for target in targets)


async def _live(
    coroutines: Iterable[Coroutine[Any, Any, None]], events: asyncio.Queue[Any]
) -> AsyncIterator[Any]:
    """Run *coroutines* concurrently, yielding what they put on *events* as it lands.

    A queue rather than `gather` then drain, so the caller sees a participant's
    events live rather than after the barrier. Exceptions are collected rather
    than propagated: the coroutines here already record their own failures, and a
    raise would leave the queue with no sentinel and the caller waiting forever.
    """
    tasks = [asyncio.create_task(coro) for coro in coroutines]

    async def barrier() -> None:
        await asyncio.gather(*tasks, return_exceptions=True)
        await events.put(_SENTINEL)

    pumping = asyncio.create_task(barrier())
    try:
        while (item := await events.get()) is not _SENTINEL:
            yield item
    finally:
        await pumping


def _merge(*groups: Sequence[Envelope]) -> tuple[Envelope, ...]:
    """Concatenate message groups, keeping the first copy of each.

    A message can arrive twice — waited on by its holder and rerouted to it by
    a repair in the same round — and an inbox with two of the same envelope
    would make the record double-count a delivery that happened once.
    """
    seen: set[str] = set()
    merged: list[Envelope] = []
    for group in groups:
        for envelope in group:
            if envelope.id not in seen:
                seen.add(envelope.id)
                merged.append(envelope)
    return tuple(merged)


def _final_answer(turns: dict[str, Turn]) -> str:
    """The round's answer, chosen deterministically.

    `turns` is filled by concurrent coroutines, so its iteration order is
    completion order — taking "the last public message" would make the result
    depend on which agent happened to finish first, and would silently discard the
    rest.

    A participant that declares `submitted` owns the answer. Otherwise every public
    message is returned, attributed and in stable name order, so nothing is
    dropped.
    """
    parts: list[str] = []
    for name in sorted(turns):
        public = turns[name].public
        if public is None or not public.content:
            continue
        if turns[name].submitted:
            return public.content
        parts.append(f"[{name}] {public.content}")
    return "\n\n".join(parts)
