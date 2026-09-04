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
from ant_ai.topology.graph import InteractionGraph, Link
from ant_ai.topology.heal import Heal
from ant_ai.topology.participant import Brief, Envelope, Participant, Turn
from ant_ai.topology.plan import Finding, RoundPlan, RunContext
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
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Which strategy and hyperparameters produced this run.",
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
                )
                plan = await self._plan(rnd, outcome.turns, run_ctx)

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
                    await self._round_end(rnd, stop)
                    break

                self.graph.record_links(plan.links, round=plan.round)
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
        plan = RoundPlan(round=rnd + 1, turns=turns)
        for stage in self.pipeline.stages:
            plan = await stage.apply(plan, ctx)
        await obs.event("topology.match", round=plan.round, links=len(plan.links))
        return plan


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
