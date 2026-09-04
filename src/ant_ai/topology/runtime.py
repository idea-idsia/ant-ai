from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
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
            yield StartEvent(
                origin=EventOrigin(layer="workflow", run_step=0),
                content="Ensemble started",
            )

            if self.seed:
                await self.pipeline.materialiser.apply(
                    RoundPlan(round=0, links=self.seed), self.participants
                )
                self.graph.record_links(self.seed, round=0)

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

                turns: dict[str, Turn] = {}
                carried: dict[str, list[Envelope]] = {}
                async for item in self._run_round(
                    rnd, active, inboxes, task, ctx, turns, carried
                ):
                    yield item

                run_ctx = RunContext(
                    round=rnd,
                    task=task,
                    participants=tuple(p.profile for p in self.participants.values()),
                    active=active,
                    graph=self.graph,
                )
                plan = await self._plan(rnd, turns, run_ctx)

                for finding in plan.findings:
                    yield HealingEvent(
                        origin=EventOrigin(layer="workflow", run_step=rnd),
                        content=finding.explanation,
                        round=rnd,
                        pattern=finding.pattern,
                        detector=finding.detector,
                        interventions=tuple(i.kind for i in finding.interventions),
                    )

                answer = _final_answer(plan.turns)
                if answer:
                    final = answer

                reason = self.pipeline.halt.halt(plan, run_ctx)
                last_round = rnd == self.pipeline.max_rounds - 1
                if reason or last_round:
                    await obs.event(
                        "topology.round.end",
                        node=f"round {rnd}",
                        run_step=rnd,
                        round=rnd,
                        output=reason or "round budget exhausted",
                    )
                    break

                self.graph.record_links(plan.links, round=plan.round)
                yield TopologyEvent(
                    origin=EventOrigin(layer="workflow", run_step=plan.round),
                    content=f"Topology for round {plan.round}",
                    round=plan.round,
                    links=plan.links,
                )

                delivered = await self.pipeline.materialiser.apply(
                    plan, self.participants
                )
                # What a turn declined to settle is still in front of it, and what
                # one handed on is in front of somebody else. Both survive the
                # round boundary, which is what makes `wait` a decision rather
                # than a way to lose a message.
                inboxes = {
                    name: _merge(carried.get(name, ()), delivered.get(name, ()))
                    for name in self.participants
                }

                await obs.event(
                    "topology.round.end",
                    node=f"round {rnd}",
                    run_step=rnd,
                    round=rnd,
                    output=f"{len(plan.links)} links",
                )

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

    async def _run_round(
        self,
        rnd: int,
        active: frozenset[str],
        inboxes: dict[str, tuple[Envelope, ...]],
        task: str,
        ctx: InvocationContext | None,
        turns: dict[str, Turn],
        carried: dict[str, list[Envelope]],
    ) -> AsyncIterator[Event]:
        """Phase 1: run the active participants concurrently, streaming their events.

        Their events are merged through a queue so the caller sees them live rather
        than after the barrier. A participant that raises is recorded as a failed
        activation and the round continues — a bare `gather` would let one agent's
        exception kill the ensemble, and a failed activation is structural signal a
        detector wants to see, not swallow.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def run_one(name: str) -> None:
            participant = self.participants[name]
            inbox = inboxes.get(name, ())
            activation = self.graph.record_activation(name, round=rnd)
            brief = Brief(round=rnd, task=task, inbox=inbox)
            turn: Turn | None = None
            error: str | None = None
            try:
                with obs.bind(agent_name=name):
                    async for item in participant.act(brief, ctx=ctx):
                        if isinstance(item, Turn):
                            turn = item
                        else:
                            await queue.put(item)
            except Exception as exc:
                error = str(exc)
                await obs.exception("topology.participant.error", exc, participant=name)

            self.graph.end_activation(activation, error=error)
            # Normalised centrally: without attributed lineage `Envelope.parents`
            # stays empty and the two lineage detectors cannot run at all, and
            # without a derived terminal flag `e_inf` never appears in the graph.
            # Lineage is what the turn *consumed*: a message it left waiting is
            # not an ancestor of anything yet.
            declared = turn or Turn(participant=name, error=error)
            resolved = declared.recorded(
                tuple(e.id for e in inbox if declared.reaction_for(e.id) == "consume")
            )
            turns[name] = resolved

            for envelope in inbox:
                # Labelled with what the participant said it did, defaulting to
                # `consume`. A delivery edge with no action would leave every
                # message looking outstanding forever and any detector reading
                # that permanently on; a delivery edge that says `consume` when
                # the agent said `wait` is the same lie in the other direction.
                self.graph.record_delivery(
                    envelope.id,
                    activation,
                    round=rnd,
                    action=resolved.reaction_for(envelope.id),
                )
            for envelope in resolved.outputs:
                self.graph.record_message(envelope, activation_id=activation)
            for callee in resolved.invoked:
                self.graph.record_invocation(name, callee, round=rnd)

            carried.setdefault(name, []).extend(
                e for e in inbox if resolved.reaction_for(e.id) == "wait"
            )
            for message_id, targets in resolved.rerouted.items():
                handed = next((e for e in inbox if e.id == message_id), None)
                if handed is None:
                    continue
                for target in targets:
                    if target in self.participants and target != name:
                        carried.setdefault(target, []).append(handed)

        tasks = [asyncio.create_task(run_one(name)) for name in sorted(active)]

        async def pump() -> None:
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(_SENTINEL)

        pumper = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            await pumper

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
