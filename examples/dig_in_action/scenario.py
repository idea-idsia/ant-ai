"""The paper's own demo, made runnable: agents counting integer frequencies.

`https://happyeureka.github.io/dig/` animates a run on a task it calls
CountFrequency — one agent splits a 10,000-element array across five others,
they count their slice, and the first agent aggregates. This is that task, with
one deliberate flaw: the coordinator treats four results out of five as good
enough. That is precisely the coordination failure `EarlyTermination` exists to
catch, so the same scenario run with and without `DigToHeal` ends with a
different *answer*, not merely a different diagram.

It runs the way the reference implementation does, with **no routing stage at
all**. Every message names its own recipients, so a round nothing has decided
reachability for delivers each message where its sender addressed it. The
coordinator therefore sends five different chunks to five agents rather than one
broadcast five agents read, the coordinator *waits* on partials instead of
consuming them as they trickle in, and the auditor's opening message —
addressed to nobody, because it does not yet know who is coordinating — is a
real orphan for `OrphanedEvent` to find and hand back.

Participants are scripted rather than model-driven on purpose. What the example
is showing is the graph, and a scripted run is deterministic, needs no API key,
and repeats the same choreography, so the healed and unhealed runs differ only
in the healing. Everything else is real: these implement the `Participant`
protocol and go through the real `Ensemble`, so the `InteractionGraph` being
drawn is the framework's own record, not a mock's.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ant_ai.core.events import Event, EventOrigin, UpdateEvent
from ant_ai.core.types import InvocationContext
from ant_ai.tools.tool import Tool
from ant_ai.topology.builtins import DigToHeal, dig_detectors
from ant_ai.topology.heal import Heal
from ant_ai.topology.participant import (
    Brief,
    Envelope,
    Participant,
    ParticipantProfile,
    Turn,
)
from ant_ai.topology.plan import SUPERVISOR
from ant_ai.topology.runtime import Ensemble
from ant_ai.topology.strategy import Halt

__all__ = ["Scenario", "build_ensemble", "build_scenario"]

ARRAY_SIZE = 10_000
VALUE_RANGE = 20
COUNTERS = ("counter_a", "counter_b", "counter_c", "counter_d", "counter_e")
COORDINATOR = "coordinator"
AUDITOR = "auditor"

TASK = (
    f"Count how often each integer 0-{VALUE_RANGE - 1} occurs in an "
    f"{ARRAY_SIZE:,}-element array. Split the array across the counting agents, "
    "then aggregate their partial counts into one answer."
)


def dataset(seed: int = 7) -> list[int]:
    """The array being counted. Seeded, so every run counts the same numbers."""
    rng = random.Random(seed)
    return [rng.randrange(VALUE_RANGE) for _ in range(ARRAY_SIZE)]


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


class ScriptedParticipant:
    """A deterministic stand-in for an agent, speaking the real protocol.

    `think` is not decoration: the diagram's x axis is wall-clock seconds, so a
    turn that returns instantly would collapse the whole run onto one tick and
    there would be nothing to watch.
    """

    def __init__(self, name: str, description: str, *, think: float = 1.2) -> None:
        self._name = name
        self._description = description
        self.think = think
        # Seeded on the name: every agent keeps its own tempo, and the same
        # cast paces the same way twice, so two runs stay comparable.
        self._pace = random.Random(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> ParticipantProfile:
        return ParticipantProfile(name=self._name, description=self._description)

    def as_tool(self) -> Tool:
        # Only `VisibilityMaterialiser` binds participants to each other as
        # tools. DIG runs under delivery, where messages carry their own
        # addressing, so nothing in this example ever asks for one.
        raise NotImplementedError(f"{self._name} is a delivery-mode participant.")

    async def bind_peers(self, peers: Mapping[str, Participant]) -> bool:
        return False

    async def act(
        self, brief: Brief, *, ctx: InvocationContext | None = None
    ) -> AsyncIterator[Event | Turn]:
        yield UpdateEvent(
            origin=EventOrigin(layer="workflow", run_step=brief.round),
            content=f"{self._name} activated with {len(brief.inbox)} message(s)",
        )
        await asyncio.sleep(self.think * self._pace.uniform(0.7, 1.8) * self.effort)
        yield self.respond(brief)

    @property
    def effort(self) -> float:
        """Multiplier for the turn about to be taken. One turn can cost more."""
        return 1.0

    def respond(self, brief: Brief) -> Turn:
        raise NotImplementedError

    # -- turn helpers -----------------------------------------------------

    def message(
        self,
        content: str,
        *,
        round: int,
        to: tuple[str, ...] = (),
        visibility: str = "private",
        terminal: bool = False,
    ) -> Envelope:
        return Envelope(
            sender=self._name,
            content=content,
            visibility="public" if visibility == "public" else "private",
            round=round,
            recipients=to,
        )

    def turn(
        self,
        *outputs: Envelope,
        brief: Brief | None = None,
        reaction: str = "consume",
        query: str = "",
        key: str = "",
        submitted: bool = False,
    ) -> Turn:
        """One turn, saying what it produced and what it did with its inbox."""
        reactions = (
            {e.id: reaction for e in brief.inbox}
            if brief is not None and reaction != "consume"
            else {}
        )
        return Turn(
            participant=self._name,
            outputs=outputs,
            reactions=reactions,
            query=query,
            key=key,
            submitted=submitted,
        )

    def silent(self, brief: Brief | None = None, *, reaction: str = "consume") -> Turn:
        """Activate and produce nothing: DIG's *wait*, at the turn level."""
        return self.turn(brief=brief, reaction=reaction)


class Coordinator(ScriptedParticipant):
    """Splits the array, then aggregates what comes back.

    It waits on partials rather than consuming them one at a time, which is what
    the reference implementation's aggregator does and what makes the buffer
    mean something: an unconsumed partial is outstanding work, visible to every
    detector, right up until the turn that settles it.

    The flaw is `patience`: with four of five chunks in hand it decides that is
    enough and submits. Nothing about that is malicious or even unusual — it is
    what an impatient prompt produces — and it is invisible in a transcript,
    because the answer it writes looks finished.
    """

    def __init__(
        self,
        *,
        counters: tuple[str, ...],
        chunks: dict[str, tuple[int, int]],
        patience: int,
        think: float,
    ) -> None:
        super().__init__(
            COORDINATOR,
            "Splits the array into chunks, assigns them and aggregates the counts.",
            think=think,
        )
        self.counters = counters
        self.chunks = chunks
        self.patience = patience
        self.partials: dict[str, Counter[int]] = {}
        self.aggregate: Counter[int] = Counter()
        self.assigned = False
        self.gave_up = False

    def respond(self, brief: Brief) -> Turn:
        nudged = False
        for envelope in brief.inbox:
            if envelope.sender == SUPERVISOR:
                # A supervisor notice — Missing Completion asking for an answer,
                # or a repaired submit handed back by Early Termination.
                nudged = True
                continue
            parsed = parse_result(envelope.content)
            if parsed is not None:
                self.partials[parsed[0]] = parsed[1]

        if not self.assigned:
            self.assigned = True
            return self.turn(
                self.message(
                    f"Splitting {ARRAY_SIZE} values across {len(self.counters)} "
                    "agents, one chunk each.",
                    round=brief.round,
                    visibility="public",
                ),
                *(
                    self.message(
                        f"ASSIGNMENT {name} {start}-{end} — count values "
                        f"0-{VALUE_RANGE - 1} in indices {start}..{end - 1} and "
                        'reply RESULT <name> <start>-<end> {"value": count, ...}.',
                        round=brief.round,
                        to=(name,),
                    )
                    for name, (start, end) in self.chunks.items()
                ),
                brief=brief,
                key="the chunk assignment and the final aggregate",
                query="partial counts for every chunk",
            )

        missing = tuple(c for c in self.counters if c not in self.partials)
        if not missing:
            return self.submit(brief, missing=())

        impatient = nudged or len(self.partials) >= self.patience
        if self.partials and impatient and not self.gave_up:
            self.gave_up = True
            return self.submit(brief, missing=missing)

        # Nothing to settle yet, so nothing is consumed: the partials stay in
        # the buffer and stay outstanding, which is the honest state of a run
        # that is waiting for someone.
        return self.silent(brief, reaction="wait")

    def submit(self, brief: Brief, *, missing: tuple[str, ...]) -> Turn:
        self.aggregate = Counter()
        for counts in self.partials.values():
            self.aggregate.update(counts)
        total = sum(self.aggregate.values())
        top = ", ".join(f"{v} ({n}x)" for v, n in self.aggregate.most_common(3))
        note = f" Never heard from: {', '.join(missing)}." if missing else ""

        outputs = [
            self.message(
                f"FINAL — counted {total} of {ARRAY_SIZE} values from "
                f"{len(self.partials)} of {len(self.counters)} chunks. "
                f"Most frequent: {top}.{note}",
                round=brief.round,
                to=(AUDITOR,),
                visibility="public",
            )
        ]
        if missing:
            # "I will go with what I have; send yours if you get it." The nudge
            # is what makes the late result exist to be caught in flight.
            outputs.append(
                self.message(
                    f"STATUS — submitting with {len(self.partials)} of "
                    f"{len(self.counters)} chunks. Send yours as soon as you have it.",
                    round=brief.round,
                    to=missing,
                )
            )
        return self.turn(
            *outputs, brief=brief, submitted=True, key="the aggregated frequency table"
        )


class ChunkCounter(ScriptedParticipant):
    """Counts the slice it was assigned, once, and then holds its peace.

    `slow` gives one of them a chunk that takes a second round. That is all it
    takes to lose a fifth of the answer: the coordinator's deadline arrives
    first.
    """

    def __init__(self, name: str, *, data: list[int], think: float, slow: bool) -> None:
        super().__init__(
            name, "Counts integer frequencies in the slice it is assigned.", think=think
        )
        self.data = data
        self.slow = slow
        self.window: tuple[int, int] | None = None
        self.warmed = False
        self.done = False

    @property
    def effort(self) -> float:
        # The chunk that is too big to finish in one round is the whole flaw;
        # it should cost visibly more on the Gantt than everyone else's.
        return 2.4 if self.slow and self.warmed and not self.done else 1.0

    def respond(self, brief: Brief) -> Turn:
        for envelope in brief.inbox:
            window = parse_assignment(envelope.content, self._name)
            if window is not None:
                self.window = window

        if self.window is None or self.done:
            return self.silent(brief)

        if self.slow and not self.warmed:
            self.warmed = True
            return self.turn(
                self.message(
                    f"WORKING — {self._name} has started on indices "
                    f"{self.window[0]}..{self.window[1] - 1}; one more round needed.",
                    round=brief.round,
                    to=(COORDINATOR,),
                ),
                brief=brief,
                key="a partial count, next round",
            )

        start, end = self.window
        counts = Counter(self.data[start:end])
        self.done = True
        payload = json.dumps({str(k): counts[k] for k in sorted(counts)})
        return self.turn(
            self.message(
                f"RESULT {self._name} {start}-{end} {payload}",
                round=brief.round,
                to=(COORDINATOR,),
            ),
            brief=brief,
            key=f"counts for indices {start}..{end - 1}",
        )


class Auditor(ScriptedParticipant):
    """A second pass over the array — if anyone ever talks to it.

    Its opening message is addressed to nobody, because at round 0 it has not
    been told who is coordinating. That is a generated event nothing was routed
    to, which is invisible in a transcript — the run just looks quiet — and is
    exactly what `OrphanedEvent` is a query for. Healing hands the message back
    to it with a note, and it restates itself to the coordinator: the correction
    the paper prescribes, doing the thing it was meant to do.
    """

    def __init__(self, *, data: list[int], think: float) -> None:
        super().__init__(
            AUDITOR,
            "Verifies an aggregate against an independent second pass.",
            think=think,
        )
        self.data = data
        self.announced = False
        self.audited = False

    def respond(self, brief: Brief) -> Turn:
        claim = next((e for e in brief.inbox if e.content.startswith("FINAL")), None)
        if claim is not None and not self.audited:
            self.audited = True
            counted = parse_total(claim.content)
            missing = len(self.data) - (counted or 0)
            verdict = (
                "matches my own pass over the array"
                if missing == 0
                else f"is short {missing} of {len(self.data)} values"
            )
            return self.turn(
                self.message(
                    f"AUDIT — the aggregate {verdict}.",
                    round=brief.round,
                    to=(COORDINATOR,),
                ),
                brief=brief,
                key="an independent verification of the aggregate",
            )

        returned = any(e.sender == self._name for e in brief.inbox)
        if returned:
            return self.turn(
                self.message(
                    "READY — restating for the coordinator: send me the aggregate "
                    "and I will check it against a second pass.",
                    round=brief.round,
                    to=(COORDINATOR,),
                ),
                brief=brief,
                key="independent verification of an aggregate",
            )

        if self.announced:
            return self.silent(brief)
        self.announced = True
        return self.turn(
            self.message(
                "READY — send me an aggregate and I will check it against a "
                "second pass over the array.",
                round=brief.round,
            ),
            brief=brief,
            query="an aggregate to verify",
            key="independent verification of an aggregate",
        )


# ---------------------------------------------------------------------------
# Parsing — the participants read each other's messages, nothing is passed
# out of band, so an intervention that rewrites a message is felt.
# ---------------------------------------------------------------------------


def parse_assignment(content: str, name: str) -> tuple[int, int] | None:
    if not content.startswith(f"ASSIGNMENT {name} "):
        return None
    span = content.split(" ", 3)[2]
    start, _, end = span.partition("-")
    try:
        return int(start), int(end)
    except ValueError:
        return None


def parse_total(content: str) -> int | None:
    """The count a `FINAL` message claims, so the audit is over the real number."""
    parts = content.replace(",", "").split()
    for i, word in enumerate(parts):
        if word == "counted" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def parse_result(content: str) -> tuple[str, Counter[int]] | None:
    if not content.startswith("RESULT "):
        return None
    parts = content.split(" ", 3)
    if len(parts) < 4:
        return None
    # A supervisor `inject` appends its note after a blank line; the payload is
    # the first line, and the note is for the reader, not the parser.
    payload = parts[3].split("\n", 1)[0]
    try:
        counts = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parts[1], Counter({int(k): int(v) for k, v in counts.items()})


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One built cast, plus the ground truth to check its answer against."""

    task: str
    participants: dict[str, Any]
    coordinator: Coordinator
    truth: Counter[int]
    order: list[str] = field(default_factory=list)

    def verdict(self) -> dict[str, Any]:
        """Did the run actually count the array?

        The point of the comparison: a premature submit does not look wrong in
        the transcript, it looks finished. It is wrong in the arithmetic.
        """
        counted = sum(self.coordinator.aggregate.values())
        expected = sum(self.truth.values())
        return {
            "answered": bool(self.coordinator.aggregate),
            "counted": counted,
            "expected": expected,
            "correct": self.coordinator.aggregate == self.truth,
            "chunks": len(self.coordinator.partials),
            "chunks_expected": len(self.coordinator.counters),
        }


def build_scenario(
    *, think: float = 1.2, patience: int = 4, seed: int = 7, slow: str = "counter_d"
) -> Scenario:
    """A fresh cast. Never reuse one across runs — participants carry state."""
    data = dataset(seed)
    size = ARRAY_SIZE // len(COUNTERS)
    chunks = {
        name: (i * size, (i + 1) * size if i < len(COUNTERS) - 1 else ARRAY_SIZE)
        for i, name in enumerate(COUNTERS)
    }

    coordinator = Coordinator(
        counters=COUNTERS, chunks=chunks, patience=patience, think=think
    )
    participants: dict[str, Any] = {COORDINATOR: coordinator}
    for name in COUNTERS:
        participants[name] = ChunkCounter(
            name, data=data, think=think, slow=name == slow
        )
    participants[AUDITOR] = Auditor(data=data, think=think)

    return Scenario(
        task=TASK,
        participants=participants,
        coordinator=coordinator,
        truth=Counter(data),
        order=[COORDINATOR, *COUNTERS, AUDITOR],
    )


def build_ensemble(
    scenario: Scenario,
    *,
    heal: bool = True,
    max_rounds: int = 8,
    repeated_subproblem: bool = False,
) -> Ensemble:
    """The same run twice over, healing being the only difference.

    There is no routing stage under the healing, which is the reference
    implementation's arrangement: messages carry their own addressing, so a
    plan with no links has no opinion and delivery follows what the senders
    said. The unhealed condition is then literally the empty pipeline —
    scheduler and materialiser identical, nothing supervising — so the two runs
    differ in one thing.

    `repeated_subproblem` is off by default. RSP asks whether two
    problem-reducing turns consumed the same upstream message; with per-agent
    addressing it has little to fire on here, and its advisory wakes the whole
    cast when it does. `--rsp` puts it back.
    """
    strategy = DigToHeal(max_rounds=max_rounds)
    pipeline = strategy.pipeline()
    detectors = [
        d for d in dig_detectors() if repeated_subproblem or d.pattern != "RSP"
    ]

    return Ensemble(
        participants=scenario.participants,
        pipeline=pipeline.model_copy(
            update={
                "stages": [Heal(detectors=detectors)] if heal else [],
                # Only the coordinator may end the run. Without this a counter
                # declaring its own slice done would end everyone's.
                "halt": Halt(deciders=frozenset({COORDINATOR})),
            }
        ),
        provenance=(
            strategy.provenance()
            if heal
            else {"strategy": "self-addressed, no healing"}
        ),
    )
