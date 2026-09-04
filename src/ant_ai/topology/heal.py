"""Detecting structural failures and repairing them.

`Detector` says what is wrong; `Heal` is the stage that runs a set of them and
turns their `Intervention`s into concrete edits to the next round's plan. The
split is the point: what varies between repair algorithms is the *detectors*, not
the mechanics of rewriting a message, so a new one is a class with one method and
never a reimplementation of delivery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.observer import obs
from ant_ai.topology.graph import EdgeAction, InteractionGraph
from ant_ai.topology.participant import Envelope, Turn
from ant_ai.topology.plan import (
    SUPERVISOR,
    Finding,
    Intervention,
    RoundPlan,
    RunContext,
)

__all__ = ["Detector", "Heal"]


class Detector(BaseModel):
    """Finds one structural failure pattern. Subclass and implement `detect`.

    A concrete base rather than a protocol so that `finding()` is available to
    whoever writes one — the private base this replaced kept that helper out of
    reach of anything subclassing the public type. `Heal.detectors` skips
    validation, so a duck-typed detector that merely has `pattern` and `detect`
    still works.

    Async because a detector may itself consult a model: the LLM-judge baseline is
    a `Detector` like any other, which is what makes comparing judged against
    structural repair a change of one list element.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern: str = Field(default="", description="Short code, e.g. 'ET' or 'CLA'.")

    async def detect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        raise NotImplementedError

    def finding(
        self, ctx: RunContext, explanation: str, *interventions: Intervention
    ) -> Finding:
        """Build a `Finding` attributed to this detector and this round."""
        return Finding(
            pattern=self.pattern,
            detector=type(self).__name__,
            round=ctx.round,
            explanation=explanation,
            interventions=interventions,
        )


class Heal(BaseModel):
    """The stage that runs detectors and applies what they prescribe.

    Concrete rather than a seam of its own: detectors vary, the repair mechanics
    do not. Placing it in the stage list is what lets a repair algorithm compose
    with a routing one — `Heal` reads the links a matching stage just wrote and
    may add to them, instead of both emitting links to be reconciled afterwards.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    needs_structured_turns: ClassVar[bool] = True
    """Every symptom this stage exists to find is declared by a participant in
    its structured turn: `submitted` is what `EarlyTermination` and
    `MissingCompletion` read, the reactions are what make a message outstanding
    rather than consumed, and `reroute` is what `ExcessiveRerouting` counts.
    Without them a run has no submits, no waits and no reroutes, so every
    detector reports a healthy run and repair is silently inert."""

    detectors: list[Annotated[Detector, SkipValidation]] = Field(default_factory=list)
    history: list[Finding] = Field(
        default_factory=list,
        description="Every finding across the run, for reporting and provenance.",
    )

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        findings = await self.inspect(ctx.graph, ctx)
        if not findings:
            return plan
        return apply_interventions(
            plan,
            ctx,
            [i for finding in findings for i in finding.interventions],
        ).model_copy(update={"findings": (*plan.findings, *findings)})

    async def inspect(self, graph: InteractionGraph, ctx: RunContext) -> list[Finding]:
        """Run every detector. One that raises is skipped, not fatal.

        A detector is diagnostic machinery; letting a bug in one abort the run it
        was meant to protect would be exactly backwards.
        """
        findings: list[Finding] = []
        for detector in self.detectors:
            try:
                findings.extend(await detector.detect(graph, ctx))
            except Exception as exc:  # pragma: no cover - defensive
                await obs.exception(
                    "topology.detector.error", exc, detector=type(detector).__name__
                )
        for finding in findings:
            await obs.event(
                "topology.detect",
                round=ctx.round,
                pattern=finding.pattern,
                detector=finding.detector,
                explanation=finding.explanation,
                interventions=[i.kind for i in finding.interventions],
            )
        self.history.extend(findings)
        return findings


def apply_interventions(
    plan: RoundPlan, ctx: RunContext, interventions: list[Intervention]
) -> RoundPlan:
    """Turn prescriptions into edits to the plan.

    Every branch also writes an `intervenes` edge to the graph, so the record says
    that a run was corrected and how — without which a repaired run and a healthy
    one look identical afterwards, and `ExcessiveRerouting` has nothing to count.

    A rewrite reaches the message wherever it lives. If the message belongs to a
    turn this round, that turn is rewritten too, so what gets delivered is the
    corrected text; if it is older than that, the graph's copy is the only copy
    and is edited in place. The alternative — only being able to repair what was
    said in the last round — makes the corrections for a stalled or ignored
    message unreachable exactly when they are needed.
    """
    edit = _Edit.of(plan, ctx)
    for action in interventions:
        edit.apply(action)
    return edit.into(plan)


@dataclass(frozen=True, slots=True)
class _Site:
    """Where a message sits in this round's turns, when it sits in one at all."""

    participant: str
    index: int


@dataclass(slots=True)
class _Edit:
    """The mutable half of a plan, while interventions are being applied.

    An object rather than three locals threaded through four branches: every
    prescription needs the same trio — the turns to rewrite, the notices to add
    to, and the graph to record against — and giving them one owner is what let
    each `kind` become a three-line method.
    """

    graph: InteractionGraph
    round: int
    names: tuple[str, ...]
    turns: dict[str, Turn]
    notices: dict[str, list[Envelope]] = field(default_factory=dict)

    @classmethod
    def of(cls, plan: RoundPlan, ctx: RunContext) -> _Edit:
        return cls(
            graph=ctx.graph,
            round=ctx.round,
            names=ctx.names,
            turns=dict(plan.turns),
            notices={k: list(v) for k, v in plan.notices.items()},
        )

    def into(self, plan: RoundPlan) -> RoundPlan:
        # `links` is deliberately untouched: repair moves messages, and leaves deciding who may reach whom to the stage whose job that is.
        return plan.model_copy(
            update={
                "turns": self.turns,
                "notices": {k: tuple(v) for k, v in self.notices.items()},
            }
        )

    def apply(self, action: Intervention) -> None:
        """Carry out one prescription, ignoring one that names no live message."""
        if action.kind == "emit":
            self._emit(action)
            return
        site = self._locate(action.message)
        envelope = (
            self.turns[site.participant].outputs[site.index]
            if site
            else self.graph.messages.get(action.message or "")
        )
        if envelope is not None:
            _HANDLERS[action.kind](self, action, envelope, site)

    def _emit(self, action: Intervention) -> None:
        envelope = Envelope(
            sender=SUPERVISOR,
            content=action.content or "",
            visibility="private",
            round=self.round,
        )
        self.graph.record_emission(envelope, reason=action.reason)
        self._notify(action.recipients or self.names, envelope)

    def _inject(
        self, action: Intervention, envelope: Envelope, site: _Site | None
    ) -> None:
        content = f"{envelope.content}\n\n[{action.reason}] {action.content}".strip()
        envelope = self._rewrite(site, envelope, {"content": content})
        self._record(envelope.id, envelope.sender, "inject", action.reason)

    def _drop(
        self, action: Intervention, envelope: Envelope, site: _Site | None
    ) -> None:
        self._rewrite(site, envelope, None)
        self._record(envelope.id, envelope.sender, "discard", action.reason)

    def _reroute(
        self, action: Intervention, envelope: Envelope, site: _Site | None
    ) -> None:
        update: dict[str, Any] = {}
        if envelope.terminal:
            # "Reroute the submit back to the issuing agent" is only a correction
            # if it also stops the run ending: an un-terminated submit is what
            # gives the agent another round to finish.
            update["terminal"] = False
            if site is not None:
                self._update_turn(site.participant, {"submitted": False})
        envelope = self._rewrite(site, envelope, update)
        # Carried as a notice rather than as a link, because a link says
        # "whatever this agent says next goes to that one" — which delivers the
        # wrong message when the agent has moved on, and nothing at all when it
        # has fallen silent, which is the case every stall-shaped repair is
        # trying to fix. The notice moves *this* message, which is what was
        # prescribed.
        self._notify(action.recipients, envelope)
        for recipient in action.recipients:
            self._record(envelope.id, recipient, "reroute", action.reason)

    # -- the shared mechanics ------------------------------------------------

    def _locate(self, message_id: str | None) -> _Site | None:
        for name, turn in self.turns.items():
            for index, envelope in enumerate(turn.outputs):
                if envelope.id == message_id:
                    return _Site(name, index)
        return None

    def _rewrite(
        self, site: _Site | None, envelope: Envelope, update: dict[str, Any] | None
    ) -> Envelope:
        """Apply an edit to one message, keeping every copy of it in step.

        Envelopes are immutable value objects, so an edit produces a new one under
        the same id; `update=None` removes it instead. The graph holds the
        authoritative copy — `terminal_message` and `reachable_work` read it — so
        failing to write back would leave a submit that healing cancelled still
        looking terminal, and `EarlyTermination` would re-fire on it for the rest
        of the run.
        """
        replacement = envelope.model_copy(update=update) if update is not None else None
        if replacement is not None:
            self.graph.messages[replacement.id] = replacement
        if site is not None:
            outputs = list(self.turns[site.participant].outputs)
            if replacement is None:
                outputs.pop(site.index)
            else:
                outputs[site.index] = replacement
            self._update_turn(site.participant, {"outputs": tuple(outputs)})
        return replacement if replacement is not None else envelope

    def _update_turn(self, participant: str, update: dict[str, Any]) -> None:
        self.turns[participant] = self.turns[participant].model_copy(update=update)

    def _notify(self, recipients: tuple[str, ...], envelope: Envelope) -> None:
        for recipient in recipients:
            self.notices.setdefault(recipient, []).append(envelope)

    def _record(
        self, message_id: str, actor: str, action: EdgeAction, reason: str
    ) -> None:
        self.graph.record_intervention(
            message_id, actor, action=action, round=self.round, reason=reason
        )


type _Handler = Callable[[_Edit, Intervention, Envelope, _Site | None], None]

_HANDLERS: dict[str, _Handler] = {
    "inject": _Edit._inject,
    "drop": _Edit._drop,
    "reroute": _Edit._reroute,
}
"""Dispatch by `Intervention.kind`. A table rather than a chain of `elif`s so
that adding a kind is adding a method and a row, and `Intervention`'s literal
type is the only place the vocabulary is spelled out."""
