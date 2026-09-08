"""Configuration checks that run before a single agent does.

Every problem this module reports was, until it existed, a run that completed
successfully and produced nonsense: a matcher scoring text that never changes, a
detector firing on defaults nobody declared, a decided topology binding tools on
agents in another process. The failures are silent because the layer's defaults
are individually reasonable and only wrong in combination.

So the checks are deliberately conservative. Each one fires only where the
outcome is *provable from the configuration alone* — not where it is merely
unlikely to be what you meant. A check that guesses would be another thing to
work around, and working around a check is how people learn to ignore them.

`Colony.ensemble()` is the guided path and runs these. Constructing an
`Ensemble` directly does not, which is the escape hatch for anyone who genuinely
knows better.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ant_ai.topology.materialise import DeliveryMaterialiser, VisibilityMaterialiser
from ant_ai.topology.problem import Problem

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ant_ai.topology.rewrite import Rewrite
    from ant_ai.topology.state import StateGraph
    from ant_ai.topology.strategy import Pipeline

__all__ = ["check", "check_rewrites"]


class _Checkable(Protocol):
    """The part of a `Pipeline` a check reads.

    Structural rather than importing `Pipeline`, which imports this module for
    its own `check()` method.
    """

    @property
    def stages(self) -> list: ...
    @property
    def materialiser(self) -> object: ...
    @property
    def max_rounds(self) -> int: ...
    @property
    def needs_structured_turns(self) -> bool: ...
    @property
    def writes_links(self) -> bool: ...


def check(
    pipeline: Pipeline | _Checkable,
    *,
    structured_turns: bool,
    local: bool = True,
    seeded: bool = False,
    participants: int = 0,
    agents_have_tools: bool = False,
    unrunnable_workflows: tuple[str, ...] = (),
) -> list[Problem]:
    """Everything provably wrong with this configuration, errors first.

    Args:
        pipeline: The assembled pipeline about to run.
        structured_turns: Whether participants will actually be invoked with a
            response schema. False for workflow-driven and for remote
            participants, both of which degrade to one plain public message.
        local: Whether participants run in this process.
        seeded: Whether the run has declared round-0 links to fall back on.
        participants: How many participants the run has.
        agents_have_tools: Whether any participant's agent carries tools of its
            own. Decides which of two very different paths a response schema
            takes — see `_coercion`.
        unrunnable_workflows: Names of participants whose registered workflow
            cannot execute. Only meaningful when workflows will actually run.

    Returns:
        Problems in severity order. Empty means the configuration can work —
        not that it will produce a good answer.
    """
    problems = [
        *_structured_turns(pipeline, structured_turns=structured_turns, local=local),
        *_routing(pipeline, structured_turns=structured_turns, seeded=seeded),
        *_remote_visibility(pipeline, local=local),
        *_workflows(unrunnable_workflows),
        *_coercion(
            pipeline,
            structured_turns=structured_turns,
            agents_have_tools=agents_have_tools,
        ),
        *_halting(pipeline, structured_turns=structured_turns),
        *_size(pipeline, participants=participants),
        *_cadence(pipeline),
    ]
    return sorted(problems, key=lambda p: p.level != "error")


def _declaring(pipeline: Pipeline | _Checkable, flag: str) -> list[str]:
    """Names of the components that set *flag*, for naming them in a message.

    A stage on a cadence is named by the stage, not the wrapper: being told that
    `Every` needs structured turns says nothing about which method to change.
    """
    return [
        type(getattr(c, "wrapped", c)).__name__
        for c in (*pipeline.stages, pipeline.materialiser)
        if getattr(c, flag, False)
    ]


def _structured_turns(
    pipeline: Pipeline | _Checkable, *, structured_turns: bool, local: bool
) -> list[Problem]:
    """E001 — components that read declared fields, with nothing declaring them.

    The failure this prevents is not an exception but a plausible run: without a
    response schema a turn carries no query, no key, no addressing, no reactions
    and no submit, so a matcher scores static profile text every round and a
    detector reads defaults nobody chose. Both then report on the schema's
    absence rather than on the collaboration.
    """
    if not pipeline.needs_structured_turns or structured_turns:
        return []
    cause = (
        "remote (A2A) participants cannot be invoked with a response schema"
        if not local
        else "workflow-driven participants cannot be invoked with a response schema"
    )
    return [
        Problem(
            code="E001",
            level="error",
            message=(
                f"{', '.join(_declaring(pipeline, 'needs_structured_turns'))} "
                f"read fields only a structured turn carries, but {cause}."
            ),
            hint=(
                "Use `ensemble(local=True)` with the default `use_workflows=False`, "
                "or choose a strategy that does not need structured turns "
                "(`Baseline`, `Static`, `RandomTopology`)."
            ),
        )
    ]


def _routing(
    pipeline: Pipeline | _Checkable, *, structured_turns: bool, seeded: bool
) -> list[Problem]:
    """E002 — delivery mode with no source of links and no way to address.

    A message that names no recipients is routed by the links, so with neither
    links nor addressing every message is generated and delivered to nobody. The
    run completes, every round, having moved nothing.

    Not raised when turns are structured: agents can then address each other
    directly, which is a legitimate configuration with no matcher under it.
    """
    if not isinstance(pipeline.materialiser, DeliveryMaterialiser):
        return []
    if pipeline.writes_links or seeded or structured_turns:
        return []
    return [
        Problem(
            code="E002",
            level="error",
            message=(
                "Delivery mode with no stage that writes links, no declared "
                "`collab()` edges, and unstructured turns that cannot address "
                "anyone — no message can reach a recipient."
            ),
            hint=(
                "Declare `collab()` edges, add a routing strategy "
                "(`DyTopo`, `Static`, `chain`/`star`/`mesh`), or enable "
                "structured turns so agents can address each other."
            ),
        )
    ]


def _remote_visibility(
    pipeline: Pipeline | _Checkable, *, local: bool
) -> list[Problem]:
    """E003 — a decided topology materialised as tools on agents elsewhere.

    A2A has no operation for attaching a tool to an agent in another process, so
    every remote participant reports itself unbindable and the topology
    constrains nothing at all. Gated on there being a stage: a colony with no
    strategy decides nothing, and its remote agents stay wired as their servers
    wired them, which is the pre-topology behaviour rather than a failure.
    """
    if local or not pipeline.stages:
        return []
    if not isinstance(pipeline.materialiser, VisibilityMaterialiser):
        return []
    return [
        Problem(
            code="E003",
            level="error",
            message=(
                "Remote (A2A) participants cannot be rebound, so a topology "
                "materialised as peer tools has no effect on them."
            ),
            hint=(
                "Pass `materialiser=DeliveryMaterialiser()` to route their "
                "messages instead, or build local participants."
            ),
        )
    ]


def _workflows(unrunnable: tuple[str, ...]) -> list[Problem]:
    """E004 — a workflow that will drive the turns and cannot execute.

    `Ensemble._act` never raises: a participant that throws is recorded as a
    failed activation and the round continues, which is right for a model error
    and wrong for a graph that could never have run. Without this the run
    completes with every activation errored, no messages, and the cheerful
    answer "Ensemble completed".
    """
    if not unrunnable:
        return []
    return [
        Problem(
            code="E004",
            level="error",
            message=(
                f"The workflow registered for {', '.join(sorted(unrunnable))} "
                "cannot run, so every turn would fail and the run would complete "
                "with no messages."
            ),
            hint=(
                "Give the workflow a valid graph (an edge from START and a path "
                "to END), or drop `use_workflows=True` to invoke the agents "
                "directly."
            ),
        )
    ]


def _coercion(
    pipeline: Pipeline | _Checkable,
    *,
    structured_turns: bool,
    agents_have_tools: bool,
) -> list[Problem]:
    """W003 — a response schema that will be coerced rather than constrained.

    `Agent` builds a tool step only when its registry is non-empty, and the ReAct
    loop applies `response_schema` natively only when there is no tool step.
    Give an agent tools — its own, or the peer tools a visibility materialiser
    binds — and the schema instead becomes a *repair* pass: the turn is generated
    as prose, then a second model is asked to "convert the following text into a
    JSON object".

    Measured, one turn: 1 LLM call without tools, 2 with. And the second call is
    the problem, not the cost. `submitted`, `query`, `key` and the reactions come
    out of a model that never saw the topology contract and is filling required
    fields from prose that was not written to answer them. A fabricated
    `submitted` ends the run through `Halt`; a fabricated `query` steers the
    matcher. Unlike a missing declaration, which degrades visibly, an invented
    one is indistinguishable from a real one.

    A warning rather than an error because the run still works and the answer is
    still the agent's — it is the *declarations* that become second-hand.
    """
    if not structured_turns or not pipeline.needs_structured_turns:
        return []
    visibility = isinstance(pipeline.materialiser, VisibilityMaterialiser)
    if not visibility and not agents_have_tools:
        return []
    cause = (
        "peer tools are bound to each participant"
        if visibility
        else "participants carry tools of their own"
    )
    return [
        Problem(
            code="W003",
            level="warning",
            message=(
                f"{', '.join(_declaring(pipeline, 'needs_structured_turns'))} "
                f"read declared fields, but {cause}, so each turn costs a second "
                "LLM call and its declarations are produced by a repair model "
                "rather than by the agent."
            ),
            hint=(
                "Use `DeliveryMaterialiser` with agents that carry no tools of "
                "their own to get one constrained call per turn, or treat "
                "`query`/`key`/`submitted` as approximate in this run."
            ),
        )
    ]


def _halting(
    pipeline: Pipeline | _Checkable, *, structured_turns: bool
) -> list[Problem]:
    """W001 — nothing can declare completion, so the budget decides the length.

    A warning rather than an error: running the full budget is exactly what a
    cost-comparison control wants. It is only surprising when it was not chosen.
    """
    if structured_turns or not pipeline.stages:
        return []
    return [
        Problem(
            code="W001",
            level="warning",
            message=(
                "No participant can declare `submitted` without a structured "
                f"turn, so this run will use all {pipeline.max_rounds} rounds."
            ),
            hint="Use `Halt.never()` to state that intent, or enable structured turns.",
        )
    ]


def _size(pipeline: Pipeline | _Checkable, *, participants: int) -> list[Problem]:
    """W002 — a topology over fewer than two participants decides nothing."""
    if participants >= 2 or not pipeline.stages:
        return []
    return [
        Problem(
            code="W002",
            level="warning",
            message=(
                f"A topology strategy is configured but the run has "
                f"{participants} participant(s), so there is nothing to rewire."
            ),
            hint="Register at least two agents on the colony.",
        )
    ]


def _cadence(pipeline: Pipeline | _Checkable) -> list[Problem]:
    """W004 — a slow loop whose period does not fit inside the run.

    The cost of a second clock is that it can be set past the end of time. A
    stage wrapped in `Every(k=5)` on a five-round budget fires once if its phase
    happens to land and never otherwise, and either way the co-evolution the
    cadence was expressing did not happen — while the run completes and the
    strategy's name is still in the report.
    """
    problems: list[Problem] = []
    for stage in pipeline.stages:
        # Read off the stage rather than isinstance-checked against `Every`, for
        # the same reason every other declaration here is read that way: a
        # wrapper somebody else wrote that answers the same three questions is
        # one this check should apply to.
        period = getattr(stage, "k", None)
        wrapped = getattr(stage, "wrapped", None)
        due = getattr(stage, "due", None)
        if period is None or wrapped is None or due is None:
            continue
        fires = sum(1 for r in range(pipeline.max_rounds) if due(r))
        if fires > 1:
            continue
        name = type(wrapped).__name__
        problems.append(
            Problem(
                code="W004",
                level="warning",
                message=(
                    f"{name} is on a cadence of {period} round(s) but the run has "
                    f"{pipeline.max_rounds}, so it fires {fires} time(s)."
                ),
                hint=(
                    "Lower `k`, raise `max_rounds`, or drop the `Every` wrapper if "
                    "the stage was meant to run every round."
                ),
            )
        )
    return problems


def check_rewrites(state: StateGraph, rewrites: Iterable[Rewrite]) -> list[Problem]:
    """Everything the state graph's schema forbids in a batch of edits.

    The same contract as `check`, one level down: `check` says a pipeline cannot
    do what it was configured to do, this says an edit cannot be applied to the
    graph it names. Kept here rather than only on `StateGraph` because this is
    where a caller looks for "is this configuration going to work", and a
    strategy that emits edges between the wrong two node kinds is exactly that
    question asked about a strategy rather than about a pipeline.

    Returns:
        Problems in the order the edits were given. Empty means every edit is
        applicable — not that applying them is a good idea.
    """
    return [problem for rewrite in rewrites for problem in state.check(rewrite)]
