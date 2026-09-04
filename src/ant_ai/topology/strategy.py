"""Packaging a published approach as a named, validated bundle.

A strategy's class body is the *delta* from framework baseline. It used to have
five overridable hooks plus a bespoke layering field, which meant a method whose
paper changes only supervision was still forced to state a routing choice it does
not make. Now there is one hook — `build()` — returning the assembled `Pipeline`,
and layering is `|`.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.topology.materialise import (
    TopologyMaterialiser,
    VisibilityMaterialiser,
)
from ant_ai.topology.plan import RoundPlan, RunContext, Stage
from ant_ai.topology.schedule import RoundScheduler, Scheduler

__all__ = ["Halt", "HaltPolicy", "Pipeline", "TopologyStrategy"]


@runtime_checkable
class HaltPolicy(Protocol):
    """Decides whether a run is over.

    A seam because "who may end the run" is a *method* choice, not a framework
    constant. Left implicit it silently distorts comparisons: with agent-declared
    halting, the first agent to finish its own piece ends everyone's round, so two
    conditions run for different numbers of rounds and their token totals stop
    being comparable.
    """

    def halt(self, plan: RoundPlan, ctx: RunContext) -> str | None:
        """Why the run should stop, or None to continue."""
        ...


class Halt(BaseModel):
    """The built-in rule: one predicate with three knobs.

    This was five classes — `AnySubmits`, `AllSubmit`, `Designated`, `NeverHalt`
    and a `MinRounds` wrapper — which forced the case benchmarking actually hits
    into an awkward nesting, `MinRounds(inner=Designated("integrator"), rounds=3)`.
    They are one predicate parameterised three ways, and combining two constraints
    should not require composing two objects.

        Halt()                                     # anyone may end the run
        Halt(unanimous=True)                       # everyone must agree
        Halt(deciders={"integrator"}, min_rounds=3)
        Halt.never()                               # pin the round budget
    """

    deciders: frozenset[str] | None = Field(
        default=None,
        description="Who may end the run. None means anyone; the empty set means "
        "nobody, which is how `never()` works.",
    )
    unanimous: bool = Field(
        default=False, description="Require every decider to submit, not just one."
    )
    min_rounds: int = Field(
        default=0,
        ge=0,
        description="Rounds that must elapse before any of this applies. Exists "
        "because a weak completion signal — docstring examples passing, say — "
        "otherwise ends a run at round 1 on false confidence and the topology "
        "never gets a second round to rewire.",
    )

    @classmethod
    def never(cls) -> Halt:
        """Run the full round budget. The control condition for cost comparisons."""
        return cls(deciders=frozenset())

    def halt(self, plan: RoundPlan, ctx: RunContext) -> str | None:
        if ctx.round < self.min_rounds:
            return None

        candidates = {
            name
            for name in plan.turns
            if self.deciders is None or name in self.deciders
        }
        if not candidates:
            return None

        submitted = sorted(n for n in candidates if plan.turns[n].submitted)
        if self.unanimous:
            if len(submitted) == len(candidates):
                return "all deciders submitted"
            return None
        return f"{submitted[0]} submitted" if submitted else None


class Pipeline(BaseModel):
    """An assembled strategy: what `Ensemble` actually runs.

    Stages run in order, each transforming the next round's plan. The order is
    load-bearing and visible — a scoring stage must precede the sparsifier that
    reads its scores — which is the honest cost of composing by concatenation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stages: list[Annotated[Stage, SkipValidation]] = Field(default_factory=list)
    scheduler: Annotated[Scheduler, SkipValidation] = Field(
        default_factory=RoundScheduler
    )
    materialiser: Annotated[TopologyMaterialiser, SkipValidation] = Field(
        default_factory=VisibilityMaterialiser
    )
    halt: Annotated[HaltPolicy, SkipValidation] = Field(default_factory=Halt)
    max_rounds: int = Field(default=10, ge=1)
    max_depth: int = Field(default=3, ge=1)

    @property
    def needs_structured_turns(self) -> bool:
        """Whether any stage reads a field only a structured turn can carry.

        A component opts in with `needs_structured_turns = True`; `getattr` rather
        than the protocol, so a duck-typed stage that has never heard of the flag
        is simply one that does not need it.

        Asked by `Colony.ensemble` before it decides how a participant takes its
        turn, and it is not a preference. A turn driven by a `Workflow` carries no
        response schema, so its output degrades to one plain public message:
        `query`/`key` are empty, no message is addressed, no reaction is declared
        and nothing is ever `submitted`. Every stage built on those reads
        something that is now always absent — a semantic matcher scores unchanging
        profile text and rewires nothing; a detector never sees a submit, a wait
        or a reroute and finds nothing to repair. Both fail by looking like a run
        with no problems in it, which is why the choice cannot be left to a
        default that does not know what is in the pipeline.
        """
        return any(
            getattr(component, "needs_structured_turns", False)
            for component in (*self.stages, self.materialiser, self.scheduler)
        )

    def __or__(self, other: Pipeline) -> Pipeline:
        """Concatenate stages; the right-hand side wins on every other field.

        Only fields the right side set *explicitly* override — `model_fields_set`
        rather than a value comparison — so composing never silently reverts a
        setting to a default just because the other side did not mention it.
        """
        overrides = {
            field: getattr(other, field)
            for field in other.model_fields_set
            if field != "stages"
        }
        return self.model_copy(
            update={"stages": [*self.stages, *other.stages], **overrides}
        )


class TopologyStrategy(BaseModel):
    """A published approach: its hyperparameters, and how they assemble.

    Hyperparameters stay pydantic fields rather than function arguments because
    all three of the things they are used for depend on it: they are validated at
    construction, varied by an ablation sweep, and read straight off
    `model_fields` for the run record.

    Named alongside `TopologyMaterialiser` and `TopologyEvent` in this package.
    Subclasses register themselves by `name`, so a strategy is constructible from
    config and a run can record which method and hyperparameters produced it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: ClassVar[str] = ""
    citation: ClassVar[str] = ""

    max_rounds: int = Field(default=10, ge=1)
    max_depth: int = Field(default=3, ge=1)

    _registry: ClassVar[dict[str, type[TopologyStrategy]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("name") or ""
        if not name:
            return
        existing = TopologyStrategy._registry.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Topology strategy '{name}' is already registered by {existing.__name__}."
            )
        TopologyStrategy._registry[name] = cls

    @classmethod
    def get(cls, name: str) -> type[TopologyStrategy]:
        try:
            return TopologyStrategy._registry[name]
        except KeyError:
            known = ", ".join(sorted(TopologyStrategy._registry)) or "none"
            raise KeyError(
                f"Unknown topology strategy '{name}'. Known: {known}."
            ) from None

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TopologyStrategy:
        """Build a strategy by name — the path config-driven ablations take."""
        return cls.get(name)(**kwargs)

    @classmethod
    def known(cls) -> list[str]:
        return sorted(TopologyStrategy._registry)

    # -- the one hook -----------------------------------------------------

    def build(self) -> Pipeline:
        """Assemble this strategy's components. Override this and nothing else."""
        return Pipeline()

    def pipeline(self) -> Pipeline:
        """`build()` plus the settings every strategy shares.

        Applied here so no subclass has to remember to thread `max_rounds` and
        `max_depth` through its own `Pipeline(...)` call.

        Only settings this strategy set *explicitly* are stamped on. Stamping the
        defaults too would make composition wrong in a way that is hard to see:
        `Custom(max_rounds=4) | DigToHeal()` would silently run for ten rounds,
        because the right-hand side would look like it had asked for its default.
        """
        built = self.build()
        settings = {
            field: getattr(self, field)
            for field in ("max_rounds", "max_depth")
            if field in self.model_fields_set and field not in built.model_fields_set
        }
        return built.model_copy(update=settings) if settings else built

    def __or__(self, other: TopologyStrategy) -> TopologyStrategy:
        """Layer *other* on top of this one."""
        return Composite(members=[self, other])

    def provenance(self) -> dict[str, Any]:
        """Which method with which hyperparameters, for the run record.

        Nested models are dumped rather than stringified, and values that genuinely
        cannot serialise — a live embedder — fall back to their `model_id`. The
        difference between a reproducible result and a directory of unlabelled
        graphs.
        """
        data: dict[str, Any] = {"strategy": self.name, "citation": self.citation}
        for field in type(self).model_fields:
            data[field] = _serialise(getattr(self, field))
        return data


class Composite(TopologyStrategy):
    """Two or more strategies layered, as produced by `a | b`.

    A class rather than a bare `Pipeline` so that provenance survives composition:
    returning the folded pipeline directly would record which components ran but
    lose which published methods and hyperparameters produced them.
    """

    members: list[Annotated[TopologyStrategy, SkipValidation]] = Field(
        default_factory=list
    )

    def build(self) -> Pipeline:
        pipeline = Pipeline()
        for member in self.members:
            pipeline = pipeline | member.pipeline()
        return pipeline

    def __or__(self, other: TopologyStrategy) -> TopologyStrategy:
        return Composite(members=[*self.members, other])

    def provenance(self) -> dict[str, Any]:
        return {
            "strategy": "|".join(m.name or type(m).__name__ for m in self.members),
            "layers": [m.provenance() for m in self.members],
        }


def _serialise(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    return getattr(value, "model_id", type(value).__name__)
