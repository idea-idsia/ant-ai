"""The append-only record of every edit, and what it would take to undo them.

A round used to hand back the topology it had decided, and the run recorded that
topology. Nothing recorded the *edit*. The difference looks academic until
something has to be asked of it: which components must be revalidated after this
update, what would undoing it restore, did this change anything it had no
business changing, has the deleted thing's influence actually gone. All four are
questions about edits, and a record of states cannot answer any of them.

So this keeps the edits. Each entry pairs a `Rewrite` with the inverse the state
graph computed while applying it — the inverse comes from the graph rather than
from the rewrite because only the graph knew the before-state — plus the scope
the edit reaches, resolved at the time it was applied rather than later, when the
graph has moved on.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ant_ai.topology.rewrite import Rewrite

if TYPE_CHECKING:
    from ant_ai.topology.state import StateGraph

__all__ = ["LogEntry", "RewriteLog"]


class LogEntry(BaseModel):
    """One applied edit, with everything an audit of it needs."""

    rewrite: Rewrite
    inverse: Rewrite | None = Field(
        default=None,
        description="What undoes it. None for a read-only `activate`, and for an "
        "edit that matched nothing.",
    )
    affected: tuple[str, ...] = Field(
        default=(),
        description="Components downstream of the target when this was applied — "
        "what a revalidation would have to cover.",
    )
    applied: bool = True
    problem: str = Field(
        default="",
        description="Why it was not applied, when it was not. A rejected edit is "
        "kept rather than dropped: a strategy emitting edits the schema forbids "
        "is a finding about the strategy, and silence about it is how that goes "
        "unnoticed for a whole benchmark run.",
    )

    @property
    def at(self) -> int:
        return self.rewrite.at


class RewriteLog(BaseModel):
    """Every edit a run made, in order."""

    entries: list[LogEntry] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def record(self, rewrite: Rewrite, state: StateGraph | None = None) -> LogEntry:
        """Apply an edit to *state* if given, and log it either way.

        A rewrite that the schema rejects is logged as unapplied rather than
        raised: the caller is a round loop running somebody's published method,
        and killing the run over one malformed edit would throw away the trace
        that shows which edit it was.
        """
        from ant_ai.topology.state import SchemaViolation

        inverse: Rewrite | None = None
        affected: tuple[str, ...] = ()
        applied = True
        problem = ""

        if state is not None:
            target_id = (
                rewrite.target.id
                if rewrite.target.ref == "node"
                else rewrite.target.dst
            )
            affected = tuple(sorted(state.affected(target_id)))
            try:
                inverse = state.apply(rewrite)
            except SchemaViolation as violation:
                applied = False
                problem = str(violation)

        entry = LogEntry(
            rewrite=rewrite,
            inverse=inverse,
            affected=affected,
            applied=applied,
            problem=problem,
        )
        self.entries.append(entry)
        return entry

    def extend(
        self, rewrites: Iterable[Rewrite], state: StateGraph | None = None
    ) -> list[LogEntry]:
        return [self.record(rewrite, state) for rewrite in rewrites]

    # -- reading it back ---------------------------------------------------

    def since(self, t: int) -> tuple[LogEntry, ...]:
        return tuple(e for e in self.entries if e.at >= t)

    def by_op(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.applied:
                counts[entry.rewrite.op] = counts.get(entry.rewrite.op, 0) + 1
        return counts

    def by_kind(self) -> dict[str, int]:
        """How many edits changed each component type.

        The one number that says whether a run evolved anything beyond its
        wiring. `activate` is excluded: it is an operator, and it appears in
        `by_op`, but it changes nothing — counting a retrieval here would make
        every run that merely *read* its memory look like one that rewrote it.
        Recorded activations are counted on their own, as `RunReport.activations`.
        """
        counts: dict[str, int] = {}
        for entry in self.entries:
            if not entry.applied or entry.rewrite.op == "activate":
                continue
            for kind in entry.rewrite.kinds:
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    def cascade(self, cause: str) -> tuple[LogEntry, ...]:
        """The paper's `C(c)`: every edit one cause produced, in order."""
        return tuple(e for e in self.entries if e.rewrite.cause == cause)

    def cascades(self) -> dict[str, tuple[LogEntry, ...]]:
        grouped: dict[str, list[LogEntry]] = {}
        for entry in self.entries:
            if entry.rewrite.cause:
                grouped.setdefault(entry.rewrite.cause, []).append(entry)
        return {cause: tuple(v) for cause, v in grouped.items()}

    def cross_component(self) -> tuple[str, ...]:
        """Causes whose cascade touched two or more component types.

        Their `|union tau(S_i)| >= 2` — the discriminator for A.4, and the only
        way to tell a co-evolution run from a run that did four unrelated things.
        """
        return tuple(
            sorted(
                cause
                for cause, entries in self.cascades().items()
                if len({k for e in entries for k in e.rewrite.kinds}) >= 2
            )
        )

    def rejected(self) -> tuple[LogEntry, ...]:
        return tuple(e for e in self.entries if not e.applied)

    # -- undoing -----------------------------------------------------------

    def rollback(self, state: StateGraph, *, to: int) -> tuple[Rewrite, ...]:
        """Undo every edit made at or after *to*, newest first.

        Newest first because the inverses were computed against the state each
        edit saw, so applying them out of order restores attributes that a later
        edit had already replaced. The undo edits are themselves applied to the
        graph and returned, but deliberately **not** logged: a rollback that
        appends to the log makes the next rollback undo the undo.
        """
        undone: list[Rewrite] = []
        for entry in reversed(self.entries):
            if entry.at < to or entry.inverse is None or not entry.applied:
                continue
            try:
                state.apply(entry.inverse)
            except Exception:  # pragma: no cover - defensive
                continue
            undone.append(entry.inverse)
        return tuple(undone)
