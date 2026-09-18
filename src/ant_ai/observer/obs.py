from __future__ import annotations

from collections.abc import Iterator
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    contextmanager,
    suppress,
)
from contextvars import ContextVar
from typing import Any


class ObservabilitySingleton:
    """Global observability singleton.

    Thread-safe and async-safe: contextvar state is task-local and restored
    on exit from `bind()`, so concurrent invocations never bleed into each other.
    """

    _sink: Any = None
    _ctx: ContextVar[dict[str, Any]] = ContextVar("obs_ctx", default={})  # noqa: B039

    def configure(self, sink: Any) -> None:
        """Set the active sink. Pass None to disable observability.

        Args:
            sink: An `ObservabilitySink` implementation, or None to disable.
        """
        self._sink = sink

    @contextmanager
    def bind(self, **fields: Any):
        """Merge fields into the current task's observability context.

        The context is restored on exit, so calls nest safely and concurrent
        invocations stay independent. All `event` and `span` calls made inside
        the block automatically include these fields.

        Leaving the block from a task other than the one that entered it (an
        async generator closed elsewhere) restores the leaving task's context;
        the entering task's context is out of reach and is left as it was.

        Restoring by value rather than via `ContextVar.reset(token)` is what
        makes that safe: a token is only valid in the Context that created it,
        so resetting it from another task's (copied) Context raises `ValueError`.
        Setting the saved value has no such restriction.

        Args:
            **fields: Key-value pairs to add to the current context.
        """
        previous = self._ctx.get()
        self._ctx.set({**previous, **fields})
        try:
            yield
        finally:
            self._ctx.set(previous)

    async def event(self, name: str, **fields: Any) -> None:
        """Emit a named lifecycle event with structured metadata.

        Context fields bound via `bind()` are merged in automatically.
        Never raises — sink errors are swallowed to protect the runtime.

        Args:
            name: Dot-namespaced event name (e.g. `step.start`).
            **fields: Additional metadata to attach to the event.
        """
        if self._sink is None:
            return
        with suppress(Exception):
            await self._sink.event(name, **{**self._ctx.get(), **fields})

    async def exception(self, name: str, error: Exception, **fields: Any) -> None:
        """Emit a named error event carrying the exception instance.

        Context fields bound via `bind()` are merged in automatically.
        Never raises — sink errors are swallowed to protect the runtime.

        Args:
            name: Dot-namespaced event name (e.g. `step.error`).
            error: The exception to record.
            **fields: Additional metadata to attach to the event.
        """
        if self._sink is None:
            return
        with suppress(Exception):
            await self._sink.exception(name, error, **{**self._ctx.get(), **fields})

    def span(self, name: str, **attrs: Any) -> AbstractAsyncContextManager[Any]:
        """Return an async context manager representing a unit of work.

        Returns a no-op context manager when no sink is configured.
        Context fields bound via `bind()` are merged into the span attributes.

        Args:
            name: Operation name for the span (e.g. `llm`, `tool`).
            **attrs: Additional attributes to attach to the span.

        Returns:
            An async context manager that opens and closes the span.
        """
        if self._sink is None:
            return _noop_span()
        return self._sink.span(name, **{**self._ctx.get(), **attrs})

    def propagation_headers(self) -> dict[str, str]:
        """Return headers encoding the current trace context for outbound requests."""
        if self._sink is None:
            return {}
        with suppress(Exception):
            return self._sink.propagation_headers()
        return {}

    @contextmanager
    def attach_propagation_context(self, headers: dict[str, str]) -> Iterator[None]:
        """Context manager that restores a remote trace context for the duration of the block."""
        if self._sink is None:
            yield
            return
        try:
            cm = self._sink.attach_propagation_context(headers)
        except Exception:
            yield
            return
        with cm:
            yield


class _NoopSpan:
    def update(self, **_: object) -> None:
        pass


@asynccontextmanager
async def _noop_span():
    yield _NoopSpan()


obs: ObservabilitySingleton = ObservabilitySingleton()
"""The global observability singleton.

Import and use this directly to emit events, record exceptions, and open spans.
"""
