from __future__ import annotations

import asyncio
from contextlib import (
    AsyncExitStack,
    ExitStack,
    asynccontextmanager,
    contextmanager,
    suppress,
)
from typing import Any


class _FanOutSpan:
    """The handle `CompositeSink.span` yields.

    Each sink yields its own span object, so the composite has to hand callers
    one thing that reaches all of them. `update()` is the whole contract call
    sites use (`llm_step`, `tool_step`), and one sink failing it must not stop
    the others -- the same guarantee `event` and `exception` already give.
    """

    def __init__(self, spans: list[Any]) -> None:
        self.spans = spans

    def update(self, **fields: Any) -> None:
        """Apply the update to every underlying sink span."""
        for span in self.spans:
            with suppress(Exception):
                span.update(**fields)


class CompositeSink:
    """Fans out all observability calls to a list of sinks.

    Each method is dispatched to all sinks concurrently. Errors from
    individual sinks are swallowed via `return_exceptions=True` so a
    failing sink never affects the others.
    """

    def __init__(self, sinks: list[Any]) -> None:
        """
        Args:
            sinks: `ObservabilitySink` implementations to fan out to.
        """
        self.sinks: list[Any] = sinks

    async def event(self, name: str, **fields) -> None:
        """Fans out the event to all sinks concurrently."""
        await asyncio.gather(
            *(s.event(name, **fields) for s in self.sinks),
            return_exceptions=True,
        )

    async def exception(self, name: str, error: Exception, **fields) -> None:
        """Fans out the error event to all sinks concurrently."""
        await asyncio.gather(
            *(s.exception(name, error, **fields) for s in self.sinks),
            return_exceptions=True,
        )

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        """Opens a span on all sinks and yields a handle that updates them all."""
        async with AsyncExitStack() as stack:
            spans = [
                await stack.enter_async_context(s.span(name, **attrs))
                for s in self.sinks
            ]
            yield _FanOutSpan(spans)

    def propagation_headers(self) -> dict[str, str]:
        """Merges propagation headers from all sinks."""
        result: dict[str, str] = {}
        for s in self.sinks:
            result.update(s.propagation_headers())
        return result

    @contextmanager
    def attach_propagation_context(self, headers: dict[str, str]):
        """Restores propagation context in all sinks."""
        with ExitStack() as stack:
            for s in self.sinks:
                stack.enter_context(s.attach_propagation_context(headers))
            yield
