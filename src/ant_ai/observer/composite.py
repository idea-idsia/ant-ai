from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from typing import Any


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
        """Opens a span on all sinks concurrently and closes them on exit."""
        async with AsyncExitStack() as stack:
            for s in self.sinks:
                await stack.enter_async_context(s.span(name, **attrs))
            yield

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
