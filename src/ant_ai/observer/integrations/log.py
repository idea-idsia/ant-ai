from __future__ import annotations

from contextlib import asynccontextmanager

from ant_ai.core.logging import bind_logger


class StructlogSink:
    """
    Emits one structured log line per lifecycle event.

    Spans are no-ops — timing belongs in OTelSink or Prometheus histograms,
    not in log fields.
    """

    async def event(self, name: str, **fields) -> None:
        bind_logger(event=name, **fields).info(name)

    async def exception(self, name: str, error: Exception, **fields) -> None:
        bind_logger(
            event=name,
            error_type=type(error).__name__,
            **fields,
        ).exception(name)

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        yield  # Logging sink: lifecycle is covered by events; spans are no-ops
