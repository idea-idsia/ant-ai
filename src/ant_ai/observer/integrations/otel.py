from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any


class _OTelSpan:
    """The handle `OTelSink.span` yields."""

    def __init__(self, span: Any) -> None:
        self.span = span

    def update(self, **fields: Any) -> None:
        """Record the fields on the span, mirroring `span()`'s own attributes."""
        from opentelemetry.trace import StatusCode

        try:
            for k, v in fields.items():
                if v is not None:
                    self.span.set_attribute(k, str(v))
            if str(fields.get("level", "")).upper() == "ERROR":
                self.span.set_status(
                    StatusCode.ERROR, str(fields.get("status_message") or "")
                )
        except Exception:
            pass


class OTelSink:
    """Creates OpenTelemetry spans for leaf operations (LLM, tool calls)."""

    def __init__(self, tracer: Any) -> None:
        self.tracer = tracer

    async def event(self, name: str, **fields) -> None:
        """Attach a named event to the currently active OTel span."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span.is_recording():
                span.add_event(name, attributes={k: str(v) for k, v in fields.items()})
        except Exception:
            pass

    async def exception(self, name: str, error: Exception, **fields) -> None:
        """Record an exception on the currently active OTel span."""
        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode

            span = trace.get_current_span()
            if span.is_recording():
                span.record_exception(error)
                span.set_status(StatusCode.ERROR, str(error))
        except Exception:
            pass

    def propagation_headers(self) -> dict[str, str]:
        from opentelemetry.propagate import inject as _inject

        carrier: dict[str, str] = {}
        _inject(carrier)
        return carrier

    @contextmanager
    def attach_propagation_context(self, headers: dict[str, str]):
        from opentelemetry import context as _ctx
        from opentelemetry.propagate import extract as _extract

        token = _ctx.attach(_extract(headers))
        try:
            yield
        finally:
            _ctx.detach(token)

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        """
        Open an OTel span for the duration of the async with block.

        Uses start_as_current_span so OTel's own context propagation
        handles parent-child relationships — no module-level dict required,
        fully safe under concurrent invocations.
        """
        from opentelemetry.trace import StatusCode

        with self.tracer.start_as_current_span(name) as span:
            for k, v in attrs.items():
                span.set_attribute(k, str(v))
            try:
                yield _OTelSpan(span)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise
