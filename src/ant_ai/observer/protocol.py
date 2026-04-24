from __future__ import annotations

from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Protocol


class ObservabilitySink(Protocol):
    """Protocol for observability backends.

    Implementations must be safe to call concurrently and must never raise —
    wrap any internal errors in a try/except.
    """

    async def event(self, name: str, **fields: object) -> None:
        """Emit a named lifecycle event with structured metadata.

        Event names follow dot-namespaced conventions, e.g.
        `workflow.start`, `node.error`, `step.end`.

        Args:
            name: Dot-namespaced event name.
            **fields: Arbitrary structured metadata to attach to the event.
        """
        ...

    async def exception(self, name: str, error: Exception, **fields: object) -> None:
        """Emit an error event carrying the exception instance.

        Tracing backends should call `span.record_exception`; log backends
        should log the exception alongside the structured fields.

        Args:
            name: Dot-namespaced event name.
            error: The exception to record.
            **fields: Arbitrary structured metadata to attach to the event.
        """
        ...

    def span(self, name: str, **attrs: object) -> AbstractAsyncContextManager:
        """Return an async context manager representing a unit of work.

        Span names use operation names, not lifecycle suffixes (e.g. `llm`,
        `tool`, `router`). Attributes carry per-operation metadata such as
        `model`, `tool_name`, `messages`, or `session_id`.

        Args:
            name: Operation name for the span.
            **attrs: Metadata attributes to attach to the span.

        Returns:
            An async context manager that opens and closes the span.
        """
        ...

    def propagation_headers(self) -> dict[str, str]:
        """Return headers encoding the current trace context for outbound requests.

        Used to propagate distributed trace context across process boundaries
        (e.g. into A2A HTTP calls). Returns an empty dict when tracing is
        inactive or not supported.
        """
        ...

    def attach_propagation_context(
        self, headers: dict[str, str]
    ) -> AbstractContextManager:
        """Return a context manager that restores a remote trace context.

        Extracts the trace context encoded in *headers* (e.g. from an inbound
        A2A request) and makes it the active context for the duration of the
        block, so child observations are nested under the remote parent span.

        Args:
            headers: Carrier dict previously produced by `propagation_headers`.
        """
        ...
