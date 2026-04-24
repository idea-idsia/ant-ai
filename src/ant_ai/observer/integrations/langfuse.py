from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from typing import Any

from langfuse import propagate_attributes
from opentelemetry import context as _otel_context
from opentelemetry.propagate import extract as _otel_extract, inject as _otel_inject

# Workflow-level observation and its context managers (kept across event callbacks)
_workflow_obs: ContextVar[Any | None] = ContextVar("lf_workflow_obs", default=None)  # noqa: B039
_workflow_cm: ContextVar[Any | None] = ContextVar("lf_workflow_cm", default=None)  # noqa: B039
_propagate_cm: ContextVar[Any | None] = ContextVar("lf_propagate_cm", default=None)  # noqa: B039

# Stack of (node_name, run_step, obs, cm) for open node spans
_node_stack: ContextVar[list[tuple[str, int, Any, Any]]] = ContextVar(
    "lf_node_stack",
    default=[],  # noqa: B039
)

# Fields accepted directly by start_as_current_observation / update
_NATIVE_FIELDS = frozenset(
    {
        "input",
        "output",
        "model",
        "usage_details",
        "cost_details",
        "metadata",
        "level",
        "status_message",
        "version",
    }
)


def _clean(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data:
        return None
    out = {k: v for k, v in data.items() if v is not None}
    return out or None


def _split_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs suitable for start_as_current_observation / update.

    Unknown keys are folded into metadata so callers don't need to pre-filter.
    """
    native = {k: v for k, v in attrs.items() if k in _NATIVE_FIELDS and v is not None}
    extra = {
        k: v for k, v in attrs.items() if k not in _NATIVE_FIELDS and v is not None
    }
    if extra:
        native["metadata"] = _clean({**(native.get("metadata") or {}), **extra})
    elif "metadata" in native:
        native["metadata"] = _clean(native["metadata"])
    return native


class LangfuseSink:
    """
    Sends workflow and agent telemetry to Langfuse v4.

    Uses start_as_current_observation() throughout so that OpenTelemetry context
    propagation automatically handles parent-child relationships — no manual span
    stack required.  Propagate_attributes is held open for the full workflow
    lifetime so session_id / user_id reach every child observation.
    """

    def __init__(self, langfuse: Any) -> None:
        self.langfuse = langfuse
        self.last_trace_id: str | None = None
        self._event_handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "workflow.start": self._on_workflow_start,
            "node.start": self._on_node_start,
            "node.end": self._on_node_end,
            "workflow.end": self._on_workflow_end,
            "workflow.max_steps": self._on_workflow_end,
        }

    async def event(self, name: str, **fields) -> None:
        with suppress(Exception):
            handler = self._event_handlers.get(name)
            if handler is not None:
                handler(fields)

    async def exception(self, name: str, error: Exception, **fields) -> None:
        with suppress(Exception):
            self._on_exception(name, error, fields)

    @asynccontextmanager
    async def span(self, name: str, **attrs):
        as_type = attrs.pop("as_type", None) or (
            "generation" if name == "llm" else "tool" if name == "tool" else "span"
        )
        kwargs = _split_attrs(attrs)

        with self.langfuse.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        ) as obs:
            try:
                yield obs
            except BaseException as exc:
                with suppress(Exception):
                    obs.update(level="ERROR", status_message=str(exc))
                raise

    def _on_workflow_start(self, fields: dict[str, Any]) -> None:
        self._teardown()

        # Keep propagate_attributes open for the full run so every child
        # observation (nodes, generations, tools) inherits session_id / user_id.
        p_cm = propagate_attributes(
            trace_name=fields.get("agent_name") or "workflow",
            user_id=fields.get("user_id"),
            session_id=fields.get("session_id"),
            version=fields.get("version"),
            tags=fields.get("tags"),
        )
        p_cm.__enter__()
        _propagate_cm.set(p_cm)

        metadata = _clean(
            {
                "start_at": fields.get("start_at"),
                "max_steps": fields.get("max_steps"),
            }
        )
        cm = self.langfuse.start_as_current_observation(
            name=fields.get("agent_name") or "workflow",
            as_type="agent",
            input=fields.get("input"),
            metadata=metadata,
        )
        obs = cm.__enter__()
        _workflow_obs.set(obs)
        _workflow_cm.set(cm)
        _node_stack.set([])

        with suppress(Exception):
            self.last_trace_id = getattr(obs, "trace_id", None)

    def _on_node_start(self, fields: dict[str, Any]) -> None:
        node_name = fields.get("node", "node")
        run_step = int(fields.get("run_step") or 0)

        cm = self.langfuse.start_as_current_observation(
            name=node_name,
            as_type="span",
            input=fields.get("input"),
            metadata=_clean({"run_step": run_step}),
        )
        obs = cm.__enter__()

        stack = [*_node_stack.get(), (node_name, run_step, obs, cm)]
        _node_stack.set(stack)

    def _on_node_end(self, fields: dict[str, Any]) -> None:
        node_name = fields.get("node")
        run_step = int(fields.get("run_step") or 0)

        stack = list(_node_stack.get())
        for i in range(len(stack) - 1, -1, -1):
            name, step, obs, cm = stack[i]
            if name == node_name and step == run_step:
                stack.pop(i)
                _node_stack.set(stack)
                with suppress(Exception):
                    output = fields.get("output")
                    if output is not None:
                        obs.update(output=output)
                    cm.__exit__(None, None, None)
                return

    def _on_workflow_end(self, fields: dict[str, Any]) -> None:
        root = _workflow_obs.get()
        if root is None:
            return

        update: dict[str, Any] = {}
        if fields.get("output") is not None:
            update["output"] = fields["output"]
        metadata = _clean(
            {
                "steps": fields.get("steps"),
                "finish_reason": fields.get("finish_reason"),
                "max_steps": fields.get("max_steps"),
            }
        )
        if metadata:
            update["metadata"] = metadata
        with suppress(Exception):
            if update:
                root.update(**update)

        self._teardown()
        with suppress(Exception):
            self.langfuse.flush()

    def _on_exception(
        self, name: str, error: Exception, fields: dict[str, Any]
    ) -> None:
        error_meta = {"error_type": type(error).__name__}

        if name == "node.error":
            # Fatal node error: mark the node span and the root, then tear down.
            run_step = int(fields.get("run_step") or 0)
            for _, step, obs, _ in reversed(_node_stack.get()):
                if step == run_step:
                    with suppress(Exception):
                        obs.update(
                            level="ERROR",
                            status_message=str(error),
                            metadata=error_meta,
                        )
                    break

            root = _workflow_obs.get()
            if root is not None:
                with suppress(Exception):
                    root.update(
                        level="ERROR", status_message=str(error), metadata=error_meta
                    )

            self._teardown()
            with suppress(Exception):
                self.langfuse.flush()
        else:
            # Non-fatal step error: annotate the current open node span only.
            # The workflow keeps running — do not tear down.
            stack = _node_stack.get()
            if stack:
                _, _, current_obs, _ = stack[-1]
                with suppress(Exception):
                    current_obs.update(
                        level="ERROR",
                        status_message=f"{name}: {error}",
                        metadata=error_meta,
                    )

    def propagation_headers(self) -> dict[str, str]:
        carrier: dict[str, str] = {}
        _otel_inject(carrier)
        return carrier

    @contextmanager
    def attach_propagation_context(self, headers: dict[str, str]):
        ctx = _otel_extract(headers)
        token = _otel_context.attach(ctx)
        try:
            yield
        finally:
            _otel_context.detach(token)

    def _teardown(self) -> None:
        for _, _, _, cm in reversed(_node_stack.get()):
            with suppress(Exception):
                cm.__exit__(None, None, None)
        _node_stack.set([])

        cm = _workflow_cm.get()
        if cm is not None:
            with suppress(Exception):
                cm.__exit__(None, None, None)
        _workflow_obs.set(None)
        _workflow_cm.set(None)

        p_cm = _propagate_cm.get()
        if p_cm is not None:
            with suppress(Exception):
                p_cm.__exit__(None, None, None)
        _propagate_cm.set(None)
