from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from a2a.types import InternalError, Task
from a2a.utils.errors import A2AError

from ant_ai.core.events import BaseEvent, ContentDeltaEvent
from ant_ai.llm import ContextWindowExceededError
from ant_ai.observer import obs


@asynccontextmanager
async def observe_execution(
    agent_name: str, task: Task, metadata: dict[str, Any]
) -> AsyncGenerator[None]:
    """Trace the execution of `task` and turn its failures into A2A errors.

    A2A errors and cancellations pass through; any other exception becomes an
    `InternalError`, which carries the message only for a context-window overflow.
    """
    with (
        obs.attach_propagation_context(metadata),
        obs.bind(agent_name=agent_name, task_id=task.id, context_id=task.context_id),
    ):
        await obs.event("a2a.execute", task_id=task.id, context_id=task.context_id)
        try:
            yield
        except asyncio.CancelledError:
            await obs.event(
                "a2a.cancelled", task_id=task.id, context_id=task.context_id
            )
            raise
        except A2AError as exc:
            await obs.exception("a2a.error", exc)
            raise
        except ContextWindowExceededError as exc:
            # The message is the library's own, not the provider's, so it
            # can cross to the caller; the provider's text stays in the log.
            await obs.exception("a2a.error", exc)
            raise InternalError(str(exc)) from exc
        except Exception as exc:
            await obs.exception("a2a.error", exc)
            raise InternalError() from exc


async def observe_cancel(task: Task) -> None:
    await obs.event("a2a.cancel", task_id=task.id, context_id=task.context_id)


async def observe_workflow(
    events: AsyncIterator[BaseEvent],
) -> AsyncIterator[BaseEvent]:
    """Yield a workflow's events, tracing its start, each event but deltas, and its end."""
    await obs.event("a2a.workflow.start")
    async for event in events:
        if not isinstance(event, ContentDeltaEvent):
            await obs.event(
                "a2a.workflow.event",
                workflow_event=getattr(event, "kind", type(event).__name__),
                node=getattr(event.origin, "node", "-"),
                step=getattr(event.origin, "run_step", "-"),
            )
        yield event
    await obs.event("a2a.workflow.end")
