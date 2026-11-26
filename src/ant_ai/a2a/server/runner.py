from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import Token

from a2a.server.tasks import TaskUpdater

from ant_ai.a2a.client.session import current_session_id
from ant_ai.a2a.server.compression import persist_compression_checkpoint
from ant_ai.a2a.server.observability import observe_workflow
from ant_ai.core.events import BaseEvent, ClarificationNeededEvent, CompletedEvent
from ant_ai.core.types import State

type EventSink = Callable[[BaseEvent, TaskUpdater], Awaitable[None]]


async def forward_events(
    events: AsyncIterator[BaseEvent],
    *,
    session_id: str,
    state: State,
    updater: TaskUpdater,
    apply: EventSink,
) -> None:
    """Apply a workflow's events to its A2A task, with `session_id` as the ambient session.

    A run that asked a question keeps its task `input-required`: its
    `CompletedEvent` is not applied, so the answer continues the task.
    """
    token: Token[str] = current_session_id.set(session_id)
    asked: bool = False
    try:
        async for event in observe_workflow(events):
            if isinstance(event, ClarificationNeededEvent):
                asked = True
            if isinstance(event, CompletedEvent):
                await persist_compression_checkpoint(state, updater)
                if asked:
                    continue
            await apply(event, updater)
    finally:
        current_session_id.reset(token)
