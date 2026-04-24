from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import singledispatchmethod
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message as A2AMessage,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from google.protobuf import json_format as _json_format
from pydantic import TypeAdapter

from ant_ai.a2a.types import A2AMetadata
from ant_ai.core.events import (
    AnyEvent,
    ClarificationNeededEvent,
    CompletedEvent,
    Event,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    StartEvent,
    ToolCallingEvent,
    ToolResultEvent,
    UpdateEvent,
)

type Handler = Callable[[Event, TaskUpdater], Awaitable[None]]
type UnboundHandler = Callable[[HVEventToA2A, Event, TaskUpdater], Awaitable[None]]

_any_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def handler(*event_types: type[Event]):
    """
    Decorator used to mark methods as handlers for specific Event classes.
    """

    def decorator(fn: UnboundHandler):
        fn._event_types: tuple[type[Event], ...] = event_types
        return fn

    return decorator


class HVEventToA2A:
    """
    Translator that converts internal HV Events to A2A updates by applying the appropriate handler based on the event class. Each handler is responsible for taking an Event and using the TaskUpdater to propagate the corresponding update to A2A.
    """

    def __init__(self) -> None:
        """
        Initializes the translator and registers handlers. Adopting a single-dispatch like approach for translating Events to A2A updates, where handlers are registered via a decorator and stored in a mapping of event class to handler method.
        """
        self._handlers: dict[type[Event], Handler] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """
        Scan instance methods and register decorated handlers.
        """
        for attr_name in dir(self):
            method = getattr(self, attr_name)
            types = getattr(method, "_event_types", None)
            if types:
                for t in types:
                    self._handlers[t] = method

    async def apply(self, event: Event, updater: TaskUpdater) -> None:
        """Applies the appropriate handler for the given Event based on its class, using the TaskUpdater to propagate updates to A2A. This method serves as the main entry point for translating Events to A2A updates, abstracting away the specific handling logic into separate methods for each event class.

        Args:
            event: The internal HV Event to be translated.
            updater: The A2A TaskUpdater instance used to propagate the translated event.

        Raises:
            ValueError: If no handler is registered for the class of the given event.
        """
        event_handler: Handler | None = self._handlers.get(type(event))
        if not event_handler:
            raise ValueError(
                f"No handler registered for event type: {type(event).__name__}"
            )

        await event_handler(event, updater)

    @handler(StartEvent)
    async def _start(self, event: Event, updater: TaskUpdater) -> None:
        await updater.start_work()

    @handler(UpdateEvent, MaxStepsReachedEvent)
    async def _update(self, event: Event, updater: TaskUpdater) -> None:
        metadata: dict[str, Any] = A2AMetadata(event=event).model_dump()
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            metadata=metadata,
        )

    @handler(
        ToolCallingEvent,
        ToolResultEvent,
        FinalAnswerEvent,
        ReasoningEvent,
    )
    async def _agent_message(self, event: Event, updater: TaskUpdater) -> None:
        metadata: dict[str, Any] = A2AMetadata(event=event).model_dump()
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=updater.new_agent_message(parts=[Part(text=event.content)]),
            metadata=metadata,
        )

    @handler(ClarificationNeededEvent)
    async def _input_required(self, event: Event, updater: TaskUpdater) -> None:
        await updater.requires_input(
            message=updater.new_agent_message(parts=[Part(text=event.content)]),
        )

    @handler(CompletedEvent)
    async def _completed(self, event: Event, updater: TaskUpdater) -> None:
        await updater.complete()


class A2AToHVEvent:
    """
    Translator that converts A2A messages and events into internal HV Events. Uses singledispatchmethod to define translation logic for different input types, allowing for flexible handling of various A2A message and event formats.
    """

    @singledispatchmethod
    def translate(self, raw: Any) -> Event | None:
        return None

    @translate.register
    def _(self, raw: A2AMessage) -> Event | None:
        if not raw.metadata:
            return None
        md: dict[str, Any] = _json_format.MessageToDict(raw.metadata)
        event = md.get("event")
        if not event:
            return None
        event["task_id"] = raw.task_id
        event["session_id"] = raw.context_id
        return _any_event_adapter.validate_python(event)

    @translate.register
    def _(self, raw: TaskStatusUpdateEvent) -> Event | None:
        if not raw.metadata:
            return None
        md: dict[str, Any] = _json_format.MessageToDict(raw.metadata)
        event = md.get("event")
        if not event:
            return None
        event["task_id"] = raw.task_id
        event["session_id"] = raw.context_id
        return _any_event_adapter.validate_python(event)

    @translate.register
    def _(self, raw: TaskArtifactUpdateEvent) -> Event | None:
        raise NotImplementedError(
            "Artifact updates are not yet supported for translation to Events yet"
        )

    @translate.register
    def _(self, raw: Task) -> Event | None:
        if not raw.metadata:
            return None
        md: dict[str, Any] = _json_format.MessageToDict(raw.metadata)
        event = md.get("event")
        if not event:
            return None
        event["task_id"] = raw.id
        event["session_id"] = raw.context_id
        return _any_event_adapter.validate_python(event)
