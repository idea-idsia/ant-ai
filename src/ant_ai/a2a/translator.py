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
    ContentDeltaEvent,
    Event,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    StartEvent,
    ToolCallingEvent,
    ToolResultEvent,
    UpdateEvent,
)
from ant_ai.core.message import (
    AnyMessage,
    Message,
    ToolCallMessage,
    ToolCallResultMessage,
)

type Handler = Callable[[Event, TaskUpdater], Awaitable[None]]
type UnboundHandler = Callable[[HVEventToA2A, Event, TaskUpdater], Awaitable[None]]

_any_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def handler(*event_types: type[Event]):
    """
    Decorator used to mark methods as handlers for specific Event classes.
    """

    def decorator(fn: UnboundHandler):
        fn._event_types = event_types  # ty:ignore[unresolved-attribute]
        return fn

    return decorator


class HVEventToA2A:
    """
    Translator that converts internal HV Events to A2A updates by applying the appropriate handler based on the event class. Each handler is responsible for taking an Event and using the TaskUpdater to propagate the corresponding update to A2A.

    Instantiated once per A2AExecutor and shared across every concurrent task
    it serves, so this class must stay stateless: `is_first`/`stream_id`
    uniqueness comes from the emitting `LLMStep` (a fresh id per generation),
    never from per-artifact bookkeeping kept here.
    """

    def __init__(self, *, stream_artifacts: bool = True) -> None:
        """
        Initializes the translator and registers handlers. Adopting a single-dispatch like approach for translating Events to A2A updates, where handlers are registered via a decorator and stored in a mapping of event class to handler method.

        Args:
            stream_artifacts: Whether to translate ContentDeltaEvent into A2A
                TaskArtifactUpdateEvent chunks via `add_artifact`. When False,
                deltas are dropped and only the terminal whole-event message
                is sent. The terminal message is always sent either way, so
                this is purely additive for peers that read artifacts.
        """
        self._stream_artifacts = stream_artifacts
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
        msg = updater.new_agent_message(parts=[Part(text=event.content)])
        msg.metadata.update(metadata)
        # This whole-event message is the definitive close-out: it is sent
        # unconditionally so naive/non-streaming clients see no difference.
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=msg,
            metadata=metadata,
        )

        stream_id = getattr(event, "stream_id", None)
        if not self._stream_artifacts or not stream_id:
            return

        # Reasoning, content, and each tool call stream under their own
        # artifact id (see _content_delta) so a peer reading raw artifact
        # text never sees them concatenated. Which of these actually exist
        # is derivable from the event itself, with no tracking needed:
        # ReasoningEvent only fires when reasoning text was non-empty (i.e.
        # reasoning deltas -- and that artifact -- exist); content deltas
        # exist iff event.content is non-empty; tool calls always streamed
        # under their index if they're present at all (stream_id is only
        # set by LLMStep.stream(), never .run()). Tool calls appear here in
        # the same order _stream_deltas first saw their index (see
        # LLMStep.stream / _ToolCallFragment), so position == original index.
        artifact_ids: list[str] = []
        if isinstance(event, ReasoningEvent):
            artifact_ids.append(f"{stream_id}:reasoning")
        else:
            if event.content:
                artifact_ids.append(f"{stream_id}:content")
            tool_calls = getattr(event, "tool_calls", None) or []
            artifact_ids += [f"{stream_id}:tool:{i}" for i in range(len(tool_calls))]

        for artifact_id in artifact_ids:
            await updater.add_artifact(
                parts=[], artifact_id=artifact_id, append=True, last_chunk=True
            )

    @handler(ContentDeltaEvent)
    async def _content_delta(
        self, event: ContentDeltaEvent, updater: TaskUpdater
    ) -> None:
        if not self._stream_artifacts:
            return
        artifact_id = (
            f"{event.stream_id}:tool:{event.tool_call_index}"
            if event.tool_call_index is not None
            else f"{event.stream_id}:{event.target_kind}"
        )
        metadata: dict[str, Any] = A2AMetadata(event=event).model_dump()
        await updater.add_artifact(
            parts=[Part(text=event.delta)],
            artifact_id=artifact_id,
            append=not event.is_first,
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
        artifact = raw.artifact
        md: dict[str, Any] = (
            _json_format.MessageToDict(artifact.metadata) if artifact.metadata else {}
        )
        event = md.get("event")
        if event:
            event["task_id"] = raw.task_id
            event["session_id"] = raw.context_id
            return _any_event_adapter.validate_python(event)

        text = "".join(
            p.text for p in artifact.parts if p.WhichOneof("content") == "text"
        )
        if not text:
            return None
        # No structured `event` metadata key, e.g. a spec-compliant peer that
        # isn't this codebase — best-effort reconstruction as a content delta.
        return ContentDeltaEvent(
            delta=text,
            stream_id=artifact.artifact_id,
            task_id=raw.task_id,
            session_id=raw.context_id,
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

    def to_history_message(self, raw: A2AMessage) -> AnyMessage | None:
        """Convert an A2A history message to the appropriate internal message type.

        Uses embedded event metadata when available to reconstruct structured
        messages (ToolCallMessage, ToolCallResultMessage). Falls back to plain
        text when no metadata is present. Returns None for non-conversation
        events (ReasoningEvent, UpdateEvent, etc.) that carry no LLM context.
        """
        from a2a.helpers import get_message_text
        from a2a.types import Role

        event = self.translate(raw)
        if event is None:
            text = get_message_text(raw)
            if raw.role != Role.ROLE_AGENT:
                return Message(role="user", content=text)
            return Message(role="assistant", content=text) if text else None
        if isinstance(event, ToolCallingEvent):
            return ToolCallMessage(tool_calls=list(event.tool_calls))
        if isinstance(event, ToolResultEvent):
            return ToolCallResultMessage(
                name=event.name,
                tool_call_id=event.tool_call_id,
                content=event.content,
            )
        if isinstance(event, FinalAnswerEvent):
            return Message(role="assistant", content=event.content)
        return None  # non-conversation event (ReasoningEvent, UpdateEvent, etc.)
