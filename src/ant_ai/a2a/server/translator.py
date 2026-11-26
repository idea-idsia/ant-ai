from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import singledispatchmethod
from typing import Any

from a2a.helpers import get_message_text
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message as A2AMessage,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from google.protobuf import json_format as _json_format
from pydantic import TypeAdapter

from ant_ai.a2a.server.types import A2AMetadata
from ant_ai.core.events import (
    AnyEvent,
    BaseEvent,
    ClarificationNeededEvent,
    CompletedEvent,
    ContentDeltaEvent,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    StartEvent,
    ToolCallingEvent,
    ToolResultEvent,
    UpdateEvent,
    event_to_message,
)
from ant_ai.core.message import AnyMessage, Message

type Handler = Callable[[BaseEvent, TaskUpdater], Awaitable[None]]
type UnboundHandler = Callable[[HVEventToA2A, BaseEvent, TaskUpdater], Awaitable[None]]

_any_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def handler(*event_types: type[BaseEvent]):
    """
    Decorator used to mark methods as handlers for specific Event classes.
    """

    def decorator(fn: UnboundHandler):
        fn._event_types = event_types  # ty:ignore[missing-attribute]
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
        self._handlers: dict[type[BaseEvent], Handler] = {}
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

    async def apply(self, event: BaseEvent, updater: TaskUpdater) -> None:
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
    async def _start(self, event: BaseEvent, updater: TaskUpdater) -> None:
        await updater.start_work()

    @handler(UpdateEvent, MaxStepsReachedEvent)
    async def _update(
        self, event: UpdateEvent | MaxStepsReachedEvent, updater: TaskUpdater
    ) -> None:
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
    async def _agent_message(
        self,
        event: ToolCallingEvent | ToolResultEvent | FinalAnswerEvent | ReasoningEvent,
        updater: TaskUpdater,
    ) -> None:
        metadata: dict[str, Any] = A2AMetadata(event=event).model_dump()
        msg = updater.new_agent_message(parts=[Part(text=event.content)])
        msg.metadata.update(metadata)

        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=msg,
            metadata=metadata,
        )

        stream_id = getattr(event, "stream_id", None)
        if not self._stream_artifacts or not stream_id:
            return

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
    async def _input_required(
        self, event: ClarificationNeededEvent, updater: TaskUpdater
    ) -> None:
        metadata: dict[str, Any] = A2AMetadata(event=event).model_dump()
        msg = updater.new_agent_message(parts=[Part(text=event.content)])
        msg.metadata.update(metadata)
        await updater.update_status(
            state=TaskState.TASK_STATE_INPUT_REQUIRED,
            message=msg,
            metadata=metadata,
        )

    @handler(CompletedEvent)
    async def _completed(self, event: BaseEvent, updater: TaskUpdater) -> None:
        await updater.complete()


class A2AToHVEvent:
    """
    Translator that converts A2A messages and events into internal HV Events. Uses singledispatchmethod to define translation logic for different input types, allowing for flexible handling of various A2A message and event formats.
    """

    @singledispatchmethod
    def translate(self, raw: Any) -> BaseEvent | None:
        return None

    @translate.register
    def _(self, raw: A2AMessage) -> BaseEvent | None:
        return _metadata_event(raw.metadata, raw.task_id, raw.context_id)

    @translate.register
    def _(self, raw: TaskStatusUpdateEvent) -> BaseEvent | None:
        return _metadata_event(raw.metadata, raw.task_id, raw.context_id)

    @translate.register
    def _(self, raw: TaskArtifactUpdateEvent) -> BaseEvent | None:
        artifact = raw.artifact
        event: BaseEvent | None = _metadata_event(
            artifact.metadata, raw.task_id, raw.context_id
        )
        if event is not None:
            return event

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
    def _(self, raw: Task) -> BaseEvent | None:
        return _metadata_event(raw.metadata, raw.id, raw.context_id)

    def task_events(
        self, task: Task, *, seen: set[str] | None = None
    ) -> list[BaseEvent]:
        """The events of a whole task, one per agent message, in order.

        A non-streaming call answers with the finished `Task` only: its agent's
        events (tool calls, results, the final answer, a question) are the
        messages in its history and its status, not its own metadata, which
        holds just the last status update (e.g. "Completed action ..."). The
        user's messages are skipped, and so are those whose ids are in `seen`
        (already received while streaming); the ids returned are added to it.

        Args:
            task: The task as the server returned it.
            seen: Message ids already translated, to skip.

        Returns:
            The events, oldest first.
        """
        events: list[BaseEvent] = []
        for msg in _task_messages(task):
            if msg.role != Role.ROLE_AGENT:
                continue
            if seen is not None:
                if msg.message_id in seen:
                    continue
                seen.add(msg.message_id)
            event: BaseEvent | None = self.translate(msg)
            if event is None:
                continue
            event.task_id = event.task_id or task.id
            event.session_id = event.session_id or task.context_id
            events.append(event)
        return events

    def has_events(self, task: Task) -> bool:
        """Whether any of the task's messages carries ant-ai event metadata."""
        return any(self.translate(m) is not None for m in _task_messages(task))

    @staticmethod
    def task_text(task: Task) -> str:
        """The text a task answered with, for a peer that sends no event metadata:
        its artifacts' text, else its last agent message's."""
        text = "".join(
            p.text
            for artifact in task.artifacts
            for p in artifact.parts
            if p.WhichOneof("content") == "text"
        )
        if text:
            return text
        agent = [m for m in _task_messages(task) if m.role == Role.ROLE_AGENT]
        return get_message_text(agent[-1]) if agent else ""

    def to_history_message(self, raw: A2AMessage) -> AnyMessage | None:
        """Convert an A2A history message to the appropriate internal message type.

        Uses embedded event metadata when available to reconstruct structured
        messages (ToolCallMessage, ToolCallResultMessage). Falls back to plain
        text when no metadata is present. Returns None for non-conversation
        events (ReasoningEvent, UpdateEvent, etc.) that carry no LLM context.
        """
        event = self.translate(raw)
        if event is None:
            text = get_message_text(raw)
            if raw.role != Role.ROLE_AGENT:
                return Message(role="user", content=text)
            return Message(role="assistant", content=text) if text else None
        return event_to_message(event)


def _metadata_event(metadata: Any, task_id: str, session_id: str) -> BaseEvent | None:
    """The ant-ai event stored under the `event` key of A2A `metadata`, if any."""
    md: dict[str, Any] = _json_format.MessageToDict(metadata) if metadata else {}
    event = md.get("event")
    if not event:
        return None
    event["task_id"] = task_id
    event["session_id"] = session_id
    return _any_event_adapter.validate_python(event)


def _task_messages(task: Task) -> list[A2AMessage]:
    """A task's history followed by its status message, if it has one."""
    messages: list[A2AMessage] = list(task.history)
    if task.status.HasField("message"):
        messages.append(task.status.message)
    return messages
