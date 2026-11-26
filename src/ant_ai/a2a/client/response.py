from __future__ import annotations

from a2a.types import Message, StreamResponse, TaskState
from a2a.types.a2a_pb2 import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent

from ant_ai.a2a.server.translator import A2AToHVEvent
from ant_ai.core.events import BaseEvent, FinalAnswerEvent

type _Update = Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent


class ResponseReader:
    """Turns the responses to one sent message into events, each agent message once."""

    def __init__(self) -> None:
        self._translator: A2AToHVEvent = A2AToHVEvent()
        # Agent messages already received, by id: a task sent later in a
        # stream repeats them in its history.
        self._seen: set[str] = set()
        self._answered: bool = False

    def read(self, chunk: StreamResponse) -> list[BaseEvent]:
        """The events a response carries that were not received before."""
        if chunk.HasField("task"):
            # A whole task (all a non-streaming call gets): its events are
            # its agent messages, not its metadata (the last status update).
            events: list[BaseEvent] = self._task_events(chunk.task)
        else:
            events = self._update_events(chunk)
        self._answered = self._answered or any(
            isinstance(ev, FinalAnswerEvent) for ev in events
        )
        return events

    def _update_events(self, chunk: StreamResponse) -> list[BaseEvent]:
        raw: _Update | None = self._update(chunk)
        if raw is None:
            return []
        ev: BaseEvent | None = self._translator.translate(raw)
        return [ev] if ev is not None else []

    def _update(self, chunk: StreamResponse) -> _Update | None:
        if chunk.HasField("status_update"):
            status: TaskStatusUpdateEvent = chunk.status_update
            if status.status.HasField("message"):
                self._seen.add(status.status.message.message_id)
            return status
        if chunk.HasField("message"):
            self._seen.add(chunk.message.message_id)
            return chunk.message
        if chunk.HasField("artifact_update"):
            return chunk.artifact_update
        return None

    def _task_events(self, task: Task) -> list[BaseEvent]:
        """A task's events not yet received, plus its answer as a final answer when
        it completed without one (a peer that sends no ant-ai event metadata)."""
        events: list[BaseEvent] = self._translator.task_events(task, seen=self._seen)
        completed: bool = task.status.state == TaskState.TASK_STATE_COMPLETED
        if (
            completed
            and not self._answered
            and not any(isinstance(e, FinalAnswerEvent) for e in events)
            and not self._translator.has_events(task)
        ):
            text: str = self._translator.task_text(task)
            if text:
                events.append(
                    FinalAnswerEvent(
                        content=text, task_id=task.id, session_id=task.context_id
                    )
                )
        return events
