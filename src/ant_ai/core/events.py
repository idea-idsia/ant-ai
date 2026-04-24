from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ant_ai.core.message import AnyMessage

type EventSource = Literal["agent", "action", "workflow"]
"""Sources where events are generated during a run."""


class EventOrigin(BaseModel):
    """Describes the origin of an event, used for tracing back to the source of an event in the system."""

    layer: EventSource = Field(
        default="agent",
        description="The layer that emitted the event: agent, action, or workflow.",
    )
    node: str | None = Field(
        default=None,
        description="Name of the workflow node where the event originated.",
    )
    run_step: int = Field(
        default=0,
        description="Step index within the current run.",
    )


class Event(BaseModel):
    """Represents an event emitted during the execution of a workflow, action, or agent."""

    origin: EventOrigin = Field(
        default_factory=EventOrigin,
        description="Tracing information identifying where in the system the event was emitted.",
    )
    content: str = Field(
        default="",
        description="Textual description of the event.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information relevant to the event.",
    )
    message: AnyMessage | None = Field(
        default=None,
        description="Message associated with the event. Only set for events tied to conversation messages.",
    )
    kind: Literal["event"] = "event"
    task_id: str | None = Field(
        default=None,
        description="A2A task ID populated from the raw transport event.",
    )
    session_id: str | None = Field(
        default=None,
        description="A2A context/session ID populated from the raw transport event.",
    )


class AgentEvent(Event):
    """Base class for events emitted by the agent layer."""

    origin: EventOrigin = Field(
        default_factory=lambda: EventOrigin(layer="agent"),
        description="Tracing information identifying where in the system the event was emitted.",
    )


class ActionEvent(Event):
    """Base class for events emitted by the action layer."""

    origin: EventOrigin = Field(
        default_factory=lambda: EventOrigin(layer="action"),
        description="Tracing information identifying where in the system the event was emitted.",
    )


class WorkflowEvent(Event):
    """Base class for events emitted by the workflow layer."""

    origin: EventOrigin = Field(
        default_factory=lambda: EventOrigin(layer="workflow"),
        description="Tracing information identifying where in the system the event was emitted.",
    )


class StartEvent(WorkflowEvent):
    """Emitted when a workflow begins execution."""

    kind: Literal["start"] = "start"


class FinalAnswerEvent(AgentEvent):
    """Emitted when the agent produces its final answer."""

    kind: Literal["final_answer"] = "final_answer"


class MaxStepsReachedEvent(AgentEvent):
    """Emitted when the agent exhausts its maximum allowed steps."""

    kind: Literal["max_steps_reached"] = "max_steps_reached"


class ClarificationNeededEvent(AgentEvent):
    """Emitted when the agent requires human input to continue."""

    kind: Literal["input_required"] = "input_required"


class UpdateEvent(WorkflowEvent):
    """Emitted for intermediate status updates during execution."""

    kind: Literal["update"] = "update"


class ToolCallingEvent(AgentEvent):
    """Emitted when the agent decides to call one or more tools."""

    kind: Literal["tool_calling"] = "tool_calling"


class ToolResultEvent(AgentEvent):
    """Emitted when a tool call completes and its result is available."""

    kind: Literal["tool_result"] = "tool_result"


class ReasoningEvent(AgentEvent):
    """Emitted when the model produces reasoning/thinking content before its answer."""

    kind: Literal["reasoning"] = "reasoning"


class CompletedEvent(WorkflowEvent):
    """Emitted when a workflow completes successfully."""

    kind: Literal["completed"] = "completed"


type AnyEvent = Annotated[
    Event
    | StartEvent
    | FinalAnswerEvent
    | MaxStepsReachedEvent
    | ClarificationNeededEvent
    | UpdateEvent
    | ToolCallingEvent
    | ToolResultEvent
    | ReasoningEvent
    | CompletedEvent,
    Field(discriminator="kind"),
]
