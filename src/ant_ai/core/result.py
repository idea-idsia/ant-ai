from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ant_ai.core.message import ToolCall


class TransitionAction(StrEnum):
    """Signal returned by a step telling the executor how to proceed."""

    CONTINUE = "continue"
    END = "end"


class Transition(BaseModel):
    """Routing instruction attached to every `StepResult`.

    `LLMStep` emits `CONTINUE, next_step="tool"` when the model requested
    tool calls, or `END` when it produced a final text answer. `ToolStep`
    emits `CONTINUE, next_step="llm"` after executing tools, or `END` when
    a tool signalled that human clarification is needed.
    """

    model_config = ConfigDict(frozen=True)

    action: TransitionAction = Field(
        default=TransitionAction.CONTINUE,
        description="Route execution to next_step on CONTINUE, or exit the loop on END.",
    )
    next_step: str | None = Field(
        default=None,
        description="Name of the registered step to run next. Only used when action is CONTINUE.",
    )


class LLMOutput(BaseModel):
    """Output produced by a single model call inside `LLMStep`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["llm"] = "llm"

    raw: str = Field(
        description="Raw text or JSON string as returned by the model.",
    )
    tool_calls: tuple[ToolCall, ...] = Field(
        default=(),
        description="Tool calls requested by the model. Empty when the model produced a final text answer.",
    )

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ToolOutput(BaseModel):
    """Output produced by `ToolStep` after executing one or more tool calls."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool"] = "tool"

    results: tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Serialized tool results, each with tool_call_id, name, and content keys.",
    )


class ClarificationNeededOutput(BaseModel):
    """Signals that a tool needs human input before execution can continue.

    Raised inside `ToolStep` when a tool returns a clarification request
    (e.g. via `HumanInputNeededTool.ask()`). The react loop returns this to
    its caller immediately, pausing the agent until the user answers.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["human"] = "human"

    question: str = Field(
        description="The question to present to the user.",
    )
    tool_call_id: str = Field(
        default="",
        description="ID of the tool call that triggered the clarification request.",
    )
    tool_name: str = Field(
        default="",
        description="Name of the tool that triggered the clarification request.",
    )


type StepOutput = Annotated[
    LLMOutput | ToolOutput | ClarificationNeededOutput,
    Field(discriminator="kind"),
]


class StepResult(BaseModel):
    """The immutable result of running a single `Step`.

    No references to `State` or any mutable object. State for the next
    iteration is passed into the executor, never stored here.
    """

    model_config = ConfigDict(frozen=True)

    output: StepOutput = Field(
        description="What the step produced. Use isinstance against LLMOutput, ToolOutput, or ClarificationNeededOutput before accessing subtype-specific fields.",
    )
    transition: Transition = Field(
        default_factory=Transition,
        description="Where to go next. The loop exits on END, or runs transition.next_step on CONTINUE.",
    )
