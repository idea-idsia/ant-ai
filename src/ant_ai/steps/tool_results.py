from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ant_ai.core.events import ClarificationNeededEvent, ToolResultEvent
from ant_ai.core.result import (
    ClarificationNeededOutput,
    StepResult,
    ToolOutput,
    Transition,
    TransitionAction,
)

if TYPE_CHECKING:
    from ant_ai.core.events import BaseEvent
    from ant_ai.core.message import ToolCallResultMessage


class ToolResults:
    """Collects the outcomes of a `ToolStep`'s calls into its events and `StepResult`."""

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []
        self._clarifications: list[ClarificationNeededOutput] = []

    def add(
        self, outcome: ToolCallResultMessage | ClarificationNeededOutput
    ) -> ToolResultEvent | None:
        """Record `outcome`; return its `ToolResultEvent`, or None for a clarification."""
        if isinstance(outcome, ClarificationNeededOutput):
            self._clarifications.append(outcome)
            return None
        self._results.append(
            {
                "tool_call_id": outcome.tool_call_id,
                "name": outcome.name,
                "content": outcome.content,
                "is_error": outcome.is_error,
            }
        )
        return ToolResultEvent(
            content=outcome.content or "",
            tool_call_id=outcome.tool_call_id,
            name=outcome.name,
            is_error=outcome.is_error,
        )

    def answer_clarifications(self) -> list[BaseEvent]:
        """Answer each clarified call with its question and return the events."""
        # A clarified call is answered with its own question. Without a result
        # the assistant's `tool_calls` turn would be left dangling, and the
        # user's reply on resume would be rejected by chat-completion APIs.
        events: list[BaseEvent] = []
        for clarification in self._clarifications:
            self._results.append(
                {
                    "tool_call_id": clarification.tool_call_id,
                    "name": clarification.tool_name,
                    "content": clarification.question,
                    "is_error": False,
                }
            )
            events.append(
                ToolResultEvent(
                    content=clarification.question,
                    tool_call_id=clarification.tool_call_id,
                    name=clarification.tool_name,
                )
            )
            events.append(
                ClarificationNeededEvent(
                    content=clarification.question,
                    metadata={
                        "tool_call_id": clarification.tool_call_id,
                        "name": clarification.tool_name,
                    },
                )
            )
        return events

    def step_result(self) -> StepResult:
        """The `StepResult`: `END` if a call needs clarification, else back to the LLM."""
        return StepResult(
            output=ToolOutput(results=tuple(self._results)),
            transition=Transition(action=TransitionAction.END)
            if self._clarifications
            else Transition(action=TransitionAction.CONTINUE, next_step="llm"),
        )
