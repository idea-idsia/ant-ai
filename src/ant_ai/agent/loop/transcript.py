from __future__ import annotations

from typing import TYPE_CHECKING

from ant_ai.core.events import EventOrigin, FinalAnswerEvent, MaxStepsReachedEvent
from ant_ai.core.message import Message, ToolCallMessage, ToolCallResultMessage

if TYPE_CHECKING:
    from ant_ai.core.result import LLMOutput, ToolOutput
    from ant_ai.core.types import State


def record_tool_calls(state: State, output: LLMOutput) -> None:
    """Add the tool calls the model requested to the conversation."""
    state.add_message(ToolCallMessage(tool_calls=list(output.tool_calls)))


def record_tool_results(state: State, output: ToolOutput) -> None:
    """Add a result message for each tool call to the conversation."""
    for r in output.results:
        state.add_message(
            ToolCallResultMessage(
                name=r["name"],
                tool_call_id=r["tool_call_id"],
                content=r["content"],
                is_error=r.get("is_error", False),
            )
        )


def record_answer(state: State, content: str) -> None:
    """Add the final answer to the conversation."""
    state.add_message(Message(role="assistant", content=content))


def final_answer_event(
    content: str, loop_step: int, stream_id: str | None
) -> FinalAnswerEvent:
    """The `FinalAnswerEvent` of an answer given at `loop_step`."""
    return FinalAnswerEvent(
        origin=EventOrigin(layer="agent", run_step=loop_step),
        content=content,
        stream_id=stream_id,
    )


def max_steps_event(max_steps: int) -> MaxStepsReachedEvent:
    """The event of a run that used all `max_steps` without an answer."""
    return MaxStepsReachedEvent(
        origin=EventOrigin(layer="agent", run_step=max_steps),
    )
