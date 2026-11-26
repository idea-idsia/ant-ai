from __future__ import annotations

from a2a.types import Message as A2AMessage, Task

from ant_ai.a2a.server.compression import (
    find_compression_checkpoint,
    is_checkpoint_message,
)
from ant_ai.a2a.server.translator import A2AToHVEvent
from ant_ai.core.message import (
    BaseMessage,
    Message,
    ToolCallMessage,
    ToolCallResultMessage,
)
from ant_ai.observer import obs

_UNANSWERED_TOOL_CALL = (
    "The run was stopped before this tool returned; it did not complete."
)


class HistoryBuilder:
    """Rebuilds the conversation a request continues from the A2A tasks it relates to."""

    def __init__(self) -> None:
        self._a2a_to_hv: A2AToHVEvent = A2AToHVEvent()

    async def build(
        self, related_tasks: list[Task], user_input: str
    ) -> list[BaseMessage]:
        """The history of `related_tasks`, from the latest compression checkpoint
        if there is one, followed by `user_input` as the user's message."""
        checkpoint, cp_idx = find_compression_checkpoint(related_tasks)
        if checkpoint is not None:
            post_a2a = [
                m
                for r_task in related_tasks[cp_idx:]
                for m in r_task.history or []
                if not is_checkpoint_message(m)
            ]
            history = list(checkpoint) + self.convert(post_a2a)
        else:
            history = self.convert(
                [m for r_task in related_tasks for m in r_task.history or []]
            )
        history.append(Message(role="user", content=user_input))
        await obs.event("a2a.history", history_messages=len(history))
        return history

    def convert(self, a2a_history: list[A2AMessage]) -> list[BaseMessage]:
        """Convert A2A history messages to internal messages, dropping those
        that carry no LLM context."""
        return self.answer_dangling_tool_calls(
            [
                m
                for msg in a2a_history
                if (m := self._a2a_to_hv.to_history_message(msg)) is not None
            ]
        )

    @staticmethod
    def answer_dangling_tool_calls(
        history: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Give every unanswered tool call a result, so the transcript replays.

        A task cancelled (or crashed) while its tools ran ends with the
        assistant's `tool_calls` turn and nothing after it: the run cannot
        emit results once the terminal status is on the queue. Chat-completion
        APIs reject a transcript where such a turn is followed by anything but
        its tool messages, which made the next turn in the same context fail.
        The synthetic results say what happened and are flagged as errors, so
        the model knows the work was not done.
        """
        repaired: list[BaseMessage] = []
        pending: dict[str, str] = {}  # call id -> tool name, unanswered so far

        def flush() -> None:
            repaired.extend(
                ToolCallResultMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=_UNANSWERED_TOOL_CALL,
                    is_error=True,
                )
                for call_id, name in pending.items()
            )
            pending.clear()

        for m in history:
            if isinstance(m, ToolCallResultMessage):
                pending.pop(m.tool_call_id, None)
            elif pending:
                flush()
            if isinstance(m, ToolCallMessage):
                pending.update((tc.id, tc.function.name) for tc in m.tool_calls)
            repaired.append(m)
        flush()
        return repaired


def without_message(task: Task, message_id: str) -> Task:
    """A copy of `task` whose history leaves out the message `message_id`.

    The SDK adds the incoming message to the history of the task it continues;
    `HistoryBuilder.build` adds it again as the user input, so it is left out here.
    """
    copy = Task()
    copy.CopyFrom(task)
    del copy.history[:]
    copy.history.extend(m for m in task.history if m.message_id != message_id)
    return copy
