from __future__ import annotations

from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState
from google.protobuf import json_format as _json_format
from pydantic import TypeAdapter

from ant_ai.core.message import AnyMessage
from ant_ai.core.types import State

_CHECKPOINT_KEY = "compressionCheckpoint"

_any_message_adapter: TypeAdapter[AnyMessage] = TypeAdapter(AnyMessage)


def _get_checkpoint_data(msg: Any) -> Any:
    md = _json_format.MessageToDict(msg.metadata) if msg.metadata else {}
    return md.get(_CHECKPOINT_KEY)


def find_compression_checkpoint(
    related_tasks: list[Task],
) -> tuple[list[AnyMessage] | None, int]:
    """Return (baseline_messages, task_index) for the most-recent compression checkpoint.

    Scans all tasks in chronological order, keeping the last checkpoint found.
    Returns (None, 0) when no checkpoint exists in the chain.
    """
    result: list[AnyMessage] | None = None
    idx = 0
    for i, task in enumerate(related_tasks):
        for msg in task.history or []:
            cp = _get_checkpoint_data(msg)
            if cp:
                try:
                    result = [_any_message_adapter.validate_python(m) for m in cp]
                    idx = i
                except Exception:
                    pass
    return result, idx


def is_checkpoint_message(msg: Any) -> bool:
    """Return True if an A2A message is a synthetic compression checkpoint."""
    return bool(_get_checkpoint_data(msg))


async def persist_compression_checkpoint(state: State, updater: TaskUpdater) -> None:
    """Persist the compression baseline into the current A2A task history.

    Called just before the task is finalised. Writes a synthetic agent message
    with empty text and checkpoint metadata so future turns can restore the
    compressed baseline without replaying the full BFS history.
    Does nothing when compression did not fire this turn.
    """
    baseline = state._compression_context
    if baseline is None:
        return
    data = [_any_message_adapter.dump_python(m, mode="json") for m in baseline]
    msg = updater.new_agent_message(parts=[Part(text="")])
    msg.metadata.update({_CHECKPOINT_KEY: data})
    await updater.update_status(state=TaskState.TASK_STATE_WORKING, message=msg)
