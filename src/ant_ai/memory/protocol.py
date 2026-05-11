from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ant_ai.core.message import Message


class Memory(BaseModel):
    """
    Pluggable memory interface for agents.

    Implement `retrieve` and `update` to connect any memory backend.
    Both methods accept `**kwargs` so callers can pass backend-specific
    parameters (e.g. mem0's `user_id`, `filters`, `run_id`) without
    changing the interface.
    """

    async def retrieve(
        self, query: str, *, top_k: int = 5, **kwargs: Any
    ) -> list[Message]:
        """Return relevant memories as Messages, ready to inject into agent state."""
        raise NotImplementedError

    async def update(self, messages: list[Message], **kwargs: Any) -> None:
        """Persist new knowledge from the current session."""
        raise NotImplementedError
