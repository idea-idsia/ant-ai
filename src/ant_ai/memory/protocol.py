from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext


class Memory(BaseModel):
    """
    Pluggable memory interface for agents.

    Implement `retrieve` and `update` to connect any memory backend.
    Both methods take `ctx` explicitly — the same `InvocationContext`
    `Tool`/`ToolStep` inject into any tool method that declares a `ctx`
    parameter (see `MemoryTool`) — plus `**kwargs` for backend-specific
    parameters (e.g. mem0's `user_id`, `filters`, `run_id`) callers can pass
    when not going through `ctx`.
    """

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        ctx: InvocationContext | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        """Return relevant memories as Messages, ready to inject into agent state."""
        raise NotImplementedError

    async def update(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist new knowledge from the current session."""
        raise NotImplementedError
