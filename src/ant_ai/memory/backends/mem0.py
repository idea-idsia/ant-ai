from __future__ import annotations

from typing import Any

from mem0 import AsyncMemoryClient
from mem0.client.types import SearchMemoryOptions
from pydantic import Field, PrivateAttr, model_validator

from ant_ai.core.types import InvocationContext
from ant_ai.memory.protocol import Memory


def _resolve_scope(ctx: InvocationContext | None) -> dict[str, str]:
    """Build a mem0 filters dict from `ctx`.

    ctx.user_id  → user_id  (cross-session)
    ctx.session_id → run_id (session-scoped fallback)

    Raises:
        ValueError: If there is no `ctx` — without it, search/add would hit
            mem0 unscoped, pooling memory across every user and session.
    """
    if ctx is None:
        raise ValueError(
            "Mem0Memory requires scoping information (pass ctx=... with a "
            "session_id/user_id) to avoid pooling memory across every user "
            "and session."
        )
    if ctx.user_id:
        return {"user_id": ctx.user_id}
    return {"run_id": ctx.session_id}


class Mem0Memory(Memory):
    """
    Long-term memory kept by the mem0 cloud service, searched by meaning.

    Requires MEM0_API_KEY in the environment, or pass ``api_key`` explicitly.
    Memories are kept per `ctx.user_id`, or per `ctx.session_id` when there is no
    user.

    Example:
        ```python
        from ant_ai.agent import Agent
        from ant_ai.memory.backends.mem0 import Mem0Memory

        agent = Agent(memory=Mem0Memory(), ...)
        ```

        The agent can now call:

            - Mem0Memory_search(query="user preferences")
            - Mem0Memory_add(facts=["The user prefers answers in Italian"])
    """

    api_key: str | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, description="Most memories one search returns.")
    _client: Any = PrivateAttr()

    @model_validator(mode="after")
    def _init_client(self) -> Mem0Memory:
        self._client: AsyncMemoryClient = (
            AsyncMemoryClient(api_key=self.api_key)
            if self.api_key
            else AsyncMemoryClient()
        )
        return self

    async def search(
        self, query: str, ctx: InvocationContext | None = None
    ) -> list[str]:
        """Search long-term memory for facts relevant to `query`. Call this
        whenever recalling something about the user or a past conversation
        would help answer the current request."""
        filters: dict[str, str] = _resolve_scope(ctx)
        options = SearchMemoryOptions(filters=filters, top_k=self.top_k)
        results: Any = await self._client.search(query, options=options)
        return [r["memory"] for r in results.get("results", []) if r.get("memory")]

    async def add(self, facts: list[str], ctx: InvocationContext | None = None) -> str:
        """Persist one or more durable facts (user preferences, personal
        details, explicit "remember this" instructions) for future
        conversations. Call this proactively as soon as you learn something
        worth keeping — do not wait to be asked, and do not wait until the
        end of the conversation."""
        facts = [f for f in facts if f]
        if not facts:
            return "Nothing to save."
        filters: dict[str, str] = _resolve_scope(ctx)
        await self._client.add(
            [{"role": "system", "content": f} for f in facts], **filters
        )
        return f"Saved {len(facts)} fact(s) to memory."
