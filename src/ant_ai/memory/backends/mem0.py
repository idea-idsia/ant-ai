from __future__ import annotations

from typing import Any

from mem0 import AsyncMemoryClient
from mem0.client.types import SearchMemoryOptions
from pydantic import Field, PrivateAttr, model_validator

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext
from ant_ai.tools.builtins.memory_tool import MemoryTool


def _resolve_scope(
    ctx: InvocationContext | None, kwargs: dict[str, Any]
) -> dict[str, str]:
    """Build a mem0 filters dict from `ctx`, or bare entity ids in kwargs.

    ctx.user_id  → user_id  (cross-session)
    ctx.session_id → run_id (session-scoped fallback)

    Raises:
        ValueError: If no scoping information (ctx or a bare
            user_id/run_id/agent_id/app_id) is available — without this,
            retrieve/update would hit mem0 unscoped, pooling memory across
            every user and session.
    """
    if ctx is not None:
        if ctx.user_id:
            return {"user_id": ctx.user_id}
        return {"run_id": ctx.session_id}
    filters: dict[str, str] = {}
    for key in ("user_id", "run_id", "agent_id", "app_id"):
        val = kwargs.pop(key, None)
        if val is not None:
            filters[key] = val
    if not filters:
        raise ValueError(
            "Mem0Memory requires scoping information (pass ctx=... with a "
            "session_id/user_id, or an explicit user_id/run_id/agent_id/"
            "app_id) to avoid pooling memory across every user and session."
        )
    return filters


class Mem0Memory(MemoryTool):
    """
    mem0 cloud backend for AgentMemory.

    Requires MEM0_API_KEY in the environment, or pass ``api_key`` explicitly.

    Example::

        memory = Mem0Memory(api_key="m0-...")
        msgs = await memory.retrieve("user preferences", user_id="alice")
        await memory.update(conversation, user_id="alice")
    """

    api_key: str | None = Field(default=None)
    _client: Any = PrivateAttr()

    @model_validator(mode="after")
    def _init_client(self) -> Mem0Memory:
        self._client: AsyncMemoryClient = (
            AsyncMemoryClient(api_key=self.api_key)
            if self.api_key
            else AsyncMemoryClient()
        )
        return self

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        ctx: InvocationContext | None = None,
        **kwargs: Any,
    ) -> list[Message]:
        filters = _resolve_scope(ctx, kwargs)
        options = SearchMemoryOptions(filters=filters or None, top_k=top_k)
        results: Any = await self._client.search(query, options=options)
        return [
            Message(role="system", content=r["memory"])
            for r in results.get("results", [])
        ]

    async def update(
        self,
        messages: list[Message],
        *,
        ctx: InvocationContext | None = None,
        **kwargs: Any,
    ) -> None:
        filters = _resolve_scope(ctx, kwargs)
        msg_dicts: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages if m.content
        ]
        if not msg_dicts:
            return
        await self._client.add(msg_dicts, **filters)
