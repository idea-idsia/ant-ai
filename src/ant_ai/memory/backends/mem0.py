from __future__ import annotations

from typing import Any

from mem0 import AsyncMemory
from pydantic import PrivateAttr

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext
from ant_ai.memory.protocol import Memory


def _extract_run_id(kwargs: dict[str, Any]) -> str | None:
    """Pop ctx (or run_id) from kwargs and return the resolved run_id."""
    ctx: InvocationContext | None = kwargs.pop("ctx", None)
    if ctx is not None:
        return ctx.session_id
    return kwargs.pop("run_id", None)


class Mem0Memory(Memory):
    """
    mem0 backend for AgentMemory.

    Entity identifiers (user_id, agent_id, run_id, app_id) are NOT stored
    on the instance — pass them at call time via **kwargs, exactly as mem0's
    own API expects.

    Example::

        memory = Mem0Memory()
        msgs = await memory.retrieve("user preferences", filters={"user_id": "alice"})
        await memory.update(conversation, user_id="alice", run_id="session-42")
    """

    _client: Any = PrivateAttr(default=AsyncMemory)

    async def retrieve(
        self, query: str, *, top_k: int = 5, **kwargs: Any
    ) -> list[Message]:
        """
        Search mem0 and return results as system Messages.

        Pass entity filters via kwargs, e.g.::

            await memory.retrieve(query, filters={"user_id": "alice"})
        """
        run_id: str | None = _extract_run_id(kwargs)
        if run_id is not None:
            kwargs["run_id"] = run_id
        results: Any = await self._client.search(query, top_k=top_k, **kwargs)
        return [
            Message(role="system", content=r["memory"])
            for r in results.get("results", [])
        ]

    async def update(self, messages: list[Message], **kwargs: Any) -> None:
        """
        Add the conversation to mem0 for memory extraction.

        Pass entity identifiers via kwargs, e.g.::

            await memory.update(messages, user_id="alice", run_id="session-42")
        """
        run_id: str | None = _extract_run_id(kwargs)
        if run_id is not None:
            kwargs["run_id"] = run_id
        msg_dicts: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages if m.content
        ]
        if not msg_dicts:
            return
        await self._client.add(msg_dicts, **kwargs)
