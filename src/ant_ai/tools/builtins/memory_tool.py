from __future__ import annotations

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext
from ant_ai.memory.protocol import Memory
from ant_ai.tools.tool import Tool


class MemoryTool(Memory, Tool):
    """
    Base class for long-term memory backends directly usable as an agent Tool.

    Subclass and implement `retrieve`/`update` (the `Memory` protocol,
    unchanged) to connect a backend — `search`/`add` below are the
    LLM-facing tools built automatically on top of it, registered as
    `<ClassName>_search` / `<ClassName>_add`.

    Example:
        ```python
        from ant_ai.agent import Agent
        from ant_ai.memory.backends.mem0 import Mem0Memory

        agent = Agent(memory=Mem0Memory(), ...)
        ```

        The agent gets `Mem0Memory_search`/`Mem0Memory_add` automatically —
        the LLM decides when to call them.

    Notes:
        `ctx` is injected automatically by `ToolStep` from the current
        `InvocationContext` and is never exposed to the LLM.
    """

    async def search(
        self, query: str, ctx: InvocationContext | None = None
    ) -> list[str]:
        """Search long-term memory for facts relevant to `query`. Call this
        whenever recalling something about the user or a past conversation
        would help answer the current request."""
        messages: list[Message] = await self.retrieve(query, ctx=ctx)
        return [m.content for m in messages if m.content]

    async def add(self, facts: list[str], ctx: InvocationContext | None = None) -> str:
        """Persist one or more durable facts (user preferences, personal
        details, explicit "remember this" instructions) for future
        conversations. Call this proactively as soon as you learn something
        worth keeping — do not wait to be asked, and do not wait until the
        end of the conversation."""
        if not facts:
            return "Nothing to save."
        await self.update([Message(role="system", content=f) for f in facts], ctx=ctx)
        return f"Saved {len(facts)} fact(s) to memory."
