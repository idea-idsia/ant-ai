from __future__ import annotations

from ant_ai.tools.tool import Tool


class Memory(Tool):
    """
    Base class for an agent's long-term memory, kept across conversations.

    A memory is a tool namespace: its public methods are the tools the model
    calls to recall and save what it knows, registered as `<ClassName>_<method>`
    when the memory is set as `Agent(memory=...)`. Each backend picks the tools
    that fit its storage: `Mem0Memory` searches and adds facts, `FileMemory` reads
    and writes one Markdown file.

    Example:
        ```python
        from ant_ai.core.types import InvocationContext
        from ant_ai.memory import Memory


        class NotesMemory(Memory):
            notes: list[str] = []

            def recall(self, ctx: InvocationContext | None = None) -> list[str]:
                # The docstring of a method is the tool description the model reads.
                return self.notes

            def save(self, note: str, ctx: InvocationContext | None = None) -> str:
                self.notes.append(note)
                return "Saved."
        ```

    Notes:
        A method that declares `ctx` receives the current `InvocationContext`
        (e.g. `ctx.user_id`, to keep each user's memories apart); it is never
        shown to the model.
    """
