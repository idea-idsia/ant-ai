from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from pydantic import Field, PrivateAttr

from ant_ai.core._files import write_atomic
from ant_ai.core.types import InvocationContext
from ant_ai.memory.protocol import Memory
from ant_ai.tools.tool import ToolError

# Writes take turns, so the check that the memory is unchanged still holds when
# the file is replaced.
_WRITE_LOCK: threading.Lock = threading.Lock()
_EMPTY: str = "No memories saved yet."
# Conversations whose last read is remembered; an older one must read again.
_MAX_SESSIONS: int = 256


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _session(ctx: InvocationContext | None) -> str:
    return ctx.session_id if ctx is not None else ""


class FileMemory(Memory):
    """
    Long-term memory kept in one Markdown file, which the agent reads and writes whole.

    Unlike `Mem0Memory` it needs no service: the file can be read and edited by
    hand, kept in git, and used offline and in tests.

    Example:
        ```python
        from ant_ai.agent import Agent
        from ant_ai.memory.backends.file import FileMemory

        agent = Agent(memory=FileMemory(path="./agent_state/MEMORY.md"), ...)
        ```

        The agent can now call:

            - FileMemory_read()
            - FileMemory_write(content="# Memory\\n\\n- The user prefers short answers\\n")

    Notes:
        A write replaces the whole memory, so it is refused unless the
        conversation has read the memory as it is now: a model cannot erase what
        it has not seen, or undo a change made since it read, by another
        conversation or by hand. A write longer than `max_chars` is refused too,
        asking the model to shorten the memory.
        The file and its parent directories are created on the first write, and
        each write replaces the file at once, so a crash never leaves half of it.
        One file is shared by every conversation and user of the memory, whatever
        `ctx` says: give each user their own agent and path when their memories
        must stay apart.
        Failures raise `ToolError`, which `ToolStep` turns into a recoverable
        `ERROR: ...` message for the agent.
    """

    path: Path = Field(description="The memory file, e.g. `./MEMORY.md`.")
    max_chars: int | None = Field(
        default=10_000,
        ge=1,
        description="Longest memory `write` accepts, or None for no limit.",
    )
    # Per conversation, the digest of the memory as it last read or wrote it.
    _seen: dict[str, str] = PrivateAttr(default_factory=dict)

    def read(self, ctx: InvocationContext | None = None) -> str:
        """Read your long-term memory: what was saved in earlier conversations. Call it at the start of a conversation, and before answering anything that may depend on what you already know about the user or the task."""
        text: str = self._read()
        self._remember(ctx, text)
        return text if text.strip() else _EMPTY

    def write(self, content: str, ctx: InvocationContext | None = None) -> str:
        """Replace your whole long-term memory with `content` (Markdown); read it first. To save a new fact, write the memory with the fact added; to forget one, write it without. Save durable facts (preferences, decisions, explicit "remember this" requests) as soon as you learn them, and keep the memory short."""
        if self.max_chars is not None and len(content) > self.max_chars:
            raise ToolError(
                f"memory is {len(content)} characters, over the limit of {self.max_chars}: "
                "shorten it, merging or dropping the least useful entries, and write it again"
            )
        with _WRITE_LOCK:
            current: str = self._read()
            if current.strip() and self._seen.get(_session(ctx)) != _digest(current):
                raise ToolError(
                    "the memory has changed since you last read it: read it, then write it again with your changes"
                )
            self._write(content)
            self._remember(ctx, content)
        return "Memory saved." if content.strip() else "Memory cleared."

    def _remember(self, ctx: InvocationContext | None, text: str) -> None:
        session: str = _session(ctx)
        self._seen.pop(session, None)
        self._seen[session] = _digest(text)
        if len(self._seen) > _MAX_SESSIONS:
            del self._seen[next(iter(self._seen))]

    def _read(self) -> str:
        """The memory file as text, or "" if there is none yet."""
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except UnicodeDecodeError as e:
            raise ToolError(f"memory file is not UTF-8 text: {self.path}") from e
        except OSError as e:
            raise ToolError(f"{e.strerror or e}: {self.path}") from e

    def _write(self, text: str) -> None:
        try:
            write_atomic(self.path, text)
        except OSError as e:
            raise ToolError(f"{e.strerror or e}: {self.path}") from e
