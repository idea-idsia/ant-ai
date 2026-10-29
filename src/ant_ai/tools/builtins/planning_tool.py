from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ant_ai.core.types import InvocationContext
from ant_ai.tools.tool import Tool, ToolError

_MARK: dict[str, str] = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_UNSAFE: re.Pattern[str] = re.compile(r"[^A-Za-z0-9_-]")
_MAX_NAME: int = 64


class Todo(BaseModel):
    """One item of an agent's plan."""

    content: str = Field(description="What has to be done.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending", description="Where the item stands."
    )


def render_plan(todos: list[Todo]) -> str:
    """The plan as a checklist, one item per line."""
    return "\n".join(f"- {_MARK[t.status]} {t.content}" for t in todos)


def _file_name(text: str) -> str:
    # Readable, and inside the plans folder: anything but letters, digits, - and _
    # becomes _. A changed name also gets a short hash after a ".", which a name
    # left as it is cannot contain, so two names never share a file.
    slug: str = _UNSAFE.sub("_", text)
    if slug == text and 0 < len(text) <= _MAX_NAME:
        return text
    return f"{slug[:_MAX_NAME]}.{hashlib.sha256(text.encode()).hexdigest()[:8]}"


class PlanStore(BaseModel):
    """Keeps the plans as files, one per agent and conversation: `<root>/<agent>/<session_id>.md`."""

    root: Path = Field(default=Path(".plans"), description="Folder of the plans.")

    def path(self, agent: str, session_id: str) -> Path:
        """The file of the plan of `agent` in the conversation `session_id`."""
        return self.root / _file_name(agent) / f"{_file_name(session_id)}.md"

    def read(self, agent: str, session_id: str) -> str | None:
        """The plan, or None if there is none."""
        try:
            text: str = self.path(agent, session_id).read_text(encoding="utf-8")
        except OSError:
            return None
        return text or None

    def write(self, agent: str, session_id: str, text: str) -> None:
        """Save the plan, replacing the previous one."""
        path: Path = self.path(agent, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write a temporary file and rename it, so a reader never sees half a plan.
        tmp: Path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def delete(self, agent: str, session_id: str) -> None:
        """Remove the plan, if there is one."""
        self.path(agent, session_id).unlink(missing_ok=True)


class PlanningTool(Tool):
    """
    Lets the agent keep a plan for a long task, saved as one file per conversation.

    Added with its `PlanningHook` by `Agent(planning=True)`. The agent can then call:

        - PlanningTool_write_todos(todos=[{"content": "fix the bug", "status": "in_progress"}])
    """

    agent: str = Field(description="Name of the agent; each agent has its own plans.")
    store: PlanStore = Field(
        default_factory=PlanStore, description="Where the plans are kept."
    )

    def write_todos(
        self, todos: list[Todo], ctx: InvocationContext | None = None
    ) -> str:
        """Write the full plan for the current task, replacing the previous one. Use it at the start of a task with several steps, and again whenever an item starts or is done. Keep exactly one item `in_progress` while working."""
        # Tool arguments reach the method as parsed JSON, not as models.
        items: list[Todo] = [Todo.model_validate(t) for t in todos]
        if sum(t.status == "in_progress" for t in items) > 1:
            raise ToolError("only one item can be in_progress at a time")
        text: str = render_plan(items)
        # Without a conversation there is no next message to keep the plan for:
        # the model sees it in this result.
        if ctx is None:
            return text or "Plan cleared."
        if not items:
            self.store.delete(self.agent, ctx.session_id)
            return "Plan cleared."
        if all(t.status == "completed" for t in items):
            self.store.delete(self.agent, ctx.session_id)
            return f"{text}\n\nAll items are completed, so the plan is cleared."
        self.store.write(self.agent, ctx.session_id, text)
        return text
