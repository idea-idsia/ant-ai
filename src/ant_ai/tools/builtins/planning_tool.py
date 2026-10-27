from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ant_ai.core.types import InvocationContext
from ant_ai.tools.tool import Tool, ToolError

PLAN_DIR: Path = Path(".plans")

_MARK: dict[str, str] = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_UNSAFE: re.Pattern[str] = re.compile(r"[^A-Za-z0-9_-]")


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
    # Anything but letters, digits, - and _ becomes _, so the file stays inside PLAN_DIR.
    return _UNSAFE.sub("_", text) or "_"


def plan_path(agent: str, ctx: InvocationContext | None) -> Path:
    """The file that holds the plan of `agent` in the conversation in `ctx`.

    `.plans/<agent>/<session_id>.md`: one file per conversation, and one folder
    per agent, because over A2A the called agent receives the caller's
    `session_id`. Without a context the plan goes to `default.md`.
    """
    session: str = ctx.session_id if ctx is not None else "default"
    return PLAN_DIR / _file_name(agent) / f"{_file_name(session)}.md"


def read_plan(agent: str, ctx: InvocationContext | None) -> str | None:
    """The saved plan of `agent` in the conversation in `ctx`, or None if it has none."""
    try:
        text: str = plan_path(agent, ctx).read_text(encoding="utf-8")
    except OSError:
        return None
    return text or None


class PlanningTool(Tool):
    """
    Lets the agent keep a plan for a long task, saved as one file per conversation.

    Added with its `PlanningHook` by `Agent(planning=True)`. The agent can then call:

        - PlanningTool_write_todos(todos=[{"content": "fix the bug", "status": "in_progress"}])
    """

    agent: str = Field(description="Name of the agent; each agent has its own plans.")

    def write_todos(
        self, todos: list[Todo], ctx: InvocationContext | None = None
    ) -> str:
        """Write the full plan for the current task, replacing the previous one. Use it at the start of a task with several steps, and again whenever an item starts or is done. Keep exactly one item `in_progress` while working."""
        # Tool arguments reach the method as parsed JSON, not as models.
        items: list[Todo] = [Todo.model_validate(t) for t in todos]
        if sum(t.status == "in_progress" for t in items) > 1:
            raise ToolError("only one item can be in_progress at a time")
        path: Path = plan_path(self.agent, ctx)
        if not items:
            path.unlink(missing_ok=True)
            return "Plan cleared."
        text: str = render_plan(items)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text
