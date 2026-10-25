from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ant_ai.core.types import InvocationContext
from ant_ai.tools.tool import Tool, ToolError

_MARK: dict[str, str] = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_SAFE_NAME: re.Pattern[str] = re.compile(r"[A-Za-z0-9_-]{1,64}")


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
    # A name that is not a plain file name is replaced by its hash, so the
    # file always stays inside plan_dir.
    if _SAFE_NAME.fullmatch(text):
        return text
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def plan_path(planning: PlanningTool, ctx: InvocationContext | None) -> Path:
    """The file that holds the plan of `planning.agent` in the conversation in `ctx`.

    One folder per agent and one file per `session_id`, so two conversations
    never share a plan, and neither do two agents of the same conversation
    (over A2A the called agent receives the caller's `session_id`). Without a
    context the plan goes to `default.md`.
    """
    session: str = ctx.session_id if ctx is not None else "default"
    return planning.plan_dir / _file_name(planning.agent) / f"{_file_name(session)}.md"


def read_plan(planning: PlanningTool, ctx: InvocationContext | None) -> str | None:
    """The saved plan of `planning.agent` in the conversation in `ctx`, or None if it has none."""
    try:
        text: str = plan_path(planning, ctx).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return text or None


class PlanningTool(Tool):
    """
    Lets the agent keep a plan for a long task, saved as one file per conversation.

    Example:
        ```python
        from ant_ai.agent import Agent
        from ant_ai.hooks.builtins import PlanningHook
        from ant_ai.tools.builtins import PlanningTool

        planning = PlanningTool(agent="worker")
        agent = Agent(name="worker", tools=[planning], hooks=[PlanningHook(planning)], ...)
        ```

        The agent can now call:

            - PlanningTool_write_todos(todos=[{"content": "fix the bug", "status": "in_progress"}])

    Notes:
        The plan is written to `plan_dir/<agent>/<session_id>.md`, so it is
        still there when the next message of the same conversation arrives, also
        over A2A, and it can be read by a person. `PlanningHook` shows it to the
        model before every call.
    """

    agent: str = Field(description="Name of the agent; each agent has its own plans.")
    plan_dir: Path = Field(default=Path(".plans"), description="Folder of the plans.")

    def write_todos(
        self, todos: list[Todo], ctx: InvocationContext | None = None
    ) -> str:
        """Write the full plan for the current task, replacing the previous one. Use it at the start of a task with several steps, and again whenever an item starts or is done. Keep exactly one item `in_progress` while working."""
        # Tool arguments reach the method as parsed JSON, not as models.
        items: list[Todo] = [Todo.model_validate(t) for t in todos]
        if sum(t.status == "in_progress" for t in items) > 1:
            raise ToolError("only one item can be in_progress at a time")
        path: Path = plan_path(self, ctx)
        if not items:
            path.unlink(missing_ok=True)
            return "Plan cleared."
        text: str = render_plan(items)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text
