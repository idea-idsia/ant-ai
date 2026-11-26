from __future__ import annotations

import re
from contextlib import aclosing
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ant_ai.core.events import (
    ClarificationNeededEvent,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
)
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.tools.tool import Tool, ToolError, tool

if TYPE_CHECKING:
    from ant_ai.agent.base import BaseAgent

_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


class SubagentTask(BaseModel):
    message: str = Field(
        description="The complete task. The subagent sees nothing else of the "
        "conversation, so include every detail it needs."
    )


def agent_as_tool(
    agent: BaseAgent,
    *,
    name: str | None,
    description: str | None,
    max_steps: int,
) -> Tool:
    """Return `agent` as a tool of another agent; see `BaseAgent.as_tool()`."""
    if agent.planning:
        raise ValueError(
            f"Agent '{agent.name}' has planning=True: a subagent does not keep a plan."
        )
    tool_name = name or agent.name
    if not _TOOL_NAME.fullmatch(tool_name):
        raise ValueError(
            f"Tool name {tool_name!r} is not valid: use 1-64 letters, digits, "
            "`_` or `-`, e.g. `as_tool(name=...)`."
        )

    async def run(message: str, ctx: InvocationContext | None = None) -> str:
        state = State(messages=[Message(role="user", content=message)])
        # aclosing: an early exit closes the run now, so its after_agent hooks run.
        async with aclosing(
            agent.stream(state, max_steps=max_steps, ctx=ctx)
        ) as events:
            async for event in events:
                if isinstance(event, FinalAnswerEvent):
                    return event.content
                if isinstance(event, ClarificationNeededEvent):
                    raise ToolError(
                        f"{agent.name} needs more information: {event.content}"
                    )
                if isinstance(event, MaxStepsReachedEvent):
                    raise ToolError(
                        f"{agent.name} stopped after {max_steps} steps without an answer"
                    )
        raise ToolError(f"{agent.name} ended without an answer")

    return tool(
        name=tool_name,
        description=description or agent.description,
        args_model=SubagentTask,
    )(run)
