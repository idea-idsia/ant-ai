from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import ClassVar

from ant_ai.core.message import BaseMessage, ToolCallResultMessage
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.protocol import AgentHook, WrapCall
from ant_ai.tools.builtins.planning_tool import PlanStore

_PLAN_PREFIX = "[Current plan]\n"


class PlanningHook(AgentHook):
    """Shows the saved plan of `agent` to the model when the conversation does not show it.

    Added with the `PlanningTool` by `Agent(planning=True)`. While the agent works,
    the result of its last `write_todos` already shows the plan. When it does not,
    because a new state was started for the conversation or history compression
    summarised it away, the plan is appended to the last message of a copy of the
    state: the start of the conversation stays the same, so prefix caching keeps
    working, and the state's messages are not changed.
    """

    name: ClassVar[str] = "planning"

    # Only the input to the model changes, never its output.
    stream_safe: ClassVar[bool] = True

    def __init__(self, agent: str, store: PlanStore) -> None:
        self.agent = agent
        self.store = store

    async def wrap_model_call(
        self,
        call_next: WrapCall,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncGenerator:
        """Call the model on a copy of `state` whose last message ends with the plan, if it is not shown yet.

        Args:
            call_next: The next callable in the chain to delegate to.
            state: Current agent state; its messages are kept as they are.
            ctx: Invocation context, or None if not available.
        """
        plan: str | None = None
        if ctx is not None and state.messages:
            plan = await asyncio.to_thread(self.store.read, self.agent, ctx.session_id)
        if plan is not None and not any(
            isinstance(m, ToolCallResultMessage) and m.content == plan
            for m in reversed(state.messages)
        ):
            last: BaseMessage = state.messages[-1]
            content: str = f"{last.content or ''}\n\n{_PLAN_PREFIX}{plan}"
            state = state.model_copy(
                update={
                    "messages": [
                        *state.messages[:-1],
                        last.model_copy(update={"content": content}),
                    ]
                }
            )
        async for item in call_next(state, ctx):
            yield item
