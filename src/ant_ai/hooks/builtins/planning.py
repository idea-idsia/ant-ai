from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.protocol import AgentHook, WrapCall
from ant_ai.tools.builtins.planning_tool import PlanningTool, read_plan

_PLAN_PREFIX = "[Current plan]\n"


class PlanningHook(AgentHook):
    """Shows the plan saved by a `PlanningTool` to the model before every call.

    The plan is read from the file of the current conversation and added as a
    system message, after any compression summary. It is not part of the
    conversation, so history compression cannot remove it, and the state's own
    messages are not changed: the model sees a copy with the plan added.

        planning = PlanningTool(agent="worker")
        agent = Agent(name="worker", tools=[planning], hooks=[PlanningHook(planning)])
    """

    name: ClassVar[str] = "planning"

    # Only the input to the model changes, never its output.
    stream_safe: ClassVar[bool] = True

    def __init__(self, planning: PlanningTool) -> None:
        self.planning = planning

    async def wrap_model_call(
        self,
        call_next: WrapCall,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncGenerator:
        """Call the model on a copy of `state` that ends its system preamble with the plan.

        Args:
            call_next: The next callable in the chain to delegate to.
            state: Current agent state; its messages are kept as they are.
            ctx: Invocation context, or None if not available.
        """
        plan: str | None = read_plan(self.planning, ctx)
        if plan is None:
            async for item in call_next(state, ctx):
                yield item
            return
        # After the leading system messages (e.g. a compression summary), before the conversation.
        split: int = next(
            (i for i, m in enumerate(state.messages) if m.role != "system"),
            len(state.messages),
        )
        view: State = state.model_copy(
            update={
                "messages": [
                    *state.messages[:split],
                    Message(role="system", content=_PLAN_PREFIX + plan),
                    *state.messages[split:],
                ]
            }
        )
        async for item in call_next(view, ctx):
            yield item
