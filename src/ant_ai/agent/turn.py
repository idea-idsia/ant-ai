from __future__ import annotations

from typing import TYPE_CHECKING

from ant_ai.core.events import FinalAnswerEvent
from ant_ai.observer import obs

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from pydantic import BaseModel

    from ant_ai.agent.loop.loop import BaseAgentLoop
    from ant_ai.core.events import BaseEvent
    from ant_ai.core.types import InvocationContext, State
    from ant_ai.hooks.layer import HookLayer


async def stream_turn(
    agent_name: str,
    hooks: HookLayer,
    loop: BaseAgentLoop,
    state: State,
    *,
    max_steps: int,
    ctx: InvocationContext | None,
    response_schema: type[BaseModel] | None,
) -> AsyncGenerator[BaseEvent]:
    """Run one agent turn through `loop` between the agent-level hooks."""
    with obs.bind(
        session_id=ctx.session_id if ctx else "",
        agent_name=agent_name,
    ):
        await hooks.run_before_agent(state, ctx)
        try:
            async for event in loop.stream(
                state,
                ctx,
                max_steps=max_steps,
                response_schema=response_schema,
            ):
                if event.origin.agent is None:
                    event.origin.agent = agent_name
                if ctx is not None and event.session_id is None:
                    event.session_id = ctx.session_id
                yield event
        finally:
            await hooks.run_after_agent(state, ctx)


async def final_answer(events: AsyncIterator[BaseEvent]) -> str:
    """The content of the last `FinalAnswerEvent` of `events`, or ""."""
    final = ""
    async for event in events:
        if isinstance(event, FinalAnswerEvent):
            final = event.content
    return final
