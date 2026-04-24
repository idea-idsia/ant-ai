from abc import abstractmethod
from collections.abc import AsyncGenerator

from ant_ai.agent.agent import Agent
from ant_ai.core.events import AgentEvent
from ant_ai.core.types import InvocationContext, State
from ant_ai.workflow.workflow import NodeYield


class BaseAction:
    async def __call__(
        self, agent: Agent, state: State, ctx: InvocationContext
    ) -> State | AsyncGenerator[NodeYield]:
        return await self.ainvoke(agent, state, ctx)

    @abstractmethod
    async def ainvoke(
        self, agent: Agent, state: State, ctx: InvocationContext
    ) -> State | AsyncGenerator[NodeYield]:
        raise NotImplementedError()

    async def _process_event(self, event: AgentEvent) -> AgentEvent:
        return event
