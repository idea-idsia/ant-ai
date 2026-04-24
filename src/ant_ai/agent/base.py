from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncGenerator
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    model_validator,
)

from ant_ai.agent.loop.loop import BaseAgentLoop
from ant_ai.core.events import Event, FinalAnswerEvent
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.layer import HookLayer
from ant_ai.hooks.protocol import AgentHook
from ant_ai.llm.protocol import ChatLLM
from ant_ai.observer import obs
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import Tool


class BaseAgent(BaseModel):
    """
    Base class for all agent implementations.

    Owns the public invocation and streaming APIs, hook chain, and
    identity fields. Subclasses implement `_make_loop()` to provide the
    reasoning loop.
    """

    name: str = Field(
        default="BaseAgent",
        description="Display name used in routing and observability.",
    )
    system_prompt: str = Field(description="System instruction for the LLM.")
    llm: Annotated[ChatLLM, SkipValidation]
    description: str = Field(
        default="A base agent.",
        description="Human-readable description of the agent for documentation.",
    )
    tools: list[Tool] = Field(
        default_factory=list,
        description="Tools available to the agent.",
    )
    hooks: list[AgentHook] = Field(
        default_factory=list,
        description="Lifecycle hooks for the agent.",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of times to retry after a hook returns RETRY.",
    )

    _registry: ToolRegistry = PrivateAttr()
    _loop: BaseAgentLoop = PrivateAttr()
    _hook_layer: HookLayer = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build(self) -> BaseAgent:
        self._registry = ToolRegistry(self.tools)
        self._hook_layer = HookLayer(hooks=self.hooks)
        self._loop = self._make_loop()
        return self

    @abstractmethod
    def _make_loop(self) -> BaseAgentLoop:
        """Create and return the reasoning loop for this agent."""
        ...

    @property
    def system_message(self) -> Message:
        """Return this agent's system message."""
        return Message(role="system", content=self.system_prompt)

    async def stream(
        self,
        state: State,
        *,
        max_steps: int = 10,
        ctx: InvocationContext | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> AsyncGenerator[Event]:
        """
        Stream events for a single agent turn.

        Pass `response_schema` to request structured output on the
        `FinalAnswerEvent`. When provided, the final event content will be
        valid JSON matching that schema.

        Args:
            state: Current agent state, including message history and other
                execution context.
            max_steps: Maximum number of loop steps before stopping.
            ctx: Invocation context for the current run.
            response_schema: Schema for structured output on the
                `FinalAnswerEvent`.

        Yields:
            Event: Events produced during execution, including intermediate
            updates and the final answer event.
        """
        with obs.bind(
            session_id=ctx.session_id if ctx else "",
            agent_name=self.name,
        ):
            await self._hook_layer.run_before_agent(state, ctx)
            try:
                async for event in self._loop.stream(
                    state,
                    ctx,
                    max_steps=max_steps,
                    response_schema=response_schema,
                ):
                    yield event
            finally:
                await self._hook_layer.run_after_agent(state, ctx)

    async def ainvoke(
        self,
        state: State,
        *,
        max_steps: int = 10,
        ctx: InvocationContext | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """
        Run a single agent turn and return the final answer.

        This is the asynchronous counterpart to `invoke`. Internally, it
        consumes `stream()` and returns the content of the `FINAL_ANSWER`
        event.

        Pass `response_schema` to request structured output on the
        `FINAL_ANSWER` event. When provided, the returned string will contain
        valid JSON matching that schema.

        Args:
            state: Current agent state, including message history and other
                execution context.
            max_steps: Maximum number of loop steps before stopping.
            ctx: Invocation context for the current run.
            response_schema: Schema for structured output on the
                `FinalAnswerEvent`.

        Returns:
            The content of the `FinalAnswerEvent`.
        """
        final = ""
        async for event in self.stream(
            state,
            max_steps=max_steps,
            ctx=ctx,
            response_schema=response_schema,
        ):
            if isinstance(event, FinalAnswerEvent):
                final: str = event.content

        return final

    def invoke(
        self,
        state: State,
        *,
        max_steps: int = 10,
        ctx: InvocationContext | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        """
        Run a single agent turn and return the final answer.

        This is the synchronous counterpart to `ainvoke`. Internally, it runs
        the asynchronous invocation path and returns the content of the
        `FinalAnswerEvent`.

        Pass `response_schema` to request structured output on the
        `FinalAnswerEvent`. When provided, the returned string will contain
        valid JSON matching that schema.

        Args:
            state: Current agent state, including message history and other
                execution context.
            max_steps: Maximum number of loop steps before stopping.
            ctx: Invocation context for the current run.
            response_schema: Schema for structured output on the
                `FinalAnswerEvent`.

        Returns:
            The content of the `FinalAnswerEvent`.

        Raises:
            RuntimeError: If called from a thread with an active event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.ainvoke(
                    state,
                    max_steps=max_steps,
                    ctx=ctx,
                    response_schema=response_schema,
                )
            )
        else:
            raise RuntimeError(
                "invoke() cannot be called while an event loop is already running. "
                "Use `await ainvoke(...)` instead."
            )

    @abstractmethod
    def add_tool(self, tool: Tool) -> None:
        """Register a tool with the agent."""
        ...

    @property
    def registry(self) -> ToolRegistry:
        """Return the tool registry for this agent."""
        return self._registry
