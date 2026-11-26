from __future__ import annotations

import asyncio
from abc import abstractmethod
from contextlib import aclosing
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    field_validator,
    model_validator,
)

from ant_ai.agent.agent_tool import agent_as_tool
from ant_ai.agent.extensions import AgentExtensions
from ant_ai.agent.skill_support import AgentSkills
from ant_ai.agent.subagent import SubagentTool
from ant_ai.agent.turn import final_answer, stream_turn
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.builtins.planning_tool import PlanningTool
from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import Tool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic import BaseModel

    from ant_ai.agent.loop.loop import BaseAgentLoop
    from ant_ai.core.events import BaseEvent
    from ant_ai.core.message import Message
    from ant_ai.core.types import InvocationContext, State


class BaseAgent(AgentSkills, AgentExtensions):
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
    tools: list[Tool | BaseAgent] = Field(
        default_factory=list,
        description="Tools available to the agent. An agent in the list becomes "
        "its `as_tool()` with the default settings.",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of times to retry after a hook returns RETRY.",
    )

    _registry: ToolRegistry = PrivateAttr()
    _loop: BaseAgentLoop = PrivateAttr()
    _private_tools: dict[int, Tool] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("tools", mode="before")
    @classmethod
    def _agents_as_tools(cls, tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools
        return [t.as_tool() if isinstance(t, BaseAgent) else t for t in tools]

    @model_validator(mode="after")
    def _build(self) -> BaseAgent:
        # Copies, so the agent's own fields are not changed and a rebuild adds nothing twice.
        # After `_agents_as_tools` the list holds only tools.
        tools: list[Tool] = [
            *cast("list[Tool]", self.tools),
            *self._build_extensions(self.name),
        ]
        self._registry = ToolRegistry()
        self._private_tools = {}
        for t in tools:
            self._register_tool(t)
        self._load_skills()
        self._loop: BaseAgentLoop = self._make_loop()
        return self

    def _register_tool(self, tool: Tool) -> None:
        """Register `tool`, keeping subagent and planning tools from subagents."""
        if isinstance(tool, SubagentTool):
            tool = tool._bind(self)
        elif not isinstance(tool, PlanningTool):
            self._registry.register(tool)
            return
        before: set[int] = {id(t) for t in self._registry.tools}
        self._registry.register(tool)
        for t in self._registry.tools:
            if id(t) not in before:
                self._private_tools[id(t)] = t

    def _delegable_tools(self) -> dict[str, Tool]:
        """The tools a `SubagentTool` can give to a subagent, by name."""
        return {
            t.name: t
            for t in self._registry.tools
            if t.name and id(t) not in self._private_tools
        }

    @abstractmethod
    def _make_loop(self) -> BaseAgentLoop:
        """Create and return the reasoning loop for this agent."""
        ...

    @property
    def system_message(self) -> Message:
        """Return this agent's system message, with skill discovery appended if present."""
        return self._system_message_with_skills(self.system_prompt)

    async def stream(
        self,
        state: State,
        *,
        max_steps: int = 10,
        ctx: InvocationContext | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> AsyncGenerator[BaseEvent]:
        """
        Stream events for a single agent turn.

        Live `ContentDeltaEvent`s are interleaved before the terminal
        `ReasoningEvent`/`ToolCallingEvent`/`FinalAnswerEvent` they build up
        to. Three independent things can force the whole-event path instead,
        each for its own reason:

        - A registered hook needs the complete response to decide (see
          `HookLayer.is_stream_safe()` / `AgentHook.stream_safe`) — tokens
          already streamed to a client cannot be retracted.
        - Structured output's final coercion step never streams, since it
          may silently rewrite the raw text after the fact.
        - Over A2A, `ContentDeltaEvent`s are only forwarded to the wire when
          the executor was built with `stream_artifacts=True` — this
          method still emits them either way.

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
            updates and the final answer event. Each event's `origin.agent`
            is set to this agent's name, and its `session_id` to the one of
            `ctx`, unless already set.
        """
        async with aclosing(
            stream_turn(
                self.name,
                self._hook_layer,
                self._loop,
                state,
                max_steps=max_steps,
                ctx=ctx,
                response_schema=response_schema,
            )
        ) as events:
            async for event in events:
                yield event

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
        return await final_answer(
            self.stream(
                state,
                max_steps=max_steps,
                ctx=ctx,
                response_schema=response_schema,
            )
        )

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

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        max_steps: int = 10,
    ) -> Tool:
        """
        Return this agent as a tool of another agent.

        Each call runs the agent on a new conversation that holds only the
        task, and returns only its final answer: the files it reads and the
        tools it calls stay out of the caller's conversation. The caller's
        `InvocationContext` is passed on, so the run shares the session and
        the trace. An agent put in the `tools` of another agent becomes this
        tool with the default settings.

        Example:

            explorer = Agent(name="explorer", description="Searches the code base.",
                             system_prompt="...", llm=llm, tools=[FilesystemTool()])
            lead = Agent(name="lead", system_prompt="...", llm=llm,
                         tools=[explorer])  # or explorer.as_tool(max_steps=20)

        Args:
            name: Tool name, by default the agent's name. Providers accept only
                letters, digits, `_` and `-`.
            description: Tool description, by default the agent's description.
            max_steps: Step budget of each run.

        Returns:
            A tool with one argument, `message`, as `A2AAgentTool`.

        Raises:
            ValueError: The agent was built with `planning=True`; only the
                agent that talks to the user keeps a plan. Or the tool name is
                not one providers accept.
        """
        return agent_as_tool(
            self, name=name, description=description, max_steps=max_steps
        )

    @abstractmethod
    def add_tool(self, tool: Tool | BaseAgent) -> None:
        """Register a tool with the agent; an agent is added as its `as_tool()`."""
        ...

    @property
    def registry(self) -> ToolRegistry:
        """Return the tool registry for this agent."""
        return self._registry
