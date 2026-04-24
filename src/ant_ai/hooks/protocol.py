from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass

from ant_ai.core.result import StepResult
from ant_ai.core.types import InvocationContext, State

type WrapCall = Callable[
    [State, InvocationContext | None],
    AsyncIterator,
]


@dataclass(frozen=True)
class PostModelPass:
    """Hook accepted the result; continue with the original result."""

    result: StepResult


@dataclass(frozen=True)
class PostModelBlock:
    """Hook hard-blocked. Raise HookBlockedError immediately."""

    reason: str | None


@dataclass(frozen=True)
class PostModelRetry:
    """Hook requests a retry with the given critique injected into state."""

    reason: str | None


@dataclass(frozen=True)
class PostModelFallback:
    """Hook provides a pre-built safe result, bypassing further validation."""

    result: StepResult


type PostModelDecision = (
    PostModelPass | PostModelBlock | PostModelRetry | PostModelFallback
)


class AgentHook:
    """Base class for agent lifecycle hooks.

    Override only the methods you need — all defaults are no-ops so partial
    implementations work without any boilerplate.

    Lifecycle order per agent invocation:

        before_agent
          (loop)
            before_model
            wrap_model_call → LLM
            after_model → PostModelDecision
            [if tool calls] wrap_tool_call → tools
            [on retry: back to wrap_model_call, skipping before_model]
        after_agent

    Example:

        class ContentGuardrailHook(AgentHook):
            async def after_model(self, result, ctx):
                if "banned" in result.output.raw:
                    return PostModelRetry(reason="Contains banned content")
                return PostModelPass(result=result)
    """

    async def before_agent(self, state: State, ctx: InvocationContext | None) -> None:
        """Called once before the agent starts processing.

        Args:
            state: Current agent state.
            ctx: Invocation context, or None if not available.
        """

    async def after_agent(self, state: State, ctx: InvocationContext | None) -> None:
        """Called once after the agent finishes (or is closed).

        Args:
            state: Current agent state.
            ctx: Invocation context, or None if not available.
        """

    async def before_model(self, state: State, ctx: InvocationContext | None) -> None:
        """Called once per outer loop step. Not called during retry attempts.

        Args:
            state: Current agent state.
            ctx: Invocation context, or None if not available.
        """

    async def after_model(
        self,
        result: StepResult,
        ctx: InvocationContext | None,
    ) -> PostModelDecision:
        """Called after each LLM step. Return a decision to control flow.

        Args:
            result: The step result produced by the LLM step.
            ctx: Invocation context, or None if not available.

        Returns:
            A `PostModelPass` to accept, `PostModelBlock` to raise an error,
            `PostModelRetry` to re-run the step with a critique, or
            `PostModelFallback` to substitute a safe pre-built result.
        """
        return PostModelPass(result=result)

    async def wrap_model_call(
        self,
        call_next: WrapCall,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncGenerator:
        """Wrapper around each individual LLM call.

        Must yield every item produced by `call_next`, or substitute its own.
        Yield a `StepResult` directly to short-circuit the LLM call entirely.

        Args:
            call_next: The next callable in the chain to delegate to.
            state: Current agent state.
            ctx: Invocation context, or None if not available.
        """
        async for item in call_next(state, ctx):
            yield item

    async def wrap_tool_call(
        self,
        call_next: WrapCall,
        state: State,
        ctx: InvocationContext | None,
    ) -> AsyncGenerator:
        """Wrapper around the tool execution step.

        Args:
            call_next: The next callable in the chain to delegate to.
            state: Current agent state.
            ctx: Invocation context, or None if not available.
        """
        async for item in call_next(state, ctx):
            yield item
