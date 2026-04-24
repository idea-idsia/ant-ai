from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from ant_ai.core.result import StepResult
from ant_ai.core.types import InvocationContext, State
from ant_ai.hooks.protocol import (
    AgentHook,
    PostModelBlock,
    PostModelDecision,
    PostModelFallback,
    PostModelPass,
    PostModelRetry,
    WrapCall,
)


class HookLayer(BaseModel):
    """Orchestrates all registered hooks at each agent lifecycle point.

    Before hooks run in registration order (inward); after hooks run in
    reverse order (outward). For hooks [A, B, C] the call order is:
    A.before → B.before → C.before → [core] → C.after → B.after → A.after

    `wrap_model_call` and `wrap_tool_call` follow the same onion pattern,
    building an ASGI-style chain where each hook wraps the next.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hooks: list[AgentHook] = Field(
        default_factory=list,
        description="Hooks to run at each lifecycle point, in registration order.",
    )

    def is_empty(self) -> bool:
        """Returns True when no hooks are registered."""
        return not self.hooks

    async def run_before_agent(
        self, state: State, ctx: InvocationContext | None
    ) -> None:
        """Runs `before_agent` on all hooks in registration order."""
        for h in self.hooks:
            await h.before_agent(state, ctx)

    async def run_after_agent(
        self, state: State, ctx: InvocationContext | None
    ) -> None:
        """Runs `after_agent` on all hooks in reverse registration order."""
        for h in reversed(self.hooks):
            await h.after_agent(state, ctx)

    async def run_before_model(
        self, state: State, ctx: InvocationContext | None
    ) -> None:
        """Runs `before_model` on all hooks in registration order."""
        for h in self.hooks:
            await h.before_model(state, ctx)

    async def run_after_model(
        self,
        result: StepResult,
        ctx: InvocationContext | None,
    ) -> PostModelDecision:
        """Runs `after_model` on all hooks and merges their decisions.

        Uses worst-verdict-wins: Fallback > Block > Retry > Pass. Returns
        `PostModelPass` directly when no hooks are registered.

        Args:
            result: The step result produced by the LLM step.
            ctx: Invocation context, or None if not available.

        Returns:
            The merged decision from all hooks.
        """
        if self.is_empty():
            return PostModelPass(result=result)

        decisions: list[PostModelDecision] = []
        for h in reversed(self.hooks):
            decisions.append(await h.after_model(result, ctx))

        return _merge_decisions(decisions, result)

    def wrap_model_call(self, core: WrapCall) -> WrapCall:
        """Chains all `wrap_model_call` hooks around `core`, outermost-first.

        Args:
            core: The innermost callable to wrap.

        Returns:
            A callable with all hooks layered around `core`.
        """
        wrapped: WrapCall = core
        for hook in reversed(self.hooks):
            wrapped: WrapCall = _make_layer("wrap_model_call", hook, wrapped)
        return wrapped

    def wrap_tool_call(self, core: WrapCall) -> WrapCall:
        """Chains all `wrap_tool_call` hooks around `core`, outermost-first.

        Args:
            core: The innermost callable to wrap.

        Returns:
            A callable with all hooks layered around `core`.
        """
        wrapped: WrapCall = core
        for hook in reversed(self.hooks):
            wrapped: WrapCall = _make_layer("wrap_tool_call", hook, wrapped)
        return wrapped


def _merge_decisions(
    decisions: list[PostModelDecision],
    result: StepResult,
) -> PostModelDecision:
    """Merges multiple hook decisions with worst-severity-wins.

    Priority order: Fallback > Block > Retry > Pass. Block and Retry reasons
    are joined with newlines when multiple hooks return the same verdict.

    Args:
        decisions: Decisions collected from each hook's `after_model`.
        result: The original step result, used as the `PostModelPass` payload.

    Returns:
        The single winning decision.
    """
    for d in decisions:
        if isinstance(d, PostModelFallback):
            return d

    block_reasons: list[str] = [
        d.reason for d in decisions if isinstance(d, PostModelBlock) and d.reason
    ]
    if any(isinstance(d, PostModelBlock) for d in decisions):
        return PostModelBlock(
            reason="\n".join(block_reasons) if block_reasons else None
        )

    retry_reasons: list[str] = [
        d.reason for d in decisions if isinstance(d, PostModelRetry) and d.reason
    ]
    if any(isinstance(d, PostModelRetry) for d in decisions):
        return PostModelRetry(
            reason="\n".join(retry_reasons) if retry_reasons else None
        )

    return PostModelPass(result=result)


def _make_layer(method: str, hook: AgentHook, next_call: WrapCall) -> WrapCall:
    """Wraps `next_call` with a single hook's `method`, forming one layer of the chain.

    Args:
        method: Name of the hook method to call (`wrap_model_call` or `wrap_tool_call`).
        hook: The hook instance providing the method.
        next_call: The callable to pass as `call_next` into the hook method.

    Returns:
        A new callable that invokes `hook.method(next_call, ...)`.
    """

    def layer(state: State, ctx: InvocationContext | None) -> AsyncIterator:
        return getattr(hook, method)(next_call, state, ctx)

    return layer
