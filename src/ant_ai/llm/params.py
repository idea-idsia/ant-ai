from __future__ import annotations

from typing import Any

from ant_ai.core.types import InvocationContext, LLMSettings


def resolve_llm_params(
    default_params: dict[str, Any],
    base: LLMSettings,
    ctx: InvocationContext | None,
) -> dict[str, Any]:
    """Merge the completion-parameter layers into a single kwargs mapping.

    Layers are applied lowest precedence first, so each later layer overrides
    the keys set by the earlier ones. Every layer is a flat mapping of
    top-level completion kwargs, so a shallow merge is correct; a nested
    ``extra_body`` in ``default_params`` is never clobbered because the typed
    layers only carry scalar knobs.

    Args:
        default_params: Raw provider long-tail set on the integration
            (``extra_body``, ``num_retries``, …). Untyped on purpose: the
            escape hatch for anything outside the safe surface. Lowest
            precedence.
        base: The instance's typed baseline, e.g.
            ``LiteLLMChat(settings=LLMSettings(temperature=0.3))``.
        ctx: Request-scoped context, or None. Its ``llm_settings``, when
            present, is the per-request override and wins over the rest.

    Returns:
        A new dict of completion kwargs; ``default_params`` is not mutated.
    """
    params = {**default_params, **base.overrides()}
    if ctx is not None and ctx.llm_settings is not None:
        params |= ctx.llm_settings.overrides()
    return params
