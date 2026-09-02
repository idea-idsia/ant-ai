from __future__ import annotations

from typing import Any

from ant_ai.core.types import InvocationContext, LLMSettings


def resolve_llm_params(
    default_params: dict[str, Any],
    base: LLMSettings,
    ctx: InvocationContext | None,
) -> dict[str, Any]:
    """Merge the completion-parameter layers, lowest precedence first.

    1. ``default_params`` — the raw provider long-tail set on the integration
       (``extra_body``, ``num_retries``, …). Untyped on purpose: it is the
       escape hatch for anything outside the safe surface.
    2. ``base`` — the instance's typed baseline, e.g.
       ``LiteLLMChat(settings=LLMSettings(temperature=0.3))``.
    3. ``ctx.llm_settings`` — the per-request override. Wins over the rest.

    Every layer is a flat mapping of top-level completion kwargs, so a shallow
    merge is correct; a nested ``extra_body`` in ``default_params`` is never
    clobbered because the typed layers only carry scalar knobs.
    """
    params = {**default_params, **base.overrides()}
    if ctx is not None and ctx.llm_settings is not None:
        params |= ctx.llm_settings.overrides()
    return params
