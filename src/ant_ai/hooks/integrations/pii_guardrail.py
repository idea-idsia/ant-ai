from __future__ import annotations

import asyncio
from typing import ClassVar, Literal

from datafog import scan as datafog_scan
from pydantic import BaseModel

from ant_ai.core.result import StepResult
from ant_ai.core.types import InvocationContext
from ant_ai.hooks._guardrail_text import model_output_text
from ant_ai.hooks.protocol import (
    AgentHook,
    PostModelBlock,
    PostModelDecision,
    PostModelPass,
    PostModelRetry,
)


class PIIGuardrailHook(AgentHook, BaseModel):
    """
    Scans LLM output for personally identifiable information using ``datafog``.

    Only overrides ``after_model`` — scans ``result.output.raw`` for PII
    entities and returns ``PostModelRetry``/``PostModelBlock`` when any are
    found. When the model produced no text (e.g. it wrote content only via a
    tool call), falls back to scanning the serialized tool-call arguments so
    content written via tools (e.g. a filesystem tool) is still checked.

    The retry/block reason names the detected entity *types* only (e.g.
    ``"PII detected: EMAIL, PHONE"``) — never the matched values — so PII
    is not re-surfaced in hook output, logs, or the retry critique sent back
    to the model.

    Args:
        entity_types: PII entity types to detect (e.g. ``["EMAIL", "PHONE"]``).
            Defaults to datafog's built-in entity set when ``None``.
        engine: Detection engine passed to ``datafog.scan``. Defaults to
            ``"regex"``, the dependency-light path that needs no NLP model.
            Use ``"spacy"``/``"gliner"``/``"smart"`` for NER-based detection
            if the corresponding ``datafog`` extra is installed.
        locales: Locale-specific entity sets to enable in addition to the
            defaults (e.g. ``["de"]`` for German IDs).
        allowlist: Exact entity texts exempted from detection.
        allowlist_patterns: Regex patterns; entities whose full text matches
            are exempted from detection.
        on_detect: ``"retry"`` (default) asks the agent to retry with a
            critique; ``"block"`` raises immediately via ``PostModelBlock``.

    Example:

        ```python
        hook = PIIGuardrailHook(entity_types=["EMAIL", "PHONE", "SSN"])
        agent = Agent(..., hooks=[hook])
        ```
    """

    name: ClassVar[str] = "pii_guardrail"
    entity_types: list[str] | None = None
    engine: str = "regex"
    locales: list[str] | None = None
    allowlist: list[str] | None = None
    allowlist_patterns: list[str] | None = None
    on_detect: Literal["retry", "block"] = "retry"

    async def after_model(
        self,
        result: StepResult,
        ctx: InvocationContext | None,
    ) -> PostModelDecision:
        raw = model_output_text(result)
        if raw is None:
            return PostModelPass(result=result)

        def _scan() -> list[str]:
            scan_result = datafog_scan(
                raw,
                engine=self.engine,
                entity_types=self.entity_types,
                locales=self.locales,
                allowlist=self.allowlist,
                allowlist_patterns=self.allowlist_patterns,
            )
            return sorted({entity.type for entity in scan_result.entities})

        try:
            found = await asyncio.to_thread(_scan)
        except Exception as exc:  # noqa: BLE001
            return PostModelRetry(reason=f"PII guardrail scan error: {exc}")

        if not found:
            return PostModelPass(result=result)

        reason = f"PII detected: {', '.join(found)}"
        if self.on_detect == "block":
            return PostModelBlock(reason=reason)
        return PostModelRetry(reason=reason)
