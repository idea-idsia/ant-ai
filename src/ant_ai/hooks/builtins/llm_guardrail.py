from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from ant_ai.core.message import BaseMessage, Message
from ant_ai.hooks._guardrail_text import TextGuardrailHook, reject

if TYPE_CHECKING:
    from ant_ai.hooks.protocol import PostModelBlock, PostModelRetry

_DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict content guardrail. Judge whether the text below "
    "satisfies the given criteria. Respond only with the requested JSON."
)


class GuardrailVerdict(BaseModel):
    """Structured judge response consumed by ``LLMGuardrailHook``."""

    passed: bool = Field(
        description="True if the text satisfies the guardrail criteria."
    )
    reason: str | None = Field(
        default=None,
        description="Explanation of the failure. Should be set when passed is False.",
    )


class LLMGuardrailHook(TextGuardrailHook, BaseModel):
    """
    Templated LLM-as-judge guardrail base class.

    Sends the model's output to a configurable judge LLM and turns its
    verdict into a ``PostModelPass``/``PostModelRetry``/``PostModelBlock``
    decision. Subclass and override ``criteria`` — or ``build_judge_messages``
    for full control over the prompt — to guardrail on domain-specific rules
    (e.g. "no medical advice", "stay on topic about product X") without
    wrapping a third-party judge framework.

    Only overrides ``after_model``. The judge call is issued directly against
    ``judge_llm`` and does not go through hooks, so it never recurses into
    this guardrail (or any other hook) and cannot trigger its own retries.

    Args:
        judge_llm: Language model invoked as the judge.
        criteria: Natural-language description of what the text must
            satisfy. Used by the default ``build_judge_messages``; ignored
            if that method is overridden.
        on_fail: ``"retry"`` (default) asks the agent to retry with the
            judge's reason as critique; ``"block"`` raises immediately.

    Example:

        ```python
        class NoMedicalAdviceGuardrail(LLMGuardrailHook):
            criteria: str = "The text must not contain medical advice or diagnoses."

        hook = NoMedicalAdviceGuardrail(judge_llm=LiteLLMChat(model="gpt-4o-mini"))
        agent = Agent(..., hooks=[hook])
        ```

        Full prompt control:

        ```python
        class JsonOnlyGuardrail(LLMGuardrailHook):
            def build_judge_messages(self, raw: str) -> list[Message]:
                return [
                    Message(role="system", content="Reply with the requested JSON only."),
                    Message(role="user", content=f"Is this valid JSON?\\n\\n{raw}"),
                ]
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: ClassVar[str] = "llm_guardrail"
    judge_llm: SkipValidation[Any] = Field(
        description="Language model invoked as the judge.",
    )
    criteria: str = Field(
        description="Natural-language description of what the text must satisfy.",
    )
    on_fail: Literal["retry", "block"] = "retry"

    def build_judge_messages(self, raw: str) -> list[BaseMessage]:
        """Build the messages sent to the judge LLM. Override for full prompt control.

        Args:
            raw: The candidate text being judged (model output, or the
                serialized tool-call arguments when the model produced no
                text).

        Returns:
            Messages sent to ``judge_llm`` with ``response_format=GuardrailVerdict``.
        """
        return [
            Message(role="system", content=_DEFAULT_JUDGE_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Criteria:\n{self.criteria}\n\nText to judge:\n{raw}",
            ),
        ]

    async def check_text(self, text: str) -> PostModelBlock | PostModelRetry | None:
        messages = self.build_judge_messages(text)

        try:
            response = await self.judge_llm.ainvoke(
                messages, response_format=GuardrailVerdict
            )
            verdict = GuardrailVerdict.model_validate_json(
                response.message.content or ""
            )
        except Exception as exc:  # noqa: BLE001
            return reject(f"LLM guardrail judge error: {exc}")

        if verdict.passed:
            return None
        return reject(
            verdict.reason or "LLM guardrail check failed", self.on_fail == "block"
        )
