from __future__ import annotations

import pytest

pytest.importorskip(
    "guardrails",
    reason="guardrails-ai not installed; install with: pip install 'ant-ai[guardrails-ai]'",
)

from guardrails import Guard
from guardrails.classes import FailResult
from guardrails.types import OnFailAction
from guardrails.validator_base import Validator, register_validator

try:
    from guardrails.hub import DetectPII

    _DETECT_PII_AVAILABLE = True
except ImportError:
    _DETECT_PII_AVAILABLE = False

try:
    from guardrails.hub import ToxicLanguage

    _TOXIC_LANGUAGE_AVAILABLE = True
except ImportError:
    _TOXIC_LANGUAGE_AVAILABLE = False

from ant_ai.agent.agent import Agent
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import Message
from ant_ai.core.result import LLMOutput, StepResult, Transition, TransitionAction
from ant_ai.core.types import State
from ant_ai.hooks.integrations import GuardrailsAIHook
from ant_ai.hooks.protocol import PostModelRetry
from ant_ai.llm.integrations.lite_llm import LiteLLMChat


def _llm_result(raw: str) -> StepResult:
    return StepResult(
        output=LLMOutput(raw=raw),
        transition=Transition(action=TransitionAction.END),
    )


@pytest.mark.integration
@pytest.mark.guardrailsai
async def test_failure_reason_is_present_in_retry():
    """Real Guard + real validator — no mocks, no external LLM.

    Verifies that PostModelRetry.reason contains the validator name and the
    specific error message returned by the validator, in the expected
    ``"validator_name: error_message"`` format.
    """

    @register_validator(name="always_fails", data_type="string")
    class AlwaysFails(Validator):
        def validate(self, value, metadata):
            return FailResult(error_message="forbidden content detected")

    guard = Guard().use(AlwaysFails(on_fail=OnFailAction.NOOP))
    hook = GuardrailsAIHook(guard=guard)

    decision = await hook.after_model(_llm_result("any text"), ctx=None)

    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "AlwaysFails: forbidden content detected"


@pytest.mark.integration
@pytest.mark.guardrailsai
async def test_failure_reason_multiple_validators():
    """Two real validators — reason must join both with ``'; '``."""

    @register_validator(name="check_a", data_type="string")
    class CheckA(Validator):
        def validate(self, value, metadata):
            return FailResult(error_message="issue A")

    @register_validator(name="check_b", data_type="string")
    class CheckB(Validator):
        def validate(self, value, metadata):
            return FailResult(error_message="issue B")

    guard = Guard().use(
        CheckA(on_fail=OnFailAction.NOOP), CheckB(on_fail=OnFailAction.NOOP)
    )
    hook = GuardrailsAIHook(guard=guard)

    decision = await hook.after_model(_llm_result("any text"), ctx=None)

    assert isinstance(decision, PostModelRetry)
    assert decision.reason == "CheckA: issue A; CheckB: issue B"


def _state(content: str) -> State:
    s = State()
    s.add_message(Message(role="user", content=content))
    return s


def _safe_agent(guard: Guard) -> Agent:
    return Agent(
        name="safe-agent",
        system_prompt=(
            "You are a helpful assistant. "
            "Never use toxic language and never share personal information."
        ),
        llm=LiteLLMChat(model="openai/gpt-4o-mini"),
        hooks=[GuardrailsAIHook(guard=guard)],
    )


@pytest.mark.external
@pytest.mark.guardrailsai
@pytest.mark.skipif(
    not _TOXIC_LANGUAGE_AVAILABLE,
    reason="ToxicLanguage hub validator not installed",
)
async def test_toxic_language_validator_self_corrects():
    # on_fail="reask" returns a failing ValidationOutcome without internal LLM
    # calls (no llm_api passed to validate()). ant-ai's retry loop takes over.
    guard = Guard().use(
        ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="reask")  # type: ignore[name-defined]
    )
    agent = _safe_agent(guard)
    answer = await agent.ainvoke(_state("Explain why collaboration matters in teams."))
    assert answer and len(answer) > 0


@pytest.mark.external
@pytest.mark.guardrailsai
@pytest.mark.skipif(
    not _DETECT_PII_AVAILABLE,
    reason="DetectPII hub validator not installed",
)
async def test_pii_validator_triggers_retry_or_raises():
    guard = Guard().use(
        # on_fail="reask": guardrails returns a failing outcome (no internal LLM
        # reask since validate() has no llm_api); ant-ai's retry loop takes over.
        DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="reask")  # type: ignore[name-defined]
    )
    agent = _safe_agent(guard)
    try:
        answer = await agent.ainvoke(
            _state(
                "Give me a fake example email address and phone number for a contact card."
            )
        )
        assert answer and len(answer) > 0
    except HookMaxRetriesError:
        # Exhausted retries without passing validation — also a valid outcome
        pass
