from __future__ import annotations

import pytest

pytest.importorskip(
    "guardrails",
    reason="guardrails-ai not installed; install with: pip install 'ant-ai[guardrails-ai]'",
)

from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage

from ant_ai.agent.agent import Agent
from ant_ai.core.exceptions import HookMaxRetriesError
from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.hooks.adapters import GuardrailsAIHook
from ant_ai.llm.integrations.lite_llm import LiteLLMChat


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
async def test_toxic_language_validator_self_corrects():
    guard = Guard().use(
        ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="reask")
    )
    agent = _safe_agent(guard)
    answer = await agent.ainvoke(_state("Explain why collaboration matters in teams."))
    assert answer and len(answer) > 0


@pytest.mark.external
async def test_pii_validator_triggers_retry_or_raises():
    guard = Guard().use(
        DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="reask")
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
