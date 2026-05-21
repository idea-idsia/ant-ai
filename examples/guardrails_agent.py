"""
Example: safety pipeline with GuardrailsAI (DetectPII).

Requires:
    uv pip install "ant-ai[guardrails-ai]"
    guardrails hub install hub://guardrails/detect_pii

Note: ToxicLanguage validator will be added once guardrails-ai is restored on PyPI.

Run:
    uv run python examples/guardrails_agent.py
"""
from __future__ import annotations

import asyncio

from guardrails import Guard
from validator.main import DetectPII  # imported directly: guardrails hub install unavailable (PyPI quarantine)

from ant_ai.agent.agent import Agent
from ant_ai.core.message import Message
from ant_ai.core.types import State
from ant_ai.hooks.adapters import GuardrailsAIHook
from ant_ai.llm.integrations.lite_llm import LiteLLMChat


def build_agent() -> Agent:
    # When validation fails, guardrails injects a critique into the state and
    # the ReAct loop retries the LLM call (up to Agent.max_retries times).
    guard = Guard().use(
        DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="reask")
    )
    return Agent(
        name="safe-agent",
        system_prompt=(
            "You are a helpful assistant. "
            "Never share personal information such as email addresses or phone numbers."
        ),
        llm=LiteLLMChat(model="openai/gpt-4o-mini"),
        hooks=[GuardrailsAIHook(guard=guard)],
    )


async def main() -> None:
    agent = build_agent()
    state = State()
    state.add_message(Message(role="user", content="Explain why collaboration matters in teams."))
    answer = await agent.ainvoke(state)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
