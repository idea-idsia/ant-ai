"""
Example: safety pipeline with GuardrailsAI (ToxicLanguage + DetectPII).

Requires:
    uv pip install "ant-ai[guardrails-ai]"
    guardrails hub install hub://guardrails/toxic_language hub://guardrails/detect_pii

Run:
    uv run python examples/guardrails_agent.py
"""

from __future__ import annotations

import asyncio

from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage

from ant_ai import Agent, Message, State
from ant_ai.hooks.adapters import GuardrailsAIHook
from ant_ai.llm.integrations.lite_llm import LiteLLMChat

guard = (
    Guard()
    .use(ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="reask"))
    .use(DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="reask"))
)

hook = GuardrailsAIHook(guard=guard)

agent = Agent(
    name="safe-agent",
    system_prompt=(
        "You are a helpful assistant. "
        "Never use toxic language and never share personal information."
    ),
    llm=LiteLLMChat(model="openai/gpt-4o-mini"),
    hooks=[hook],
)


async def main() -> None:
    state = State()
    state.add_message(
        Message(
            role="user",
            content="Create a sample contact card for a fictional person, including their email address and phone number.",
        )
    )
    answer = await agent.ainvoke(state)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
