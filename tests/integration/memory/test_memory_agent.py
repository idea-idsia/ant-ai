from __future__ import annotations

import asyncio

import pytest

from ant_ai.a2a.client import A2AClient
from ant_ai.a2a.config import A2AConfig
from ant_ai.core.events import Event, FinalAnswerEvent


async def _send(client: A2AClient, message: str, user_id: str) -> str:
    events: list[Event] = [
        ev
        async for ev in client.send_message(
            message,
            request_metadata={"user_id": user_id},
        )
    ]
    final: FinalAnswerEvent | None = next(
        (ev for ev in events if isinstance(ev, FinalAnswerEvent)), None
    )
    return final.content if final else ""


@pytest.mark.integration
@pytest.mark.mem0
@pytest.mark.external
async def test_memory_agent_recalls_preference_across_sessions(
    memory_agent_server: dict,
) -> None:
    """Agent stores a fact in turn 1 and recalls it in a fresh session (turn 2)."""
    url = memory_agent_server["url"]
    user_id = memory_agent_server["user_id"]

    async with A2AClient(config=A2AConfig(endpoint=url)) as client:
        await _send(client, "My favourite language is Rust.", user_id)

    # mem0 user_id-scoped memory extraction is async — indexing takes ~5-7 s.
    await asyncio.sleep(10)

    async with A2AClient(config=A2AConfig(endpoint=url)) as client:
        response = await _send(
            client, "What is my favourite programming language?", user_id
        )

    assert "rust" in response.lower(), (
        f"Expected agent to recall 'rust' from memory, got: {response!r}"
    )
