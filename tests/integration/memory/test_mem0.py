from __future__ import annotations

import asyncio
import uuid

import pytest

from ant_ai.core.message import Message
from ant_ai.memory.backends.mem0 import Mem0Memory


async def _poll(
    memory: Mem0Memory,
    query: str,
    keyword: str,
    *,
    timeout: int = 30,
    interval: int = 3,
    **kwargs,
) -> str:
    """Poll retrieve() until keyword appears in results or timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        results = await memory.retrieve(query, **kwargs)
        contents = " ".join(m.content for m in results).lower()
        if keyword in contents:
            return contents
        await asyncio.sleep(interval)
    return ""


@pytest.mark.external
@pytest.mark.mem0
async def test_mem0_update_and_retrieve_roundtrip():
    """update() stores a fact; retrieve() returns it in a subsequent search."""
    user_id = f"test-{uuid.uuid4()}"
    memory = Mem0Memory()

    await memory.update(
        [
            Message(role="user", content="My favourite programming language is Rust."),
            Message(
                role="assistant", content="Got it, I'll remember that you prefer Rust."
            ),
        ],
        user_id=user_id,
    )

    contents = await _poll(
        memory, "programming language preference", "rust", user_id=user_id
    )
    assert "rust" in contents, (
        f"Expected 'rust' in retrieved memories, got: {contents!r}"
    )


@pytest.mark.external
@pytest.mark.mem0
async def test_mem0_retrieve_returns_messages():
    """retrieve() always returns a list of Message objects with role='system'."""
    user_id = f"test-{uuid.uuid4()}"
    memory = Mem0Memory()

    await memory.update(
        [Message(role="user", content="I live in Zurich.")],
        user_id=user_id,
    )

    results = await memory.retrieve("location", user_id=user_id)

    assert isinstance(results, list)
    for msg in results:
        assert isinstance(msg, Message)
        assert msg.role == "system"
        assert msg.content


@pytest.mark.external
@pytest.mark.mem0
async def test_mem0_empty_retrieve_returns_empty_list():
    """retrieve() returns [] when no relevant memories exist for the user."""
    user_id = f"empty-{uuid.uuid4()}"
    memory = Mem0Memory()

    results = await memory.retrieve("anything", user_id=user_id)

    assert results == []
