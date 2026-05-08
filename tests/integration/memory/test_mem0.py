"""
Integration tests for Mem0Memory.

Requires:
- mem0ai installed: pip install 'ant-ai[mem0]'
- MEM0_API_KEY set in the environment (or .env)

Run with: uv run pytest tests/integration/memory/ -m external
"""

from __future__ import annotations

import uuid

import pytest

from ant_ai.core.message import Message
from ant_ai.memory.backends.mem0 import Mem0Memory


@pytest.mark.external
async def test_mem0_update_and_retrieve_roundtrip():
    """update() stores a fact; retrieve() returns it in a subsequent search."""
    run_id = f"test-{uuid.uuid4()}"
    memory = Mem0Memory()

    messages = [
        Message(role="user", content="My favourite programming language is Rust."),
        Message(
            role="assistant", content="Got it, I'll remember that you prefer Rust."
        ),
    ]
    await memory.update(messages, run_id=run_id)

    results = await memory.retrieve("programming language preference", run_id=run_id)

    contents = " ".join(m.content for m in results).lower()
    assert "rust" in contents, f"Expected 'rust' in retrieved memories, got: {contents}"


@pytest.mark.external
async def test_mem0_retrieve_returns_messages():
    """retrieve() always returns a list of Message objects with role='system'."""
    run_id = f"test-{uuid.uuid4()}"
    memory = Mem0Memory()

    await memory.update(
        [Message(role="user", content="I live in Zurich.")],
        run_id=run_id,
    )

    results = await memory.retrieve("location", run_id=run_id)

    assert isinstance(results, list)
    for msg in results:
        assert isinstance(msg, Message)
        assert msg.role == "system"
        assert msg.content


@pytest.mark.external
async def test_mem0_empty_retrieve_returns_empty_list():
    """retrieve() returns [] when no relevant memories exist for the run."""
    run_id = f"empty-{uuid.uuid4()}"
    memory = Mem0Memory()

    results = await memory.retrieve("anything", run_id=run_id)

    assert results == []
