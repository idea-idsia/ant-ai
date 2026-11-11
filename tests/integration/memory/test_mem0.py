from __future__ import annotations

import asyncio
import uuid

import pytest

from ant_ai.core.types import InvocationContext
from ant_ai.memory.backends.mem0 import Mem0Memory


async def _poll(
    memory: Mem0Memory,
    query: str,
    keyword: str,
    ctx: InvocationContext,
    *,
    timeout: int = 30,
    interval: int = 3,
) -> str:
    """Poll search() until keyword appears in results or timeout expires."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        contents = " ".join(await memory.search(query, ctx=ctx)).lower()
        if keyword in contents:
            return contents
        await asyncio.sleep(interval)
    return ""


def _user(prefix: str) -> InvocationContext:
    return InvocationContext(session_id="s1", user_id=f"{prefix}-{uuid.uuid4()}")


@pytest.mark.external
@pytest.mark.mem0
async def test_mem0_add_and_search_roundtrip():
    """add() stores a fact; search() returns it in a subsequent search."""
    ctx = _user("test")
    memory = Mem0Memory()

    await memory.add(["My favourite programming language is Rust."], ctx=ctx)

    contents = await _poll(memory, "programming language preference", "rust", ctx)
    assert "rust" in contents, (
        f"Expected 'rust' in retrieved memories, got: {contents!r}"
    )


@pytest.mark.external
@pytest.mark.mem0
async def test_mem0_search_returns_strings():
    """search() always returns a list of non-empty strings."""
    ctx = _user("test")
    memory = Mem0Memory()

    await memory.add(["I live in Zurich."], ctx=ctx)

    results = await memory.search("location", ctx=ctx)

    assert isinstance(results, list)
    assert all(isinstance(r, str) and r for r in results)


@pytest.mark.external
@pytest.mark.mem0
async def test_mem0_empty_search_returns_empty_list():
    """search() returns [] when no relevant memories exist for the user."""
    memory = Mem0Memory()

    assert await memory.search("anything", ctx=_user("empty")) == []
