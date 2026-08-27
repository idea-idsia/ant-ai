from __future__ import annotations

import pytest

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext
from ant_ai.tools.registry import ToolRegistry


@pytest.mark.unit
async def test_search_returns_retrieved_message_contents(stub_memory):
    stub_memory.set_entries(
        [Message(role="system", content="likes rust"), Message(role="system")]
    )

    result = await stub_memory.search("preferences")

    assert result == ["likes rust"]


@pytest.mark.unit
async def test_search_returns_empty_list_when_nothing_found(stub_memory):
    assert await stub_memory.search("anything") == []


@pytest.mark.unit
async def test_search_forwards_query_and_ctx(stub_memory):
    ctx = InvocationContext(session_id="s1", user_id="alice")

    await stub_memory.search("preferences", ctx=ctx)

    assert stub_memory.retrieve_calls == [
        {"query": "preferences", "top_k": 5, "ctx": ctx}
    ]


@pytest.mark.unit
async def test_add_calls_update_with_system_messages(stub_memory):
    result = await stub_memory.add(["likes rust", "lives in Zurich"])

    assert result == "Saved 2 fact(s) to memory."
    assert len(stub_memory.update_calls) == 1
    messages: list[Message] = stub_memory.update_calls[0]["messages"]
    assert [m.content for m in messages] == ["likes rust", "lives in Zurich"]
    assert all(m.role == "system" for m in messages)


@pytest.mark.unit
async def test_add_forwards_ctx(stub_memory):
    ctx = InvocationContext(session_id="s1", user_id="alice")

    await stub_memory.add(["fact"], ctx=ctx)

    assert stub_memory.update_calls[0]["ctx"] is ctx


@pytest.mark.unit
async def test_add_with_no_facts_is_a_noop(stub_memory):
    result = await stub_memory.add([])

    assert result == "Nothing to save."
    assert stub_memory.update_calls == []


@pytest.mark.unit
def test_registered_as_search_and_add_tools(stub_memory):
    registry = ToolRegistry([stub_memory])

    assert "StubMemory_search" in registry
    assert "StubMemory_add" in registry
    assert registry["StubMemory_search"].wants_context is True
    assert registry["StubMemory_add"].wants_context is True
    assert "ctx" not in registry["StubMemory_search"].parameters["properties"]
    assert "ctx" not in registry["StubMemory_add"].parameters["properties"]
    # retrieve/update (the storage protocol methods) must never be
    # LLM-callable directly.
    assert "StubMemory_retrieve" not in registry
    assert "StubMemory_update" not in registry
