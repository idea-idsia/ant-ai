from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ant_ai.core.types import InvocationContext
from ant_ai.memory.backends.mem0 import Mem0Memory, _resolve_scope
from ant_ai.tools.registry import ToolRegistry

ALICE = InvocationContext(session_id="s1", user_id="alice")


@pytest.fixture
def mem0() -> Mem0Memory:
    with patch("ant_ai.memory.backends.mem0.AsyncMemoryClient") as mock_client_cls:
        mock_client_cls.return_value = AsyncMock()
        memory = Mem0Memory(api_key="test-key")
        memory._client.search.return_value = {"results": []}
        yield memory


@pytest.mark.unit
def test_resolve_scope_with_user_id_ctx_returns_user_id_filter():
    assert _resolve_scope(ALICE) == {"user_id": "alice"}


@pytest.mark.unit
def test_resolve_scope_with_session_only_ctx_falls_back_to_run_id():
    ctx = InvocationContext(session_id="s1", user_id=None)
    assert _resolve_scope(ctx) == {"run_id": "s1"}


@pytest.mark.unit
def test_resolve_scope_raises_when_unscoped():
    with pytest.raises(ValueError, match="requires scoping information"):
        _resolve_scope(None)


@pytest.mark.unit
async def test_search_raises_when_unscoped(mem0: Mem0Memory):
    """ToolStep's generic exception handling turns this into a clean
    tool-result error for the LLM."""
    with pytest.raises(ValueError, match="requires scoping information"):
        await mem0.search("query")


@pytest.mark.unit
async def test_add_raises_when_unscoped(mem0: Mem0Memory):
    with pytest.raises(ValueError, match="requires scoping information"):
        await mem0.add(["likes rust"])


@pytest.mark.unit
async def test_search_returns_the_memories_of_the_user(mem0: Mem0Memory):
    mem0._client.search.return_value = {
        "results": [{"memory": "likes rust"}, {"memory": ""}, {"id": "x"}]
    }

    assert await mem0.search("preferences", ctx=ALICE) == ["likes rust"]

    query, kwargs = (
        mem0._client.search.await_args.args[0],
        mem0._client.search.await_args.kwargs,
    )
    assert query == "preferences"
    assert kwargs["options"].filters == {"user_id": "alice"}
    assert kwargs["options"].top_k == 5


@pytest.mark.unit
async def test_search_returns_at_most_top_k(mem0: Mem0Memory):
    mem0.top_k = 2
    await mem0.search("preferences", ctx=ALICE)
    assert mem0._client.search.await_args.kwargs["options"].top_k == 2


@pytest.mark.unit
async def test_add_saves_each_fact_for_the_user(mem0: Mem0Memory):
    result = await mem0.add(["likes rust", "", "lives in Zurich"], ctx=ALICE)

    assert result == "Saved 2 fact(s) to memory."
    mem0._client.add.assert_awaited_once_with(
        [
            {"role": "system", "content": "likes rust"},
            {"role": "system", "content": "lives in Zurich"},
        ],
        user_id="alice",
    )


@pytest.mark.unit
async def test_add_with_no_facts_saves_nothing(mem0: Mem0Memory):
    assert await mem0.add([], ctx=ALICE) == "Nothing to save."
    mem0._client.add.assert_not_awaited()


@pytest.mark.unit
def test_registered_as_search_and_add_tools(mem0: Mem0Memory):
    registry = ToolRegistry([mem0])

    assert {t["function"]["name"] for t in registry.to_serialized()} == {
        "Mem0Memory_search",
        "Mem0Memory_add",
    }
    for name in ("Mem0Memory_search", "Mem0Memory_add"):
        assert name in registry
        assert registry[name].wants_context is True
        assert "ctx" not in registry[name].parameters["properties"]
