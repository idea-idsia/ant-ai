from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext
from ant_ai.memory.backends.mem0 import Mem0Memory, _resolve_ctx


@pytest.fixture
def mem0() -> Mem0Memory:
    with patch("ant_ai.memory.backends.mem0.AsyncMemoryClient") as mock_client_cls:
        mock_client_cls.return_value = AsyncMock()
        memory = Mem0Memory(api_key="test-key")
        memory._client.search.return_value = {"results": []}
        yield memory


@pytest.mark.unit
def test_resolve_ctx_with_user_id_ctx_returns_user_id_filter():
    ctx = InvocationContext(session_id="s1", user_id="alice")
    assert _resolve_ctx({"ctx": ctx}) == {"user_id": "alice"}


@pytest.mark.unit
def test_resolve_ctx_with_session_only_ctx_falls_back_to_run_id():
    ctx = InvocationContext(session_id="s1", user_id=None)
    assert _resolve_ctx({"ctx": ctx}) == {"run_id": "s1"}


@pytest.mark.unit
def test_resolve_ctx_with_bare_user_id_kwarg():
    assert _resolve_ctx({"user_id": "alice"}) == {"user_id": "alice"}


@pytest.mark.unit
def test_resolve_ctx_raises_when_unscoped():
    with pytest.raises(ValueError, match="requires scoping information"):
        _resolve_ctx({})


@pytest.mark.unit
async def test_mem0memory_retrieve_raises_when_unscoped(mem0: Mem0Memory):
    with pytest.raises(ValueError, match="requires scoping information"):
        await mem0.retrieve("query")


@pytest.mark.unit
async def test_mem0memory_update_raises_when_unscoped(mem0: Mem0Memory):
    with pytest.raises(ValueError, match="requires scoping information"):
        await mem0.update([Message(role="user", content="hi")])


@pytest.mark.unit
async def test_mem0memory_retrieve_succeeds_with_ctx(mem0: Mem0Memory):
    ctx = InvocationContext(session_id="s1", user_id="alice")
    await mem0.retrieve("query", ctx=ctx)
    mem0._client.search.assert_awaited_once()


@pytest.mark.unit
async def test_mem0memory_search_tool_method_raises_when_unscoped(mem0: Mem0Memory):
    """The LLM-facing `search` tool method surfaces the same scoping
    requirement as the underlying `retrieve` — ToolStep's generic exception
    handling turns this into a clean tool-result error for the LLM."""
    with pytest.raises(ValueError, match="requires scoping information"):
        await mem0.search("query")
