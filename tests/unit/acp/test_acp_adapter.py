from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import TextContentBlock

from ant_ai.acp.adapter import ACPAdapter
from ant_ai.core.events import FinalAnswerEvent, StartEvent


def _make_adapter() -> ACPAdapter:
    agent = MagicMock()
    agent.name = "TestAgent"
    workflow = MagicMock()
    return ACPAdapter(agent=agent, workflow=workflow)


@pytest.mark.asyncio
async def test_initialize_returns_agent_name():
    adapter = _make_adapter()
    result = await adapter.initialize(protocol_version=1)
    assert result.protocol_version == 1
    assert result.agent_info.name == "TestAgent"
    assert result.agent_capabilities.load_session is True


@pytest.mark.asyncio
async def test_new_session_returns_unique_ids():
    adapter = _make_adapter()
    r1 = await adapter.new_session(cwd="/tmp")
    r2 = await adapter.new_session(cwd="/tmp")
    assert r1.session_id != r2.session_id
    assert r1.session_id in adapter._sessions
    assert r2.session_id in adapter._sessions


@pytest.mark.asyncio
async def test_load_session_returns_none_for_unknown():
    adapter = _make_adapter()
    result = await adapter.load_session(cwd="/tmp", session_id="nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_load_session_returns_response_for_known():
    adapter = _make_adapter()
    resp = await adapter.new_session(cwd="/tmp")
    result = await adapter.load_session(cwd="/tmp", session_id=resp.session_id)
    assert result is not None


@pytest.mark.asyncio
async def test_list_sessions_reflects_created_sessions():
    adapter = _make_adapter()
    r1 = await adapter.new_session(cwd="/tmp")
    r2 = await adapter.new_session(cwd="/tmp")
    listed = await adapter.list_sessions()
    ids = {s.session_id for s in listed.sessions}
    assert r1.session_id in ids
    assert r2.session_id in ids


@pytest.mark.asyncio
async def test_prompt_streams_updates_and_returns_end_turn():
    adapter = _make_adapter()
    session = await adapter.new_session(cwd="/tmp")

    client = MagicMock()
    client.session_update = AsyncMock()
    adapter.on_connect(client)

    final_event = FinalAnswerEvent(content="Hello!")

    async def _fake_stream(**_):
        yield StartEvent()
        yield final_event

    adapter._workflow.create_state.return_value = MagicMock()
    adapter._workflow.stream = MagicMock(return_value=_fake_stream())

    prompt_blocks = [TextContentBlock(type="text", text="Hi")]
    result = await adapter.prompt(prompt=prompt_blocks, session_id=session.session_id)

    assert result.stop_reason == "end_turn"
    client.session_update.assert_awaited_once()
    call_kwargs = client.session_update.call_args.kwargs
    assert call_kwargs["session_id"] == session.session_id


@pytest.mark.asyncio
async def test_prompt_appends_history():
    adapter = _make_adapter()
    session = await adapter.new_session(cwd="/tmp")

    client = MagicMock()
    client.session_update = AsyncMock()
    adapter.on_connect(client)

    final_event = FinalAnswerEvent(content="Reply")

    async def _fake_stream(**_):
        yield final_event

    adapter._workflow.create_state.return_value = MagicMock()
    adapter._workflow.stream = MagicMock(return_value=_fake_stream())

    await adapter.prompt(
        prompt=[TextContentBlock(type="text", text="Question")],
        session_id=session.session_id,
    )

    history = adapter._sessions[session.session_id]
    assert any(m.role == "user" for m in history)
    assert any(m.role == "assistant" and m.content == "Reply" for m in history)
