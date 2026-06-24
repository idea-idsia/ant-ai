from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import (
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    FileSystemCapabilities,
    HttpMcpServer,
    TextContentBlock,
    TextResourceContents,
)

from ant_ai.acp.adapter import ACPAdapter
from ant_ai.core.events import FinalAnswerEvent, StartEvent


def _make_adapter(slash_commands=None) -> ACPAdapter:
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.tools = []
    workflow = MagicMock()
    return ACPAdapter(agent=agent, workflow=workflow, slash_commands=slash_commands)


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


# ---------------------------------------------------------------------------
# initialize — capabilities + MCP advertisement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_stores_client_capabilities():
    adapter = _make_adapter()
    caps = ClientCapabilities(fs=FileSystemCapabilities(read_text_file=True))
    await adapter.initialize(protocol_version=1, client_capabilities=caps)
    assert adapter._client_capabilities is caps


@pytest.mark.asyncio
async def test_initialize_advertises_mcp_http_and_sse():
    adapter = _make_adapter()
    result = await adapter.initialize(protocol_version=1)
    mcp = result.agent_capabilities.mcp_capabilities
    assert mcp is not None
    assert mcp.http is True
    assert mcp.sse is True


# ---------------------------------------------------------------------------
# Embedded resource content blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_includes_embedded_resource_text():
    adapter = _make_adapter()
    session = await adapter.new_session(cwd="/tmp")

    client = MagicMock()
    client.session_update = AsyncMock()
    adapter.on_connect(client)

    captured_text: list[str] = []

    async def _fake_stream(**_):
        yield FinalAnswerEvent(content="ok")

    def _capture_state(messages):
        # Capture the text passed to workflow
        captured_text.extend(m.content for m in messages if m.role == "user")
        return MagicMock()

    adapter._workflow.create_state.side_effect = _capture_state
    adapter._workflow.stream = MagicMock(return_value=_fake_stream())

    resource = EmbeddedResourceContentBlock(
        type="resource",
        resource=TextResourceContents(uri="file:///foo.py", text="x = 1"),
    )
    await adapter.prompt(
        prompt=[
            TextContentBlock(type="text", text="Look at this:"),
            resource,
        ],
        session_id=session.session_id,
    )

    assert len(captured_text) == 1
    assert "Look at this:" in captured_text[0]
    assert "[File: file:///foo.py]" in captured_text[0]
    assert "x = 1" in captured_text[0]


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_commands_sent_on_first_prompt_only():
    cmd = AvailableCommand(name="test-cmd", description="A test command")
    adapter = _make_adapter(slash_commands=[cmd])
    session = await adapter.new_session(cwd="/tmp")

    client = MagicMock()
    client.session_update = AsyncMock()
    adapter.on_connect(client)

    async def _fake_stream(**_):
        yield FinalAnswerEvent(content="done")

    adapter._workflow.create_state.return_value = MagicMock()
    adapter._workflow.stream = MagicMock(side_effect=lambda **_: _fake_stream())

    # First prompt — should send slash commands + FinalAnswer update = 2 calls
    await adapter.prompt(
        prompt=[TextContentBlock(type="text", text="hi")],
        session_id=session.session_id,
    )
    first_call_count = client.session_update.call_count

    # Verify slash commands update was sent
    updates = [c.kwargs["update"] for c in client.session_update.call_args_list]
    assert any(isinstance(u, AvailableCommandsUpdate) for u in updates)

    # Second prompt — should NOT send slash commands again
    client.session_update.reset_mock()
    adapter._workflow.stream = MagicMock(side_effect=lambda **_: _fake_stream())
    await adapter.prompt(
        prompt=[TextContentBlock(type="text", text="hi again")],
        session_id=session.session_id,
    )
    second_updates = [c.kwargs["update"] for c in client.session_update.call_args_list]
    assert not any(isinstance(u, AvailableCommandsUpdate) for u in second_updates)


# ---------------------------------------------------------------------------
# close_session cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_session_cleans_up_state():
    cmd = AvailableCommand(name="cmd", description="desc")
    adapter = _make_adapter(slash_commands=[cmd])
    session = await adapter.new_session(cwd="/project")

    # Simulate a prompt so commands_sent is populated
    client = MagicMock()
    client.session_update = AsyncMock()
    adapter.on_connect(client)

    async def _fake_stream(**_):
        yield FinalAnswerEvent(content="done")

    adapter._workflow.create_state.return_value = MagicMock()
    adapter._workflow.stream = MagicMock(side_effect=lambda **_: _fake_stream())
    await adapter.prompt(
        prompt=[TextContentBlock(type="text", text="hi")],
        session_id=session.session_id,
    )

    sid = session.session_id
    assert sid in adapter._session_commands_sent
    assert sid in adapter._session_agents

    await adapter.close_session(session_id=sid)

    assert sid not in adapter._sessions
    assert sid not in adapter._session_agents
    assert sid not in adapter._session_cwds
    assert sid not in adapter._session_commands_sent


# ---------------------------------------------------------------------------
# Per-session MCP tool injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_session_with_http_mcp_server_extends_agent_tools():
    from ant_ai.tools.tool import Tool

    fake_tool = MagicMock(spec=Tool)
    fake_tool.name = "mcp_tool"

    with MagicMock() as mock_loader:
        import ant_ai.acp.adapter as adapter_mod

        original = adapter_mod.mcp_tools_from_url
        adapter_mod.mcp_tools_from_url = AsyncMock(return_value=[fake_tool])
        try:
            adapter = _make_adapter()
            srv = HttpMcpServer(
                type="http", name="my-mcp", url="http://localhost:8000/mcp", headers=[]
            )
            resp = await adapter.new_session(cwd="/tmp", mcp_servers=[srv])
            sid = resp.session_id
            session_agent = adapter._session_agents[sid]
            assert session_agent is not adapter._agent
            adapter_mod.mcp_tools_from_url.assert_awaited_once_with(
                "http://localhost:8000/mcp"
            )
        finally:
            adapter_mod.mcp_tools_from_url = original
