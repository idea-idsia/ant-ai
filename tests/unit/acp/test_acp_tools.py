from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import (
    ClientCapabilities,
    FileSystemCapabilities,
    ReadTextFileResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
)

from ant_ai.acp.tools import (
    _acp_capabilities,
    _acp_client,
    _acp_cwd,
    _acp_session_id,
    _resolve_path,
    acp_fs_read_file,
    acp_fs_write_file,
    acp_get_cwd,
    acp_list_directory,
    acp_send_plan,
    acp_terminal_run,
)


def _make_caps(*, read=True, write=True, terminal=True) -> ClientCapabilities:
    return ClientCapabilities(
        fs=FileSystemCapabilities(read_text_file=read, write_text_file=write),
        terminal=terminal,
    )


def _inject(client, session_id: str = "sess-1", caps=None, cwd: str | None = None):
    """Set contextvars and return tokens for cleanup."""
    t1 = _acp_client.set(client)
    t2 = _acp_session_id.set(session_id)
    t3 = _acp_capabilities.set(caps or _make_caps())
    t4 = _acp_cwd.set(cwd)
    return t1, t2, t3, t4


def _reset(*tokens):
    for t in tokens:
        t.var.reset(t)


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_read_file_calls_client():
    client = MagicMock()
    client.read_text_file = AsyncMock(
        return_value=ReadTextFileResponse(content="hello world")
    )
    tokens = _inject(client)
    try:
        result = await acp_fs_read_file("/tmp/foo.txt")
        assert result == "hello world"
        client.read_text_file.assert_awaited_once_with(
            path="/tmp/foo.txt", session_id="sess-1", line=None, limit=None
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_read_file_raises_when_cap_missing():
    client = MagicMock()
    caps = _make_caps(read=False)
    tokens = _inject(client, caps=caps)
    try:
        with pytest.raises(RuntimeError, match="fs/read_text_file"):
            await acp_fs_read_file("/tmp/foo.txt")
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_write_file_calls_client():
    client = MagicMock()
    client.write_text_file = AsyncMock(return_value=None)
    tokens = _inject(client)
    try:
        await acp_fs_write_file("/tmp/bar.txt", "content")
        client.write_text_file.assert_awaited_once_with(
            path="/tmp/bar.txt", content="content", session_id="sess-1"
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_write_file_raises_when_cap_missing():
    client = MagicMock()
    caps = _make_caps(write=False)
    tokens = _inject(client, caps=caps)
    try:
        with pytest.raises(RuntimeError, match="fs/write_text_file"):
            await acp_fs_write_file("/tmp/bar.txt", "content")
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_tools_raise_outside_prompt_turn():
    with pytest.raises(RuntimeError, match="Not running inside an ACP prompt turn"):
        await acp_fs_read_file("/tmp/foo.txt")


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminal_run_full_lifecycle():
    client = MagicMock()
    client.create_terminal = AsyncMock(return_value=MagicMock(terminal_id="term-1"))
    client.wait_for_terminal_exit = AsyncMock(
        return_value=WaitForTerminalExitResponse(exit_code=0, signal=None)
    )
    client.terminal_output = AsyncMock(
        return_value=TerminalOutputResponse(output="ls output", truncated=False)
    )
    client.release_terminal = AsyncMock(return_value=None)

    tokens = _inject(client)
    try:
        result = await acp_terminal_run("ls", args=["-la"], cwd="/tmp")
        assert result == "ls output"
        client.create_terminal.assert_awaited_once()
        client.wait_for_terminal_exit.assert_awaited_once_with(
            session_id="sess-1", terminal_id="term-1"
        )
        client.terminal_output.assert_awaited_once_with(
            session_id="sess-1", terminal_id="term-1"
        )
        client.release_terminal.assert_awaited_once_with(
            session_id="sess-1", terminal_id="term-1"
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_terminal_run_kills_on_timeout():
    import asyncio

    client = MagicMock()
    client.create_terminal = AsyncMock(return_value=MagicMock(terminal_id="term-2"))

    async def _hang(*args, **kwargs):
        await asyncio.sleep(9999)

    client.wait_for_terminal_exit = AsyncMock(side_effect=_hang)
    client.kill_terminal = AsyncMock(return_value=None)
    client.terminal_output = AsyncMock(
        return_value=TerminalOutputResponse(output="partial", truncated=True)
    )
    client.release_terminal = AsyncMock(return_value=None)

    tokens = _inject(client)
    try:
        result = await acp_terminal_run("sleep", args=["9999"], timeout_sec=0.05)
        assert result == "partial"
        client.kill_terminal.assert_awaited_once()
        client.release_terminal.assert_awaited_once()
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_terminal_run_raises_when_cap_missing():
    client = MagicMock()
    caps = _make_caps(terminal=False)
    tokens = _inject(client, caps=caps)
    try:
        with pytest.raises(RuntimeError, match="terminal"):
            await acp_terminal_run("ls")
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_terminal_tools_raise_outside_prompt_turn():
    with pytest.raises(RuntimeError, match="Not running inside an ACP prompt turn"):
        await acp_terminal_run("ls")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_plan_sends_update():
    from acp.schema import AgentPlanUpdate

    client = MagicMock()
    client.session_update = AsyncMock(return_value=None)
    tokens = _inject(client)
    try:
        result = await acp_send_plan(
            [{"content": "Step 1", "priority": "high", "status": "pending"}]
        )
        assert result == "plan sent"
        client.session_update.assert_awaited_once()
        call_kwargs = client.session_update.call_args.kwargs
        assert call_kwargs["session_id"] == "sess-1"
        update = call_kwargs["update"]
        assert isinstance(update, AgentPlanUpdate)
        assert len(update.entries) == 1
        assert update.entries[0].content == "Step 1"
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_send_plan_raises_outside_prompt_turn():
    with pytest.raises(RuntimeError, match="Not running inside an ACP prompt turn"):
        await acp_send_plan([{"content": "x", "priority": "low", "status": "pending"}])


# ---------------------------------------------------------------------------
# Path resolution — _resolve_path
#
# These tests document exactly what path our code sends to the ACP client.
# If something isn't working in acp-ui, compare against these expectations
# to determine whether the fault is here or in the client.
# ---------------------------------------------------------------------------


def test_resolve_path_absolute_is_unchanged():
    """An absolute path must never be modified."""
    t = _acp_cwd.set("/Users/alice/project")
    try:
        assert _resolve_path("/etc/hosts") == "/etc/hosts"
        assert (
            _resolve_path("/Users/alice/project/README.md")
            == "/Users/alice/project/README.md"
        )
    finally:
        _acp_cwd.reset(t)


def test_resolve_path_relative_joined_with_cwd():
    """A relative path is joined with the session cwd."""
    t = _acp_cwd.set("/Users/alice/project")
    try:
        assert _resolve_path("README.md") == "/Users/alice/project/README.md"
        assert _resolve_path("src/main.py") == "/Users/alice/project/src/main.py"
        assert _resolve_path("a/b/c.txt") == "/Users/alice/project/a/b/c.txt"
    finally:
        _acp_cwd.reset(t)


def test_resolve_path_relative_no_cwd_unchanged():
    """Without a cwd contextvar, a relative path is returned as-is."""
    # cwd contextvar defaults to None — no joining happens
    assert _resolve_path("README.md") == "README.md"


def test_resolve_path_relative_cwd_is_itself_relative():
    """If the user typed a relative cwd (e.g. 'IDeA'), we join but stay relative.
    This exposes the case where acp-ui receives a relative path and may fail.
    """
    t = _acp_cwd.set("IDeA")
    try:
        result = _resolve_path("README.md")
        # Result is still relative — acp-ui must handle this correctly.
        assert result == "IDeA/README.md"
    finally:
        _acp_cwd.reset(t)


# ---------------------------------------------------------------------------
# acp_get_cwd
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_cwd_returns_contextvar_value():
    """acp_get_cwd() must return exactly what the adapter stored."""
    client = MagicMock()
    tokens = _inject(client, cwd="/Users/alice/project")
    try:
        result = await acp_get_cwd()
        assert result == "/Users/alice/project"
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_get_cwd_raises_outside_prompt_turn():
    with pytest.raises(RuntimeError, match="Not running inside an ACP prompt turn"):
        await acp_get_cwd()


# ---------------------------------------------------------------------------
# acp_fs_read_file — path the client actually receives
#
# These tests pin the contract: what exact `path` argument does the ACP
# client's read_text_file() receive for a given tool call?  If reads fail
# in practice, check the logged path against these expectations.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_read_file_absolute_path_sent_unchanged():
    """An absolute path is forwarded to the client verbatim."""
    client = MagicMock()
    client.read_text_file = AsyncMock(return_value=ReadTextFileResponse(content="data"))
    tokens = _inject(client, cwd="/Users/alice/project")
    try:
        await acp_fs_read_file("/etc/hosts")
        client.read_text_file.assert_awaited_once_with(
            path="/etc/hosts", session_id="sess-1", line=None, limit=None
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_read_file_relative_path_resolved_against_cwd():
    """The client receives the resolved absolute path, not the bare filename."""
    client = MagicMock()
    client.read_text_file = AsyncMock(
        return_value=ReadTextFileResponse(content="# My Project")
    )
    tokens = _inject(client, cwd="/Users/alice/project")
    try:
        result = await acp_fs_read_file("README.md")
        assert result == "# My Project"
        # This is the contract: acp-ui must be able to read this path.
        client.read_text_file.assert_awaited_once_with(
            path="/Users/alice/project/README.md",
            session_id="sess-1",
            line=None,
            limit=None,
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_read_file_nested_relative_path():
    """Nested relative paths (e.g. 'src/main.py') are resolved correctly."""
    client = MagicMock()
    client.read_text_file = AsyncMock(return_value=ReadTextFileResponse(content="code"))
    tokens = _inject(client, cwd="/Users/alice/project")
    try:
        await acp_fs_read_file("src/main.py")
        client.read_text_file.assert_awaited_once_with(
            path="/Users/alice/project/src/main.py",
            session_id="sess-1",
            line=None,
            limit=None,
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_read_file_relative_cwd_produces_relative_result():
    """If the user set a relative cwd (e.g. 'IDeA'), the resolved path is
    still relative. This is the scenario that fails in acp-ui — the test
    documents it so we can tell our code is correct but the cwd is wrong.
    """
    client = MagicMock()
    client.read_text_file = AsyncMock(return_value=ReadTextFileResponse(content=""))
    tokens = _inject(client, cwd="IDeA")
    try:
        await acp_fs_read_file("README.md")
        # Our code correctly joins, but the result is still relative.
        # acp-ui will receive "IDeA/README.md" — a relative path it cannot serve.
        client.read_text_file.assert_awaited_once_with(
            path="IDeA/README.md", session_id="sess-1", line=None, limit=None
        )
    finally:
        _reset(*tokens)


@pytest.mark.asyncio
async def test_fs_read_file_with_line_and_limit():
    """line and limit are forwarded unchanged to the client."""
    client = MagicMock()
    client.read_text_file = AsyncMock(
        return_value=ReadTextFileResponse(content="line10")
    )
    tokens = _inject(client, cwd="/proj")
    try:
        await acp_fs_read_file("file.py", line=10, limit=5)
        client.read_text_file.assert_awaited_once_with(
            path="/proj/file.py", session_id="sess-1", line=10, limit=5
        )
    finally:
        _reset(*tokens)


# ---------------------------------------------------------------------------
# acp_fs_write_file — path the client actually receives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_file_relative_path_resolved_against_cwd():
    """Write path is resolved the same way as read path."""
    client = MagicMock()
    client.write_text_file = AsyncMock(return_value=None)
    tokens = _inject(client, cwd="/Users/alice/project")
    try:
        await acp_fs_write_file("output.txt", "hello")
        client.write_text_file.assert_awaited_once_with(
            path="/Users/alice/project/output.txt",
            content="hello",
            session_id="sess-1",
        )
    finally:
        _reset(*tokens)


# ---------------------------------------------------------------------------
# acp_list_directory — server-side directory listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_directory_defaults_to_cwd(tmp_path):
    (tmp_path / "README.md").write_text("hello")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    t = _acp_cwd.set(str(tmp_path))
    try:
        result = await acp_list_directory()
        lines = result.splitlines()
        assert "README.md" in lines
        assert "src/" in lines
    finally:
        _acp_cwd.reset(t)


@pytest.mark.asyncio
async def test_list_directory_absolute_path(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    t = _acp_cwd.set("/irrelevant")
    try:
        result = await acp_list_directory(str(tmp_path))
        assert "a.txt" in result.splitlines()
        assert "b.txt" in result.splitlines()
    finally:
        _acp_cwd.reset(t)


@pytest.mark.asyncio
async def test_list_directory_relative_path_resolved_against_cwd(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "file.py").write_text("")
    t = _acp_cwd.set(str(tmp_path))
    try:
        result = await acp_list_directory("sub")
        assert "file.py" in result.splitlines()
    finally:
        _acp_cwd.reset(t)


@pytest.mark.asyncio
async def test_list_directory_marks_subdirs_with_slash(tmp_path):
    (tmp_path / "file.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    t = _acp_cwd.set(str(tmp_path))
    try:
        result = await acp_list_directory()
        assert "subdir/" in result.splitlines()
        assert "file.txt" in result.splitlines()
        assert "file.txt/" not in result.splitlines()
    finally:
        _acp_cwd.reset(t)


@pytest.mark.asyncio
async def test_list_directory_empty_dir(tmp_path):
    t = _acp_cwd.set(str(tmp_path))
    try:
        result = await acp_list_directory()
        assert result == "(empty directory)"
    finally:
        _acp_cwd.reset(t)


@pytest.mark.asyncio
async def test_list_directory_raises_on_missing_path(tmp_path):
    t = _acp_cwd.set(str(tmp_path))
    try:
        with pytest.raises(FileNotFoundError):
            await acp_list_directory("does_not_exist")
    finally:
        _acp_cwd.reset(t)


@pytest.mark.asyncio
async def test_list_directory_raises_on_file_path(tmp_path):
    (tmp_path / "file.txt").write_text("")
    t = _acp_cwd.set(str(tmp_path))
    try:
        with pytest.raises(NotADirectoryError):
            await acp_list_directory("file.txt")
    finally:
        _acp_cwd.reset(t)
