from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from typing import Any, Literal, cast

from acp.interfaces import Client
from acp.schema import (
    AgentPlanUpdate,
    ClientCapabilities,
    PlanEntry,
)
from loguru import logger

from ant_ai.tools.tool import tool

_acp_client: ContextVar[Client | None] = ContextVar("_acp_client", default=None)
_acp_session_id: ContextVar[str | None] = ContextVar("_acp_session_id", default=None)
_acp_capabilities: ContextVar[ClientCapabilities | None] = ContextVar(
    "_acp_capabilities", default=None
)
_acp_cwd: ContextVar[str | None] = ContextVar("_acp_cwd", default=None)


def _require_context() -> tuple[Client, str]:
    client = _acp_client.get()
    session_id = _acp_session_id.get()
    if client is None or session_id is None:
        raise RuntimeError(
            "Not running inside an ACP prompt turn. "
            "ACP tool functions can only be called during a session/prompt invocation."
        )
    return client, session_id


def _require_fs_read() -> tuple[Client, str]:
    client, session_id = _require_context()
    caps = _acp_capabilities.get()
    if not (caps and caps.fs and caps.fs.read_text_file):
        raise RuntimeError(
            "Client does not support fs/read_text_file. "
            "Check clientCapabilities.fs.readTextFile during initialize()."
        )
    return client, session_id


def _require_fs_write() -> tuple[Client, str]:
    client, session_id = _require_context()
    caps = _acp_capabilities.get()
    if not (caps and caps.fs and caps.fs.write_text_file):
        raise RuntimeError(
            "Client does not support fs/write_text_file. "
            "Check clientCapabilities.fs.writeTextFile during initialize()."
        )
    return client, session_id


def _require_terminal() -> tuple[Client, str]:
    client, session_id = _require_context()
    caps = _acp_capabilities.get()
    if not (caps and caps.terminal):
        raise RuntimeError(
            "Client does not support terminal/*. "
            "Check clientCapabilities.terminal during initialize()."
        )
    return client, session_id


@tool
async def acp_get_cwd() -> str:
    """Return the working directory set by the user for this IDE session.

    Call this to get the base directory when the user refers to files without
    giving a full path. Then use os.path.join(cwd, filename) to build the
    full path and pass it to acp_fs_read_file or acp_terminal_run.

    Example: cwd="/Users/alice/myproject", read "README.md" →
        acp_fs_read_file("/Users/alice/myproject/README.md")

    Returns:
        Path to the session's working directory (may be absolute or relative
        depending on what the user typed in the client).
    """
    cwd = _acp_cwd.get()
    if cwd is None:
        raise RuntimeError(
            "Not running inside an ACP prompt turn. "
            "ACP tool functions can only be called during a session/prompt invocation."
        )
    return cwd


def _resolve_path(path: str) -> str:
    """Resolve a possibly-relative path against the session cwd."""
    if not os.path.isabs(path):
        cwd = _acp_cwd.get()
        if cwd:
            resolved = os.path.join(cwd, path)
            logger.debug("acp: resolved '{}' → '{}' (cwd='{}')", path, resolved, cwd)
            return resolved
    return path


@tool
async def acp_list_directory(path: str | None = None) -> str:
    """List files and subdirectories at a path on the agent's filesystem.

    Use this to explore the project before reading specific files.
    Directories are shown with a trailing '/'. Relative paths are resolved
    against the session working directory. Defaults to the working directory
    when no path is given.

    Args:
        path: Directory to list. Absolute or relative to cwd. Defaults to cwd.

    Returns:
        Newline-separated entries — directories have a trailing '/'.
    """
    cwd = _acp_cwd.get()
    target = cwd or "." if path is None else _resolve_path(path)

    logger.debug("acp_list_directory: listing '{}'", target)
    try:
        entries = sorted(os.listdir(target))
        lines = [
            e + "/" if os.path.isdir(os.path.join(target, e)) else e for e in entries
        ]
        result = "\n".join(lines) if lines else "(empty directory)"
        logger.debug("acp_list_directory: {} entries", len(lines))
        return result
    except Exception as exc:
        logger.error("acp_list_directory failed: path='{}' error={}", target, exc)
        raise


@tool
async def acp_fs_read_file(
    path: str,
    line: int | None = None,
    limit: int | None = None,
) -> str:
    """Read a text file from the IDE's filesystem via ACP.

    Relative paths are automatically resolved against the session working
    directory, so you can pass just a filename like "README.md" and it will
    be read from the cwd. Use acp_get_cwd() to see what that directory is.

    Args:
        path: Path to the file — absolute or relative to the working directory.
        line: 1-based starting line (optional).
        limit: Maximum number of lines to return (optional).

    Returns:
        File content as a string.
    """
    client, session_id = _require_fs_read()
    resolved = _resolve_path(path)
    logger.debug("acp_fs_read_file: path='{}' resolved='{}'", path, resolved)
    try:
        response = await client.read_text_file(
            path=resolved, session_id=session_id, line=line, limit=limit
        )
        logger.debug("acp_fs_read_file: got {} chars", len(response.content))
        return response.content
    except Exception as exc:
        logger.error("acp_fs_read_file failed: path='{}' error={}", resolved, exc)
        raise


@tool
async def acp_fs_write_file(path: str, content: str) -> None:
    """Write or overwrite a text file in the IDE's filesystem via ACP.

    Relative paths are resolved against the session working directory.

    Args:
        path: Path to the file — absolute or relative to the working directory.
        content: Text content to write.
    """
    client, session_id = _require_fs_write()
    resolved = _resolve_path(path)
    logger.debug(
        "acp_fs_write_file: path='{}' resolved='{}' len={}",
        path,
        resolved,
        len(content),
    )
    try:
        await client.write_text_file(
            path=resolved, content=content, session_id=session_id
        )
        logger.debug("acp_fs_write_file: done")
    except Exception as exc:
        logger.error("acp_fs_write_file failed: path='{}' error={}", resolved, exc)
        raise


@tool
async def acp_terminal_create(
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: list[dict[str, str]] | None = None,
    output_byte_limit: int | None = None,
) -> str:
    """Create a terminal in the IDE and start executing a command.

    Returns the terminal_id immediately without waiting for completion.
    Use acp_terminal_wait_for_exit() to block until done.

    Args:
        command: The command to execute.
        args: Optional list of arguments.
        cwd: Optional absolute working directory.
        env: Optional environment variables as list of {"name": ..., "value": ...}.
        output_byte_limit: Maximum output bytes before truncation.

    Returns:
        terminal_id string for subsequent terminal operations.
    """
    from acp.schema import EnvVariable

    client, session_id = _require_terminal()
    env_vars = (
        [EnvVariable(name=e["name"], value=e["value"]) for e in env] if env else None
    )
    response = await client.create_terminal(
        command=command,
        session_id=session_id,
        args=args,
        cwd=cwd,
        env=env_vars,
        output_byte_limit=output_byte_limit,
    )
    return response.terminal_id


@tool
async def acp_terminal_output(terminal_id: str) -> str:
    """Get the current output of a running or completed terminal.

    Args:
        terminal_id: Terminal identifier from acp_terminal_create().

    Returns:
        Captured output string (may be truncated if output_byte_limit was hit).
    """
    client, session_id = _require_context()
    response = await client.terminal_output(
        session_id=session_id, terminal_id=terminal_id
    )
    return response.output


@tool
async def acp_terminal_wait_for_exit(terminal_id: str) -> dict[str, Any]:
    """Block until a terminal command completes.

    Args:
        terminal_id: Terminal identifier from acp_terminal_create().

    Returns:
        Dict with 'exit_code' (int | None) and 'signal' (str | None).
    """
    client, session_id = _require_context()
    response = await client.wait_for_terminal_exit(
        session_id=session_id, terminal_id=terminal_id
    )
    return {"exit_code": response.exit_code, "signal": response.signal}


@tool
async def acp_terminal_kill(terminal_id: str) -> None:
    """Kill a running terminal process (keeps the terminal for output retrieval).

    Args:
        terminal_id: Terminal identifier from acp_terminal_create().
    """
    client, session_id = _require_context()
    await client.kill_terminal(session_id=session_id, terminal_id=terminal_id)


@tool
async def acp_terminal_release(terminal_id: str) -> None:
    """Kill any running process and release terminal resources.

    After release, the terminal_id is no longer valid.

    Args:
        terminal_id: Terminal identifier from acp_terminal_create().
    """
    client, session_id = _require_context()
    await client.release_terminal(session_id=session_id, terminal_id=terminal_id)


@tool
async def acp_terminal_run(
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: list[dict[str, str]] | None = None,
    timeout_sec: float | None = None,
) -> str:
    """Run a command in the IDE terminal and return its output.

    Convenience wrapper: creates a terminal, waits for the command to finish
    (killing it on timeout), retrieves output, and releases resources.

    Args:
        command: The command to execute.
        args: Optional list of arguments.
        cwd: Optional absolute working directory.
        env: Optional environment variables as list of {"name": ..., "value": ...}.
        timeout_sec: If set, kills the process after this many seconds.

    Returns:
        Combined stdout/stderr output from the command.
    """
    terminal_id = await acp_terminal_create(
        command=command, args=args, cwd=cwd, env=env
    )
    try:
        if timeout_sec is not None:
            try:
                await asyncio.wait_for(
                    acp_terminal_wait_for_exit(terminal_id), timeout=timeout_sec
                )
            except TimeoutError:
                await acp_terminal_kill(terminal_id)
        else:
            await acp_terminal_wait_for_exit(terminal_id)
        return await acp_terminal_output(terminal_id)
    finally:
        await acp_terminal_release(terminal_id)


@tool
async def acp_send_plan(entries: list[dict[str, str]]) -> str:
    """Send an agent plan update to the IDE.

    Each entry must have 'content', 'priority' ("high"|"medium"|"low"),
    and 'status' ("pending"|"in_progress"|"completed").

    Args:
        entries: List of plan entry dicts.

    Returns:
        Confirmation string "plan sent".
    """
    client, session_id = _require_context()
    plan_entries = [
        PlanEntry(
            content=e["content"],
            priority=cast(Literal["high", "medium", "low"], e["priority"]),
            status=cast(Literal["pending", "in_progress", "completed"], e["status"]),
        )
        for e in entries
    ]
    await client.session_update(
        session_id=session_id,
        update=AgentPlanUpdate(session_update="plan", entries=plan_entries),
    )
    return "plan sent"


ACP_SESSION_TOOLS = [acp_get_cwd]
ACP_FILESYSTEM_TOOLS = [acp_list_directory, acp_fs_read_file, acp_fs_write_file]
ACP_TERMINAL_TOOLS = [
    acp_terminal_run,
    acp_terminal_create,
    acp_terminal_output,
    acp_terminal_wait_for_exit,
    acp_terminal_kill,
    acp_terminal_release,
]
ACP_PLAN_TOOLS = [acp_send_plan]
ACP_ALL_TOOLS = [
    *ACP_SESSION_TOOLS,
    *ACP_FILESYSTEM_TOOLS,
    *ACP_TERMINAL_TOOLS,
    *ACP_PLAN_TOOLS,
]
