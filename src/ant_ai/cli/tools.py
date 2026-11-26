from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypedDict, cast

from ant_ai.tools.builtins.filesystem_tool import FilesystemTool
from ant_ai.tools.builtins.shell_tool import ShellTool
from ant_ai.tools.tool import Tool, ToolError, tool

type Confirm = Callable[[str], Awaitable[bool]]
"""Asks the user to allow an action described in one line; True to go ahead."""


class CommandResult(TypedDict):
    """What `run_command` returns."""

    stdout: str
    stderr: str
    returncode: int


def coding_tools(
    workspace: Path, *, confirm: Confirm, timeout: int = 120
) -> list[Tool]:
    """The tools of the terminal agent.

    Reading is free; editing or writing a file and running a command go
    through `confirm` first, and a refusal reaches the model as a tool error
    it can react to.

    Args:
        workspace: The folder of the file tools, which cannot reach outside
            it. A command starts there but runs with the user's own rights:
            it can reach any file the user can, hence the question first.
        confirm: Asked before each write, edit or command; pass one that
            returns True to allow everything.
        timeout: Seconds a command may run.

    Returns:
        `read_file`, `list_dir`, `search`, `edit_file`, `write_file`, `run_command`.
    """
    fs = FilesystemTool(workspace_root=str(workspace))
    shell = ShellTool(cwd=str(workspace), timeout=timeout)
    # One question at a time: tool calls of a step run concurrently.
    asking = asyncio.Lock()

    async def ask(action: str) -> None:
        async with asking:
            if not await confirm(action):
                raise ToolError(f"The user did not allow: {action}")

    @tool
    def read_file(path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read a text file of the workspace, relative to it. For a long file, pass `offset` (first line, from 1) and `limit` (number of lines)."""
        return fs.read_file(path, offset, limit)

    @tool
    def list_dir(path: str = ".", depth: int = 2) -> list[str]:
        """List files and directories under `path`, `depth` levels deep, skipping what .gitignore ignores."""
        return fs.list_dir(path, depth)

    @tool
    def search(pattern: str, path: str = ".") -> str:
        """Search a regular expression in the files under `path`; returns `file:line: text` matches."""
        return fs.search(pattern, path)

    @tool
    async def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace `old_string` with `new_string` in a file, leaving the rest unchanged. `old_string` must appear once; include surrounding text to make it unique, or set `replace_all`. Prefer this to rewriting a whole file."""
        await ask(f"edit {path}")
        return fs.edit_file(path, old_string, new_string, replace_all)

    @tool
    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file of the workspace with `content`."""
        await ask(f"write {path} ({len(content.splitlines())} lines)")
        return fs.write_file(path, content)

    @tool
    async def run_command(command: str) -> CommandResult:
        """Run a shell command in the workspace and return stdout, stderr and returncode."""
        await ask(f"run `{command}`")
        return cast(CommandResult, await asyncio.to_thread(shell.run, command))

    return [read_file, list_dir, search, edit_file, write_file, run_command]
