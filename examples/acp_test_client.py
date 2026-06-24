"""Interactive ACP test client — exercises filesystem, terminal, plan and slash-command features.

Run with:
    uv run python examples/acp_test_client.py

What it does
------------
1. Spawns examples/acp_agent.py as a stdio subprocess.
2. Acts as a full ACP client — implements fs/read, fs/write, terminal/*, and
   request_permission so the agent can exercise all capabilities.
3. Sends prompts and prints every session update in coloured, human-readable format.

Requires an LLM API key in the environment (e.g. OPENAI_API_KEY).
Override the model via ACP_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

from acp import spawn_agent_process
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CreateTerminalResponse,
    FileSystemCapabilities,
    Implementation,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    SelectedPermissionOutcome,
    TerminalExitStatus,
    TerminalOutputResponse,
    ToolCallProgress,
    ToolCallStart,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
DIM = "\033[2m"
RED = "\033[31m"


def _hdr(label: str, colour: str = CYAN) -> str:
    return f"{colour}{BOLD}[{label}]{RESET}"


# ── Live terminal process registry ────────────────────────────────────────────
_terminals: dict[str, asyncio.subprocess.Process] = {}
_terminal_output: dict[str, bytearray] = {}


# ── Client implementation ─────────────────────────────────────────────────────


class TestClient:
    """ACP Client that provides real filesystem and terminal capabilities."""

    def __init__(self, workdir: str) -> None:
        self._workdir = workdir

    # session_update ──────────────────────────────────────────────────────────

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if isinstance(update, AgentMessageChunk):
            text = getattr(update.content, "text", "")
            if text:
                print(f"{_hdr('AGENT')} {text}")
        elif isinstance(update, AgentThoughtChunk):
            text = getattr(update.content, "text", "")
            if text:
                print(f"{_hdr('THINK', DIM)} {DIM}{text}{RESET}")
        elif isinstance(update, ToolCallStart):
            print(
                f"{_hdr('TOOL', YELLOW)} {YELLOW}{update.title}{RESET} "
                f"{DIM}[{update.tool_call_id[:8]}…] status={update.status}{RESET}"
            )
        elif isinstance(update, ToolCallProgress):
            print(
                f"{_hdr('TOOL↩', GREEN)} {DIM}[{update.tool_call_id[:8]}…] "
                f"status={update.status}{RESET}"
            )
        elif isinstance(update, AgentPlanUpdate):
            print(f"\n{_hdr('PLAN', MAGENTA)}")
            for entry in update.entries:
                icon = {"pending": "○", "in_progress": "◉", "completed": "✓"}.get(
                    entry.status, "?"
                )
                print(f"  {MAGENTA}{icon}{RESET} [{entry.priority}] {entry.content}")
        elif isinstance(update, AvailableCommandsUpdate):
            names = [f"/{c.name}" for c in update.available_commands]
            print(f"{_hdr('CMDS', BLUE)} Slash commands: {', '.join(names)}")
        else:
            print(f"{_hdr('UPD', DIM)} {type(update).__name__}")

    # filesystem ──────────────────────────────────────────────────────────────

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        print(f"{_hdr('FS↑', CYAN)} read {path}")
        try:
            all_lines = Path(path).read_text().splitlines()
            if line is not None:
                all_lines = all_lines[line - 1 :]
            if limit is not None:
                all_lines = all_lines[:limit]
            content = "\n".join(all_lines)
        except FileNotFoundError:
            content = f"<file not found: {path}>"
        print(
            f"{_hdr('FS↓', CYAN)} {DIM}{content[:120]}{'…' if len(content) > 120 else ''}{RESET}"
        )
        return ReadTextFileResponse(content=content)

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        print(f"{_hdr('FS✎', CYAN)} write {path} ({len(content)} chars)")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)
        return WriteTextFileResponse()

    # terminal ────────────────────────────────────────────────────────────────

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[Any] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        argv = [command, *(args or [])]
        env_dict = dict(os.environ)
        for ev in env or []:
            env_dict[ev.name] = ev.value
        print(f"{_hdr('TERM▶', YELLOW)} {' '.join(argv)}  cwd={cwd or self._workdir}")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd or self._workdir,
            env=env_dict,
        )
        terminal_id = str(uuid.uuid4())
        _terminals[terminal_id] = proc
        _terminal_output[terminal_id] = bytearray()

        limit = output_byte_limit

        async def _drain() -> None:
            assert proc.stdout
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                _terminal_output[terminal_id].extend(chunk)
                if limit and len(_terminal_output[terminal_id]) > limit:
                    _terminal_output[terminal_id] = bytearray(
                        _terminal_output[terminal_id][-limit:]
                    )

        asyncio.create_task(_drain())
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        output = _terminal_output.get(terminal_id, bytearray()).decode(errors="replace")
        return TerminalOutputResponse(output=output, truncated=False)

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        proc = _terminals.get(terminal_id)
        if proc:
            await proc.wait()
            rc = proc.returncode
            print(f"{_hdr('TERM■', YELLOW)} exit_code={rc}")
            return WaitForTerminalExitResponse(
                exit_code=rc,
                signal=None,
                exit_status=TerminalExitStatus(exit_code=rc, signal=None),
            )
        return WaitForTerminalExitResponse(exit_code=0, signal=None)

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalResponse | None:
        proc = _terminals.get(terminal_id)
        if proc and proc.returncode is None:
            proc.kill()
            print(f"{_hdr('TERM✕', RED)} killed {terminal_id[:8]}…")
        return KillTerminalResponse()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        _terminals.pop(terminal_id, None)
        _terminal_output.pop(terminal_id, None)
        return ReleaseTerminalResponse()

    # permissions ─────────────────────────────────────────────────────────────

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        opt = options[0] if options else None
        print(
            f"{_hdr('PERM?', MAGENTA)} '{getattr(tool_call, 'title', '?')}' "
            f"→ auto-allowing '{opt.name if opt else '?'}'"
        )
        return RequestPermissionResponse(
            outcome=SelectedPermissionOutcome(
                outcome="selected",
                option_id=opt.option_id if opt else "",
            )
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass


# ── Prompts to run ────────────────────────────────────────────────────────────

PROMPTS = [
    "Use acp_send_plan to show a 2-step plan: first 'Read files' (high/pending), then 'Write summary' (medium/pending)",
    "Read the file examples/acp_agent.py — just tell me the first 5 lines",
    "Run the shell command: echo 'Hello from the IDE terminal!'",
    "Write 'ACP works!' to /tmp/acp_test.txt, then read it back to confirm",
]


async def main() -> None:
    script = Path(__file__).parent / "acp_agent.py"
    workdir = str(Path(__file__).parent.parent)
    client = TestClient(workdir=workdir)

    caps = ClientCapabilities(
        fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
        terminal=True,
    )

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}ACP Test Client  •  spawning {script.name}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}\n")

    async with spawn_agent_process(
        client,
        "uv",
        "run",
        "python",
        str(script),
        cwd=workdir,
    ) as (agent, _proc):
        init = await agent.initialize(
            protocol_version=1,
            client_capabilities=caps,
            client_info=Implementation(name="test-client", version="0.1.0"),
        )
        print(
            f"{_hdr('INIT', GREEN)} '{init.agent_info.name}'  "
            f"protocol v{init.protocol_version}"
        )
        mcp = init.agent_capabilities.mcp_capabilities
        if mcp:
            print(f"{_hdr('CAPS', GREEN)} MCP http={mcp.http} sse={mcp.sse}")

        session = await agent.new_session(cwd=workdir)
        print(f"{_hdr('SESS', GREEN)} {session.session_id[:8]}…\n")

        for i, prompt_text in enumerate(PROMPTS, 1):
            print(f"\n{BOLD}{'─' * 60}{RESET}")
            print(f"{BOLD}Prompt {i}/{len(PROMPTS)}:{RESET} {prompt_text}")
            print(f"{BOLD}{'─' * 60}{RESET}")

            resp = await agent.prompt(
                prompt=[{"type": "text", "text": prompt_text}],
                session_id=session.session_id,
            )
            print(f"\n{DIM}stop_reason={resp.stop_reason}{RESET}")

        await agent.close_session(session_id=session.session_id)
        print(f"\n{_hdr('DONE', GREEN)} Session closed.\n")


if __name__ == "__main__":
    asyncio.run(main())
