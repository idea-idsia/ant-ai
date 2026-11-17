"""The ACP client behind `ant --tui`: the agent runs in another process and
the view shows what arrives over the protocol.

The agent can be `ant --serve` (the same coding agent, served over ACP) or any
other ACP agent started by a command.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractContextManager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

from acp import PROTOCOL_VERSION, Client, RequestError, connect_to_agent
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    Implementation,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
)

from ant_ai.core.events import (
    BaseEvent,
    FinalAnswerEvent,
    MaxStepsReachedEvent,
    ReasoningEvent,
    ToolCallingEvent,
    ToolResultEvent,
)
from ant_ai.core.message import ToolCall, ToolFunction

# Seconds to wait for more tool calls of the same step before printing it.
STEP_WAIT = 0.05

# What the user answers to `allow ...? y/n/a`, as the kind of ACP option it picks.
ANSWERS = {"y": "allow_once", "yes": "allow_once", "a": "allow_always"}


class View(Protocol):
    """What a session draws on: the Textual app, or the terminal `Renderer` in tests."""

    def event(self, event: BaseEvent) -> None: ...

    def note(self, text: str) -> None: ...

    def partial(self, text: str) -> None: ...

    def turn(self) -> AbstractContextManager[None]: ...

    def paused(self) -> AbstractContextManager[None]: ...


def _text(update: AgentMessageChunk | AgentThoughtChunk) -> str:
    return getattr(update.content, "text", "") or ""


def _arguments(raw_input: Any) -> str:
    if raw_input is None:
        return "{}"
    return raw_input if isinstance(raw_input, str) else json.dumps(raw_input)


class ACPEvents:
    """Turns the session updates of one turn back into the events of the agent loop.

    ACP sends the text of the agent in chunks and one update per tool call. The
    text written before a group of tool calls is the reason for that step; the
    text left at the end of the turn is the answer. Tool calls that arrive one
    after the other belong to the same step.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._step: list[ToolCall] = []
        self._reason = ""
        self._names: dict[str, str] = {}

    def update(self, update: Any) -> list[BaseEvent]:
        if isinstance(update, ToolCallStart):
            if not self._step:
                self._reason = "".join(self._text).strip()
                self._text.clear()
            self._names[update.tool_call_id] = update.title
            self._step.append(
                ToolCall(
                    id=update.tool_call_id,
                    function=ToolFunction(
                        name=update.title, arguments=_arguments(update.raw_input)
                    ),
                )
            )
            return []
        events = self.flush()
        match update:
            case AgentMessageChunk():
                self._text.append(_text(update))
            case AgentThoughtChunk() if _text(update):
                events.append(ReasoningEvent(content=_text(update)))
            case ToolCallProgress() if update.status in ("completed", "failed"):
                output = update.raw_output
                events.append(
                    ToolResultEvent(
                        tool_call_id=update.tool_call_id,
                        name=self._names.pop(
                            update.tool_call_id, update.title or "tool"
                        ),
                        content=output
                        if isinstance(output, str)
                        else json.dumps(output or ""),
                        is_error=update.status == "failed",
                    )
                )
        return events

    def finish(self, stop_reason: str | None) -> list[BaseEvent]:
        """The events that close the turn: the answer, or why there is none."""
        events = self.flush()
        answer = "".join(self._text).strip()
        self._text.clear()
        if stop_reason == "max_turn_requests":
            events.append(MaxStepsReachedEvent())
        elif answer:
            events.append(FinalAnswerEvent(content=answer))
        return events

    def flush(self) -> list[BaseEvent]:
        """The step whose tool calls arrived so far, if it is not out yet."""
        if not self._step:
            return []
        event = ToolCallingEvent(content=self._reason, tool_calls=tuple(self._step))
        self._step, self._reason = [], ""
        return [event]


class _Client(Client):
    """The client side of the connection: prints updates, asks for permissions.

    It offers no file system and no terminal: the agent works on its own
    machine and only reports what it does.
    """

    def __init__(self, renderer: View, read_answer: Callable[[str], Awaitable[str]]):
        self.renderer = renderer
        self.events = ACPEvents()
        self._read_answer = read_answer
        # Events that arrive while the user answers a question wait here, so
        # they do not land in the middle of the question.
        self._held: list[BaseEvent] | None = None
        self._pending_step: asyncio.TimerHandle | None = None

    def _show(self, events: list[BaseEvent]) -> None:
        if self._held is not None:
            self._held.extend(events)
            return
        for event in events:
            self.renderer.event(event)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if self._pending_step is not None:
            self._pending_step.cancel()
        if isinstance(update, AgentMessageChunk) and _text(update):
            self.renderer.partial(_text(update))
        self._show(self.events.update(update))
        if isinstance(update, ToolCallStart):
            # The other calls of the step come right after; if none does, the
            # step is printed so a long tool does not leave the screen empty.
            self._pending_step = asyncio.get_running_loop().call_later(
                STEP_WAIT, lambda: self._show(self.events.flush())
            )

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        # A tool can ask before any update closes its step: print the step first.
        self._show(self.events.flush())
        self._held = []
        try:
            with self.renderer.paused():
                answer = await self._read_answer(f"allow {tool_call.title}? y/n/a")
        finally:
            held, self._held = self._held, None
            self._show(held)
        wanted = ANSWERS.get(answer.strip().lower(), "reject_once")
        chosen = next((o for o in options if o.kind == wanted), None)
        if chosen is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=chosen.option_id)
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass

    # File system, terminal, elicitation and MCP: not offered (see the class docstring).

    async def write_text_file(self, *args: Any, **kwargs: Any) -> None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(self, *args: Any, **kwargs: Any) -> None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> None:
        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("elicitation/create")

    async def complete_elicitation(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def connect_mcp(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("mcp/connect")

    async def disconnect_mcp(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("mcp/disconnect")

    async def mcp_message(self, *args: Any, **kwargs: Any) -> Any:
        raise RequestError.method_not_found("mcp/message")

    async def notify_mcp(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_connect(self, conn: Any) -> None:
        pass


class RemoteSession:
    """One conversation with an agent that runs in another process, over ACP."""

    def __init__(
        self,
        command: list[str],
        workspace: Path,
        *,
        label: str,
        renderer: View,
        read_answer: Callable[[str], Awaitable[str]],
    ):
        self.command = command
        self.workspace = workspace.resolve()
        self.label = label
        self.renderer = renderer
        self.client = _Client(renderer, read_answer)
        self.session_id = ""
        self.agent_name = ""
        self._conn: Any = None
        self.stderr: deque[str] = deque(maxlen=20)

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncGenerator[RemoteSession]:
        """Start the agent process, open a session, and stop the process at the end."""
        # A process group of its own: Ctrl-C in the terminal reaches only `ant`,
        # which then asks the agent to cancel the turn instead of killing it.
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None
        conn = connect_to_agent(self.client, process.stdin, process.stdout)
        drain = asyncio.ensure_future(self._drain(process))
        try:
            init = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(name="ant", version=version("ant-ai")),
            )
            self.agent_name = init.agent_info.name if init.agent_info else ""
            self._conn = conn
            await self.clear()
            yield self
        finally:
            await conn.close()
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                process.kill()
                await process.wait()
            drain.cancel()

    async def _drain(self, process: asyncio.subprocess.Process) -> None:
        # The agent logs to stderr; a full pipe would block it. The last lines
        # are kept to explain a crash.
        if process.stderr is None:
            return
        async for line in process.stderr:
            self.stderr.append(line.decode(errors="replace").rstrip())

    async def clear(self) -> None:
        """Start a new conversation."""
        session = await self._conn.new_session(cwd=str(self.workspace))
        self.session_id = session.session_id

    async def ask(self, text: str) -> None:
        """Send one user message and print the run until the answer."""
        with self.renderer.turn():
            request = asyncio.ensure_future(
                self._conn.prompt(
                    session_id=self.session_id,
                    prompt=[TextContentBlock(type="text", text=text)],
                )
            )
            try:
                response = await asyncio.shield(request)
            except asyncio.CancelledError:
                # Ask the agent to stop, then wait for the turn to end so its
                # last updates do not land after the prompt.
                await self._conn.cancel(session_id=self.session_id)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(request, 10)
                self.client.events = ACPEvents()
                self.renderer.note("interrupted")
                return
            for event in self.client.events.finish(response.stop_reason):
                self.renderer.event(event)
