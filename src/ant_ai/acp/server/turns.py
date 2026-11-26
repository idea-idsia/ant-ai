from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from acp.schema import (
    EmbeddedResourceContentBlock,
    PromptResponse,
    TextContentBlock,
    TextResourceContents,
)
from loguru import logger

from ant_ai.acp.server.translator import HVEventToACP
from ant_ai.acp.tools import (
    _acp_capabilities,
    _acp_client,
    _acp_cwd,
    _acp_permissions,
    _acp_session_id,
)
from ant_ai.core.events import event_to_message
from ant_ai.core.message import Message

if TYPE_CHECKING:
    from acp.interfaces import Client
    from acp.schema import ClientCapabilities

    from ant_ai.acp.server.sessions import ACPSessions
    from ant_ai.core.message import BaseMessage
    from ant_ai.core.types import InvocationContext, State
    from ant_ai.workflow.workflow import Workflow


def prompt_text(prompt: list[Any]) -> str:
    """The text of a prompt's text blocks and embedded text resources."""
    parts = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            if block.text:
                parts.append(block.text)
        elif isinstance(block, EmbeddedResourceContentBlock):
            r = block.resource
            if isinstance(r, TextResourceContents) and r.text:
                parts.append(f"[File: {r.uri}]\n{r.text}")
    return "\n".join(parts)


def bind_turn(
    session_id: str,
    sessions: ACPSessions,
    client: Client | None,
    capabilities: ClientCapabilities | None,
) -> None:
    """Expose the session to the ACP tools for the prompt turn being handled."""
    _acp_client.set(client)
    _acp_session_id.set(session_id)
    _acp_capabilities.set(capabilities)
    _acp_permissions.set(sessions.permissions.setdefault(session_id, {}))
    cwd: str | None = sessions.cwds.get(session_id)
    _acp_cwd.set(cwd)
    logger.debug(
        "acp: prompt session={} cwd='{}' fs_read={} fs_write={} terminal={}",
        session_id[:8],
        cwd,
        bool(capabilities and capabilities.fs and capabilities.fs.read_text_file),
        bool(capabilities and capabilities.fs and capabilities.fs.write_text_file),
        bool(capabilities and capabilities.terminal),
    )


class ACPTurnRunner:
    """Runs a session's model turns through the workflow, one cancellable task each."""

    def __init__(
        self, workflow: Workflow[State], context_class: type[InvocationContext]
    ) -> None:
        self._workflow: Workflow[State] = workflow
        self._context_class: type[InvocationContext] = context_class
        self._running: dict[str, asyncio.Task[None]] = {}
        self._translator = HVEventToACP()

    async def run(
        self,
        session_id: str,
        text: str,
        sessions: ACPSessions,
        client: Client | None,
    ) -> PromptResponse:
        """Run one turn for the user's `text` and record it in the session."""
        history: list[BaseMessage] = list(sessions.histories.get(session_id, []))
        history.append(Message(role="user", content=text))

        ctx: InvocationContext = self._context_class.from_metadata(
            session_id=session_id
        )
        agent = sessions.agent(session_id)
        state: State = self._workflow.create_state(messages=history)

        # The turn is kept as the model saw it (tool calls and their results
        # included), so a turn that stopped on a clarification still holds the
        # question when the user's reply arrives as the next prompt.
        turn: list[BaseMessage] = [Message(role="user", content=text)]

        async def run() -> None:
            async for event in self._workflow.stream(agent=agent, ctx=ctx, state=state):
                if client:
                    await self._translator.apply(event, client, session_id)
                if (message := event_to_message(event)) is not None:
                    turn.append(message)

        # A task of its own, so that `cancel` can stop it without stopping the
        # connection that is waiting for the prompt's response.
        task = asyncio.ensure_future(run())
        self._running[session_id] = task
        try:
            await task
        except asyncio.CancelledError:
            # Only a `cancel` from the client ends the turn normally; a
            # cancellation of the prompt itself goes on up.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return PromptResponse(stop_reason="cancelled")
        finally:
            if self._running.get(session_id) is task:
                del self._running[session_id]

        if session_id in sessions.histories:
            sessions.histories[session_id].extend(turn)

        return PromptResponse(stop_reason="end_turn")

    def cancel(self, session_id: str) -> None:
        task = self._running.get(session_id)
        if task is not None:
            task.cancel()
