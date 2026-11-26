from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import (
    AgentMessageChunk,
    AvailableCommandsUpdate,
    PromptResponse,
    TextContentBlock,
)
from loguru import logger

from ant_ai.acp.server.commands import ACPCommandContext, parse_slash_command

if TYPE_CHECKING:
    from acp.interfaces import Client
    from acp.schema import AvailableCommand, ClientCapabilities

    from ant_ai.acp.server.commands import ACPCommand
    from ant_ai.acp.server.sessions import ACPSessions


class ACPCommandDispatcher:
    """Advertises an adapter's slash commands and runs the ``kind="code"`` ones."""

    def __init__(self, commands: list[ACPCommand] | None) -> None:
        self.commands: dict[str, ACPCommand] = {c.name: c for c in (commands or [])}
        self.available: list[AvailableCommand] = [
            c.to_available_command() for c in (commands or [])
        ]
        self.sent: set[str] = set()

    def match(self, text: str) -> tuple[ACPCommand | None, str]:
        """The command the prompt `text` invokes, if any, and its arguments."""
        name, args = parse_slash_command(text)
        return (self.commands.get(name) if name else None), args

    async def advertise(self, client: Client | None, session_id: str) -> None:
        """Send the available commands to the client, once per session."""
        if not self.available or not client or session_id in self.sent:
            return
        await client.session_update(
            session_id=session_id,
            update=AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=self.available,
            ),
        )
        self.sent.add(session_id)

    async def run(
        self,
        command: ACPCommand,
        args: str,
        session_id: str,
        sessions: ACPSessions,
        client: Client | None,
        capabilities: ClientCapabilities | None,
    ) -> PromptResponse:
        """Dispatch a ``kind="code"`` command -- no model turn."""
        assert command.handler is not None  # guaranteed by ACPCommand.__post_init__
        ctx = ACPCommandContext(
            args=args,
            session_id=session_id,
            client=client,
            capabilities=capabilities,
            cwd=sessions.cwds.get(session_id),
            history=sessions.histories.setdefault(session_id, []),
            agent=sessions.agent(session_id),
            replace_agent=lambda a: sessions.set_agent(session_id, a),
        )
        try:
            reply = await command.handler(ctx)
        except Exception:
            # Detail stays in the server log; the client gets a generic message so
            # handler internals (paths, config) are not disclosed over the wire.
            logger.exception("acp: /{} failed", command.name)
            reply = f"/{command.name} failed - see server logs."

        if reply and client:
            await client.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=reply),
                ),
            )
        return PromptResponse(stop_reason="end_turn")
