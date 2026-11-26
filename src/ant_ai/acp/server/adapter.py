from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ant_ai.acp.server.defaults import ACPAgentDefaults, initialize_response
from ant_ai.acp.server.dispatch import ACPCommandDispatcher
from ant_ai.acp.server.sessions import ACPSessions, McpServers
from ant_ai.acp.server.turns import ACPTurnRunner, bind_turn, prompt_text
from ant_ai.core.types import InvocationContext

if TYPE_CHECKING:
    from acp.interfaces import Client
    from acp.schema import (
        ClientCapabilities,
        CloseSessionResponse,
        DeleteSessionResponse,
        ForkSessionResponse,
        Implementation,
        InitializeResponse,
        ListSessionsResponse,
        LoadSessionResponse,
        NewSessionResponse,
        PromptResponse,
    )

    from ant_ai.acp.server.commands import ACPCommand
    from ant_ai.agent.agent import Agent
    from ant_ai.core.types import State
    from ant_ai.workflow.workflow import Workflow


class ACPAdapter(ACPAgentDefaults):
    """Adapts an ant-ai Agent + Workflow to the ACP Agent stdio protocol.

    Explicitly implements :class:`acp.interfaces.Agent` so type checkers
    verify the full protocol surface is covered.

    Can be used directly with :func:`acp.run_agent` for stdio mode, or
    wrapped in an :class:`ACPServer` for WebSocket/ASGI mode.
    """

    def __init__(
        self,
        agent: Agent,
        workflow: Workflow,
        *,
        commands: list[ACPCommand] | None = None,
        context_class: type[InvocationContext] = InvocationContext,
    ) -> None:
        self._agent: Agent = agent
        self._workflow: Workflow[State] = workflow
        self._client: Client | None = None
        self._client_capabilities: ClientCapabilities | None = None
        self._sessions = ACPSessions(agent)
        self._commands = ACPCommandDispatcher(commands)
        self._turns = ACPTurnRunner(workflow, context_class)

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_capabilities = client_capabilities
        return initialize_response(protocol_version, self._agent.name)

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: McpServers = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        return await self._sessions.new(cwd, mcp_servers)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: McpServers = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        return self._sessions.load(session_id)

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        return self._sessions.list_all()

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        bind_turn(session_id, self._sessions, self._client, self._client_capabilities)
        text: str = prompt_text(prompt)
        await self._commands.advertise(self._client, session_id)

        command, args = self._commands.match(text)
        if command is not None and command.kind == "code":
            return await self._commands.run(
                command,
                args,
                session_id,
                self._sessions,
                self._client,
                self._client_capabilities,
            )
        if command is not None and command.kind == "prompt":
            text = command.expand(args)
        return await self._turns.run(session_id, text, self._sessions, self._client)

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: McpServers = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        return self._sessions.fork(session_id)

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse | None:
        self._sessions.drop(session_id)
        self._commands.sent.discard(session_id)
        return None

    async def delete_session(
        self, session_id: str, **kwargs: Any
    ) -> DeleteSessionResponse:
        self._commands.sent.discard(session_id)
        return self._sessions.delete(session_id)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._turns.cancel(session_id)
