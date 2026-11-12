from __future__ import annotations

import uuid
from typing import Any, Literal

from acp import RequestError
from acp.interfaces import Agent as ACPAgent, Client
from acp.meta import AGENT_METHODS
from acp.schema import (
    AcpMcpServer,
    AgentCapabilities,
    AgentMessageChunk,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CloseNesResponse,
    CloseSessionResponse,
    DeleteSessionResponse,
    DisableProviderResponse,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    HttpMcpServer,
    Implementation,
    InitializeResponse,
    ListProvidersResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    LogoutResponse,
    McpCapabilities,
    McpServerStdio,
    NesRepository,
    NesSuggestContext,
    NewSessionResponse,
    Position,
    PromptResponse,
    Range,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionDeleteCapabilities,
    SessionInfo,
    SetProviderResponse,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    StartNesResponse,
    SuggestNesResponse,
    TextContentBlock,
    TextDocumentContentChangeEvent,
    TextResourceContents,
    WorkspaceFolder,
)
from loguru import logger

from ant_ai.acp.commands import ACPCommand, ACPCommandContext, parse_slash_command
from ant_ai.acp.tools import (
    _acp_capabilities,
    _acp_client,
    _acp_cwd,
    _acp_session_id,
)
from ant_ai.acp.translator import HVEventToACP
from ant_ai.agent.agent import Agent
from ant_ai.core.events import FinalAnswerEvent
from ant_ai.core.message import BaseMessage, Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.tools.tool import Tool, mcp_tools_from_url
from ant_ai.workflow.workflow import Workflow

_McpServers = list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None


def _extract_prompt_text(prompt: list[Any]) -> str:
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


class ACPAdapter(ACPAgent):
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
        self._context_class: type[InvocationContext] = context_class
        self._commands: dict[str, ACPCommand] = {c.name: c for c in (commands or [])}
        self._available_commands: list[AvailableCommand] = [
            c.to_available_command() for c in (commands or [])
        ]
        self._client: Client | None = None
        self._client_capabilities: ClientCapabilities | None = None
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._session_agents: dict[str, Agent] = {}
        self._session_cwds: dict[str, str] = {}
        self._session_commands_sent: set[str] = set()
        self._translator = HVEventToACP()

    def on_connect(self, conn: Client) -> None:
        self._client: Client = conn

    def _set_session_agent(self, session_id: str, agent: Agent) -> None:
        self._session_agents[session_id] = agent

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_capabilities: ClientCapabilities | None = client_capabilities
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name=self._agent.name, version="1.0.0"),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    delete=SessionDeleteCapabilities()
                ),
                mcp_capabilities=McpCapabilities(http=True, sse=True),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServers = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        self._session_cwds[session_id] = cwd
        logger.debug("acp: new_session cwd='{}'", cwd)

        session_agent: Agent = self._agent
        if mcp_servers:
            extra_tools: list[Tool] = []
            for srv in mcp_servers:
                if isinstance(srv, HttpMcpServer):
                    extra_tools.extend(await mcp_tools_from_url(srv.url))
                elif isinstance(srv, SseMcpServer):
                    headers: dict[str, str] = {
                        h.name: h.value for h in (srv.headers or [])
                    }
                    extra_tools.extend(
                        await mcp_tools_from_url(
                            srv.url, headers=headers, transport="sse"
                        )
                    )
                # McpServerStdio: subprocess management out of scope
            if extra_tools:
                # Not model_copy: it skips validators, so the copy would keep
                # the base agent's registry and loop.
                session_agent: Agent = self._agent.model_validate(
                    {**dict(self._agent), "tools": [*self._agent.tools, *extra_tools]}
                )
                # Tools added to the base agent with `add_tool` (e.g. by a
                # colony) are only in its registry, not in `tools`.
                known = {t.name for t in session_agent.registry.tools}
                for tool in self._agent.registry.tools:
                    if tool.name not in known:
                        session_agent.add_tool(tool)
        self._session_agents[session_id] = session_agent

        return NewSessionResponse(session_id=session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: _McpServers = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        if session_id not in self._sessions:
            return None
        return LoadSessionResponse()

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        sessions: list[SessionInfo] = [
            SessionInfo(session_id=sid, cwd="") for sid in self._sessions
        ]
        return ListSessionsResponse(sessions=sessions)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        _acp_client.set(self._client)
        _acp_session_id.set(session_id)
        _acp_capabilities.set(self._client_capabilities)
        cwd: str | None = self._session_cwds.get(session_id)
        _acp_cwd.set(cwd)
        caps: ClientCapabilities | None = self._client_capabilities
        logger.debug(
            "acp: prompt session={} cwd='{}' fs_read={} fs_write={} terminal={}",
            session_id[:8],
            cwd,
            bool(caps and caps.fs and caps.fs.read_text_file),
            bool(caps and caps.fs and caps.fs.write_text_file),
            bool(caps and caps.terminal),
        )

        text: str = _extract_prompt_text(prompt)

        if (
            self._available_commands
            and self._client
            and session_id not in self._session_commands_sent
        ):
            await self._client.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._available_commands,
                ),
            )
            self._session_commands_sent.add(session_id)

        name, args = parse_slash_command(text)
        command = self._commands.get(name) if name else None
        if command is not None and command.kind == "code":
            return await self._run_code_command(command, args, session_id)
        if command is not None and command.kind == "prompt":
            text = command.expand(args)

        history: list[BaseMessage] = list(self._sessions.get(session_id, []))
        history.append(Message(role="user", content=text))

        ctx: InvocationContext = self._context_class.from_metadata(
            session_id=session_id
        )
        agent: Agent = self._session_agents.get(session_id, self._agent)
        state: State = self._workflow.create_state(messages=history)

        final_content = ""
        async for event in self._workflow.stream(agent=agent, ctx=ctx, state=state):
            if self._client:
                await self._translator.apply(event, self._client, session_id)
            if isinstance(event, FinalAnswerEvent):
                final_content = event.content

        if session_id in self._sessions:
            self._sessions[session_id].append(Message(role="user", content=text))
            if final_content:
                self._sessions[session_id].append(
                    Message(role="assistant", content=final_content)
                )

        return PromptResponse(stop_reason="end_turn")

    async def _run_code_command(
        self, command: ACPCommand, args: str, session_id: str
    ) -> PromptResponse:
        """Dispatch a ``kind="code"`` command -- no model turn."""
        assert command.handler is not None  # guaranteed by ACPCommand.__post_init__
        ctx = ACPCommandContext(
            args=args,
            session_id=session_id,
            client=self._client,
            capabilities=self._client_capabilities,
            cwd=self._session_cwds.get(session_id),
            history=self._sessions.setdefault(session_id, []),
            agent=self._session_agents.get(session_id, self._agent),
            replace_agent=lambda a: self._set_session_agent(session_id, a),
        )
        try:
            reply = await command.handler(ctx)
        except Exception:
            # Detail stays in the server log; the client gets a generic message so
            # handler internals (paths, config) are not disclosed over the wire.
            logger.exception("acp: /{} failed", command.name)
            reply = f"/{command.name} failed - see server logs."

        if reply and self._client:
            await self._client.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=reply),
                ),
            )
        return PromptResponse(stop_reason="end_turn")

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServers = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        new_id = str(uuid.uuid4())
        self._sessions[new_id] = list(self._sessions.get(session_id, []))
        self._session_agents[new_id] = self._session_agents.get(session_id, self._agent)
        return ForkSessionResponse(session_id=new_id)

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServers = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        return ResumeSessionResponse()

    def _drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._session_agents.pop(session_id, None)
        self._session_cwds.pop(session_id, None)
        self._session_commands_sent.discard(session_id)

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse | None:
        self._drop_session(session_id)
        return None

    async def delete_session(
        self, session_id: str, **kwargs: Any
    ) -> DeleteSessionResponse:
        # Sessions live only in memory, so deleting one is the same as closing it.
        self._drop_session(session_id)
        return DeleteSessionResponse()

    async def authenticate(
        self, method_id: str, **kwargs: Any
    ) -> AuthenticateResponse | None:
        return None

    async def logout(self, **kwargs: Any) -> LogoutResponse:
        return LogoutResponse()

    # The adapter advertises none of the capabilities below (providers, MCP over
    # ACP, next-edit suggestions, document sync). Requests are refused so a
    # client never mistakes a missing feature for an empty success; notifications
    # need no reply and are dropped.

    async def list_providers(self, **kwargs: Any) -> ListProvidersResponse:
        raise RequestError.method_not_found(AGENT_METHODS["providers_list"])

    async def set_provider(
        self,
        provider_id: str,
        api_type: str,
        base_url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SetProviderResponse:
        raise RequestError.method_not_found(AGENT_METHODS["providers_set"])

    async def disable_provider(
        self, provider_id: str, **kwargs: Any
    ) -> DisableProviderResponse:
        raise RequestError.method_not_found(AGENT_METHODS["providers_disable"])

    async def mcp_message(
        self,
        connection_id: str,
        method: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise RequestError.method_not_found(AGENT_METHODS["mcp_message"])

    async def notify_mcp(
        self,
        connection_id: str,
        method: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def start_nes(
        self,
        workspace_uri: str | None = None,
        workspace_folders: list[WorkspaceFolder] | None = None,
        repository: NesRepository | None = None,
        **kwargs: Any,
    ) -> StartNesResponse:
        raise RequestError.method_not_found(AGENT_METHODS["nes_start"])

    async def suggest_nes(
        self,
        session_id: str,
        uri: str,
        version: int,
        position: Position,
        trigger_kind: Literal["automatic", "diagnostic", "manual"],
        selection: Range | None = None,
        context: NesSuggestContext | None = None,
        **kwargs: Any,
    ) -> SuggestNesResponse:
        raise RequestError.method_not_found(AGENT_METHODS["nes_suggest"])

    async def close_nes(self, session_id: str, **kwargs: Any) -> CloseNesResponse:
        raise RequestError.method_not_found(AGENT_METHODS["nes_close"])

    async def accept_nes(self, session_id: str, id: str, **kwargs: Any) -> None:  # noqa: A002
        pass

    async def reject_nes(
        self,
        session_id: str,
        id: str,  # noqa: A002
        reason: Literal["rejected", "ignored", "replaced", "cancelled"] | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    async def did_open(
        self,
        session_id: str,
        uri: str,
        language_id: str,
        version: int,
        text: str,
        **kwargs: Any,
    ) -> None:
        pass

    async def did_change(
        self,
        session_id: str,
        uri: str,
        version: int,
        content_changes: list[TextDocumentContentChangeEvent],
        **kwargs: Any,
    ) -> None:
        pass

    async def did_close(self, session_id: str, uri: str, **kwargs: Any) -> None:
        pass

    async def did_save(self, session_id: str, uri: str, **kwargs: Any) -> None:
        pass

    async def did_focus(
        self,
        session_id: str,
        uri: str,
        version: int,
        position: Position,
        visible_range: Range,
        **kwargs: Any,
    ) -> None:
        pass

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        return None

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        pass

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass
