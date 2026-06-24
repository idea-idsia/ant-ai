from __future__ import annotations

import uuid
from typing import Any

from acp.interfaces import Agent as ACPAgent, Client
from acp.schema import (
    AcpMcpServer,
    AgentCapabilities,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CloseSessionResponse,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    HttpMcpServer,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpCapabilities,
    McpServerStdio,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionInfo,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
)
from loguru import logger

from ant_ai.acp.tools import (
    _acp_capabilities,
    _acp_client,
    _acp_cwd,
    _acp_session_id,
)
from ant_ai.acp.translator import HVEventToACP
from ant_ai.agent.agent import Agent
from ant_ai.core.events import FinalAnswerEvent
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.tools.tool import mcp_tools_from_url
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
        slash_commands: list[AvailableCommand] | None = None,
    ) -> None:
        self._agent: Agent = agent
        self._workflow: Workflow[State] = workflow
        self._slash_commands = slash_commands or []
        self._client: Client | None = None
        self._client_capabilities: ClientCapabilities | None = None
        self._sessions: dict[str, list[Message]] = {}
        self._session_agents: dict[str, Agent] = {}
        self._session_cwds: dict[str, str] = {}
        self._session_commands_sent: set[str] = set()
        self._translator = HVEventToACP()

    def on_connect(self, conn: Client) -> None:
        self._client: Client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_capabilities = client_capabilities
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name=self._agent.name, version="1.0.0"),
            agent_capabilities=AgentCapabilities(
                load_session=True,
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

        session_agent = self._agent
        if mcp_servers:
            extra_tools = []
            for srv in mcp_servers:
                if isinstance(srv, HttpMcpServer):
                    extra_tools.extend(await mcp_tools_from_url(srv.url))
                elif isinstance(srv, SseMcpServer):
                    headers = {h.name: h.value for h in (srv.headers or [])}
                    extra_tools.extend(
                        await mcp_tools_from_url(
                            srv.url, headers=headers, transport="sse"
                        )
                    )
                # McpServerStdio: subprocess management out of scope
            if extra_tools:
                session_agent = self._agent.model_copy(
                    update={"tools": [*self._agent.tools, *extra_tools]}
                )
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
        sessions = [SessionInfo(session_id=sid, cwd="") for sid in self._sessions]
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
        cwd = self._session_cwds.get(session_id)
        _acp_cwd.set(cwd)
        caps = self._client_capabilities
        logger.debug(
            "acp: prompt session={} cwd='{}' fs_read={} fs_write={} terminal={}",
            session_id[:8],
            cwd,
            bool(caps and caps.fs and caps.fs.read_text_file),
            bool(caps and caps.fs and caps.fs.write_text_file),
            bool(caps and caps.terminal),
        )

        text = _extract_prompt_text(prompt)

        if (
            self._slash_commands
            and self._client
            and session_id not in self._session_commands_sent
        ):
            await self._client.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._slash_commands,
                ),
            )
            self._session_commands_sent.add(session_id)

        history: list[Message] = list(self._sessions.get(session_id, []))
        history.append(Message(role="user", content=text))

        ctx = InvocationContext(session_id=session_id)
        agent = self._session_agents.get(session_id, self._agent)
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

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse | None:
        self._sessions.pop(session_id, None)
        self._session_agents.pop(session_id, None)
        self._session_cwds.pop(session_id, None)
        self._session_commands_sent.discard(session_id)
        return None

    async def authenticate(
        self, method_id: str, **kwargs: Any
    ) -> AuthenticateResponse | None:
        return None

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
