from __future__ import annotations

import uuid
from typing import Any

from acp.interfaces import Client
from acp.schema import (
    AcpMcpServer,
    AgentCapabilities,
    AuthenticateResponse,
    ClientCapabilities,
    CloseSessionResponse,
    ForkSessionResponse,
    HttpMcpServer,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionInfo,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    TextContentBlock,
)

from ant_ai.acp.translator import HVEventToACP
from ant_ai.agent.agent import Agent
from ant_ai.core.events import FinalAnswerEvent
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext, State
from ant_ai.workflow.workflow import Workflow

_McpServers = list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None


class ACPAdapter:
    """Adapts an ant-ai Agent + Workflow to the ACP Agent stdio protocol.

    Can be used directly with :func:`acp.run_agent` for stdio mode, or
    wrapped in an :class:`ACPServer` for WebSocket/ASGI mode.
    """

    def __init__(self, agent: Agent, workflow: Workflow) -> None:
        self._agent: Agent = agent
        self._workflow: Workflow[State] = workflow
        self._client: Client | None = None
        self._sessions: dict[str, list[Message]] = {}
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
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name=self._agent.name, version="1.0.0"),
            agent_capabilities=AgentCapabilities(load_session=True),
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
        text = " ".join(
            block.text for block in prompt if isinstance(block, TextContentBlock)
        )

        history: list[Message] = list(self._sessions.get(session_id, []))
        history.append(Message(role="user", content=text))

        ctx = InvocationContext(session_id=session_id)
        state: State = self._workflow.create_state(messages=history)

        final_content = ""
        async for event in self._workflow.stream(
            agent=self._agent, ctx=ctx, state=state
        ):
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
