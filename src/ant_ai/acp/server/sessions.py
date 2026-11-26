from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from acp.schema import (
    AcpMcpServer,
    DeleteSessionResponse,
    ForkSessionResponse,
    HttpMcpServer,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PermissionOptionKind,
    SessionInfo,
    SseMcpServer,
)
from loguru import logger

from ant_ai.tools.tool import mcp_tools_from_url

if TYPE_CHECKING:
    from ant_ai.agent.agent import Agent
    from ant_ai.core.message import BaseMessage
    from ant_ai.tools.tool import Tool

McpServer = HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio
McpServers = list[McpServer] | None


async def _server_tools(srv: McpServer) -> list[Tool]:
    if isinstance(srv, HttpMcpServer):
        return await mcp_tools_from_url(srv.url)
    if isinstance(srv, SseMcpServer):
        headers: dict[str, str] = {h.name: h.value for h in (srv.headers or [])}
        return await mcp_tools_from_url(srv.url, headers=headers, transport="sse")
    # McpServerStdio: subprocess management out of scope
    return []


async def with_mcp_tools(agent: Agent, mcp_servers: McpServers) -> Agent:
    """Return `agent` extended with the tools of the session's MCP servers."""
    extra_tools: list[Tool] = []
    for srv in mcp_servers or []:
        extra_tools.extend(await _server_tools(srv))
    if not extra_tools:
        return agent
    # Not model_copy: it skips validators, so the copy would keep
    # the base agent's registry and loop.
    session_agent: Agent = agent.model_validate(
        {**dict(agent), "tools": [*agent.tools, *extra_tools]}
    )
    # Tools added to the base agent with `add_tool` (e.g. by a
    # colony) are only in its registry, not in `tools`.
    known = {t.name for t in session_agent.registry.tools}
    for tool in agent.registry.tools:
        if tool.name not in known:
            session_agent.add_tool(tool)
    return session_agent


class ACPSessions:
    """The in-memory sessions of an ACP adapter: history, agent, cwd, permissions."""

    def __init__(self, agent: Agent) -> None:
        self.base_agent: Agent = agent
        self.histories: dict[str, list[BaseMessage]] = {}
        self.agents: dict[str, Agent] = {}
        self.cwds: dict[str, str] = {}
        self.permissions: dict[str, dict[str, PermissionOptionKind]] = {}

    def agent(self, session_id: str) -> Agent:
        return self.agents.get(session_id, self.base_agent)

    def set_agent(self, session_id: str, agent: Agent) -> None:
        self.agents[session_id] = agent

    async def new(self, cwd: str, mcp_servers: McpServers) -> NewSessionResponse:
        session_id = str(uuid.uuid4())
        self.histories[session_id] = []
        self.cwds[session_id] = cwd
        logger.debug("acp: new_session cwd='{}'", cwd)
        self.agents[session_id] = await with_mcp_tools(self.base_agent, mcp_servers)
        return NewSessionResponse(session_id=session_id)

    def load(self, session_id: str) -> LoadSessionResponse | None:
        if session_id not in self.histories:
            return None
        return LoadSessionResponse()

    def list_all(self) -> ListSessionsResponse:
        sessions: list[SessionInfo] = [
            SessionInfo(session_id=sid, cwd="") for sid in self.histories
        ]
        return ListSessionsResponse(sessions=sessions)

    def fork(self, session_id: str) -> ForkSessionResponse:
        new_id = str(uuid.uuid4())
        self.histories[new_id] = list(self.histories.get(session_id, []))
        self.agents[new_id] = self.agent(session_id)
        self.permissions[new_id] = dict(self.permissions.get(session_id, {}))
        return ForkSessionResponse(session_id=new_id)

    def drop(self, session_id: str) -> None:
        self.histories.pop(session_id, None)
        self.agents.pop(session_id, None)
        self.cwds.pop(session_id, None)
        self.permissions.pop(session_id, None)

    def delete(self, session_id: str) -> DeleteSessionResponse:
        # Sessions live only in memory, so deleting one is the same as closing it.
        self.drop(session_id)
        return DeleteSessionResponse()
