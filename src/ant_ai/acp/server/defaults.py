from __future__ import annotations

from typing import Any, Literal

from acp.exceptions import RequestError
from acp.interfaces import Agent as ACPAgent
from acp.meta import AGENT_METHODS
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    CloseNesResponse,
    DisableProviderResponse,
    Implementation,
    InitializeResponse,
    ListProvidersResponse,
    LogoutResponse,
    McpCapabilities,
    NesRepository,
    NesSuggestContext,
    Position,
    Range,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionDeleteCapabilities,
    SetProviderResponse,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    StartNesResponse,
    SuggestNesResponse,
    TextDocumentContentChangeEvent,
    WorkspaceFolder,
)

from ant_ai.acp.server.sessions import McpServers


def initialize_response(protocol_version: int, agent_name: str) -> InitializeResponse:
    """The reply to ``initialize``: the adapter's identity and capabilities."""
    return InitializeResponse(
        protocol_version=protocol_version,
        agent_info=Implementation(name=agent_name, version="1.0.0"),
        agent_capabilities=AgentCapabilities(
            load_session=True,
            session_capabilities=SessionCapabilities(
                delete=SessionDeleteCapabilities()
            ),
            mcp_capabilities=McpCapabilities(http=True, sse=True),
        ),
    )


class ACPAgentDefaults(ACPAgent):
    """The ACP agent methods that get a fixed reply, whatever the agent."""

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: McpServers = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        return ResumeSessionResponse()

    async def authenticate(
        self, method_id: str, **kwargs: Any
    ) -> AuthenticateResponse | None:
        return None

    async def logout(self, **kwargs: Any) -> LogoutResponse:
        return LogoutResponse()

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        return None

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass

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
