from typing import Any

import pytest
from a2a.types import AgentCard, AgentInterface

from ant_ai.a2a.client import A2AClient
from ant_ai.a2a.client.config import A2AConfig
from ant_ai.a2a.client.tool import A2AAgentTool


@pytest.fixture
async def a2a_tool(codegen_agent_server: str) -> A2AAgentTool:
    """
    Fully initialize an A2AAgentTool against the local test codegen agent.
    """
    endpoint: str = codegen_agent_server
    timeout = 100

    config = A2AConfig(endpoint=endpoint, timeout=timeout)
    tool: A2AAgentTool = await A2AAgentTool.from_config(config)

    a2a: A2AClient | None = tool._a2a
    assert a2a is not None

    card: AgentCard = await a2a.get_agent_card()

    # Patch the first supported interface's URL to point at the local test server.
    patched_card = AgentCard()
    patched_card.CopyFrom(card)
    if patched_card.supported_interfaces:
        patched_card.supported_interfaces[0].url = endpoint
    else:
        patched_card.supported_interfaces.append(
            AgentInterface(protocol_binding="JSONRPC", url=endpoint)
        )

    # Reconnect with the patched card instead of the one fetched from the server.
    await a2a.aclose()
    a2a._agent_card = patched_card

    return tool


@pytest.mark.integration
@pytest.mark.a2a
async def test_from_config_initializes_tool(a2a_tool: A2AAgentTool) -> None:
    assert a2a_tool._agent_card is not None
    assert a2a_tool._a2a is not None

    assert a2a_tool.name == "codegen"
    assert "A 10x Software Developer" in a2a_tool.description
    assert "A 10x Software Developer" in a2a_tool._agent_card.description

    assert a2a_tool.parameters["type"] == "object"
    assert "message" in a2a_tool.parameters["properties"]
    assert "message" in a2a_tool.parameters["required"]

    assert a2a_tool.is_namespace is False


@pytest.mark.integration
@pytest.mark.a2a
async def test_sending_message_roundtrip(a2a_tool: A2AAgentTool) -> None:
    request_text = "ping"
    response: Any = await a2a_tool(request_text)

    assert isinstance(response, str)
    assert response != ""
    assert not response.startswith("Error sending message"), response
    assert not response.startswith("No response from"), response
