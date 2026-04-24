from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

import ant_ai.a2a.colony as co


@dataclass
class PatchedAgentSpec:
    agent: object
    workflow: object
    url: str
    host: str
    port: int
    card: object


class FakeAgent:
    def __init__(self) -> None:
        self.tools: list[object] = []
        self.add_tool = MagicMock(side_effect=self.tools.append)


class FakeWorkflow:
    pass


class _FakeInterface:
    def __init__(self, url: str) -> None:
        self.url = url


class FakeAgentCard:
    def __init__(self, url: str) -> None:
        self.supported_interfaces = [_FakeInterface(url)]


class FakeConfig:
    def __init__(self, endpoint: str) -> None:
        self.endpoint: str = endpoint


class FakeTool:
    """Replacement for A2AAgentTool used in isinstance checks inside Colony."""

    def __init__(self, config: FakeConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: FakeConfig, agent_card: FakeAgentCard) -> FakeTool:
        return cls(config=config)


class FakeServer:
    def __init__(self, *, agent, workflow, host, port, agent_card, task_store) -> None:
        self.agent = agent
        self.workflow = workflow
        self.host = host
        self.port = port
        self.agent_card = agent_card
        self.task_store = task_store
        self._fastapi_app = object()
        self._starlette_app = object()

    def fastapi_app(self):
        return self._fastapi_app

    def starlette_app(self):
        return self._starlette_app


@pytest.fixture
def colony(monkeypatch):
    monkeypatch.setattr(co, "AgentSpec", PatchedAgentSpec, raising=True)
    monkeypatch.setattr(co, "A2AAgentTool", FakeTool, raising=True)
    monkeypatch.setattr(co, "A2AServer", FakeServer, raising=True)

    monkeypatch.setattr(
        co.Colony, "get_task_store", lambda self, table: object(), raising=True
    )

    return co.Colony()


def register(h, name: str, *, url: str):
    agent = FakeAgent()
    workflow = FakeWorkflow()
    card = FakeAgentCard(url=url)
    out = h.agent(name, agent=agent, workflow=workflow, card=card)
    return out, agent, workflow, card


def test_agent_registers_and_returns_self_and_parses_host_port(colony):
    out, a, w, c = register(colony, "alpha", url="http://127.0.0.1:1111/")

    assert out is colony
    assert "alpha" in colony._specs

    spec = colony._specs["alpha"]
    assert spec.agent is a
    assert spec.workflow is w
    assert spec.card is c
    assert spec.host == "127.0.0.1"
    assert spec.port == 1111


def test_agent_duplicate_name_raises_value_error(colony):
    register(colony, "alpha", url="http://127.0.0.1:1111/")
    with pytest.raises(ValueError, match=r"Agent 'alpha' already registered\."):
        register(colony, "alpha", url="http://127.0.0.1:1111/")


def test_collab_unknown_source_raises_key_error(colony):
    register(colony, "beta", url="http://127.0.0.1:2222/")
    with pytest.raises(KeyError, match=r"Unknown agent 'alpha' in collaboration\."):
        colony.collab("alpha", "beta", config=FakeConfig("http://x"))


def test_collab_unknown_target_raises_key_error_when_config_provided(colony):
    register(colony, "alpha", url="http://127.0.0.1:1111/")
    with pytest.raises(KeyError, match=r"Unknown agent 'beta' in collaboration\."):
        colony.collab("alpha", "beta", config=FakeConfig("http://x"))


def test_collab_unknown_target_without_config_currently_raises_raw_keyerror(colony):
    """
    Current implementation touches self._specs[target] to build default config
    before _add_edge() validates membership, so the KeyError is the missing dict key.
    """
    register(colony, "alpha", url="http://127.0.0.1:1111/")
    with pytest.raises(KeyError, match=r"'beta'"):
        colony.collab("alpha", "beta")  # no config


def test_collab_adds_edge_and_last_config_wins(colony):
    register(colony, "alpha", url="http://alpha:1111/")
    register(colony, "beta", url="http://beta:2222/")

    cfg1 = FakeConfig("http://beta:2222/")
    cfg2 = FakeConfig("http://beta:2222/")
    colony.collab("alpha", "beta", config=cfg1)
    colony.collab("alpha", "beta", config=cfg2)

    assert colony._edges["alpha"]["beta"] is cfg2


def test_collab_mutual_adds_both_directions(colony):
    register(colony, "alpha", url="http://alpha:1111/")
    register(colony, "beta", url="http://beta:2222/")

    cfg = FakeConfig("http://peer/")
    colony.collab("alpha", "beta", config=cfg, mutual=True)

    assert colony._edges["alpha"]["beta"] is cfg
    assert colony._edges["beta"]["alpha"] is cfg


def test_asgi_unknown_agent_raises_key_error(colony):
    with pytest.raises(
        KeyError, match=r"Agent 'missing' is not registered in Colony\."
    ):
        colony.asgi(agent_name="missing", use_fastapi=True)


def test_asgi_builds_server_wires_tools_and_returns_fastapi_app(colony):
    _, alpha_agent, _, _ = register(colony, "alpha", url="http://alpha:1111/")
    register(colony, "beta", url="http://beta:2222/")

    cfg = FakeConfig(endpoint="http://beta:2222/")
    colony.collab("alpha", "beta", config=cfg)

    app = colony.asgi(agent_name="alpha", use_fastapi=True)

    assert app is not None
    assert len(alpha_agent.tools) == 1
    assert isinstance(alpha_agent.tools[0], FakeTool)
    assert alpha_agent.tools[0].config is cfg
    alpha_agent.add_tool.assert_called_once()


def test_asgi_returns_starlette_app_when_use_fastapi_false(colony):
    register(colony, "alpha", url="http://alpha:1111/")
    app = colony.asgi(agent_name="alpha", use_fastapi=False)
    assert app is not None


def test_wire_a2a_tools_no_outgoing_edges_noop(colony):
    _, alpha_agent, _, _ = register(colony, "alpha", url="http://alpha:1111/")
    colony._wire_a2a_tools("alpha", agent=alpha_agent)
    assert alpha_agent.tools == []
    alpha_agent.add_tool.assert_not_called()


def test_wire_a2a_tools_skips_if_endpoint_tool_already_present(colony):
    _, alpha_agent, _, _ = register(colony, "alpha", url="http://alpha:1111/")
    register(colony, "beta", url="http://beta:2222/")

    cfg = FakeConfig(endpoint="http://beta:2222/")
    colony.collab("alpha", "beta", config=cfg)

    alpha_agent.tools.append(FakeTool(config=FakeConfig(endpoint="http://beta:2222/")))

    colony._wire_a2a_tools("alpha", agent=alpha_agent)

    assert len(alpha_agent.tools) == 1
    alpha_agent.add_tool.assert_not_called()
