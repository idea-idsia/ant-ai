import sys

import pytest

from ant_ai.a2a.agent import A2AAgentTool
from ant_ai.a2a.config import A2AConfig
from ant_ai.core.events import FinalAnswerEvent, UpdateEvent

MODULE = sys.modules[A2AAgentTool.__module__]


class DummySkill:
    def __init__(self, name: str = "dummy_skill"):
        self.name: str = name
        self.description: str = "The skill is dummy.  It does nothing."
        self.tags: list[str] = ["dummy_skill", "very_dummy_skill"]
        self.examples = None


class DummyAgentCard:
    def __init__(self, name="agent-name", description="base-desc", skills=None):
        self.name = name
        self.description = description
        self.skills = skills or []


@pytest.fixture
def config() -> A2AConfig:
    # int timeout is accepted and normalized to httpx.Timeout by A2AConfig
    return A2AConfig(endpoint="https://example.com", timeout=1)


@pytest.mark.unit
@pytest.mark.a2a
async def test_ensure_initialized_populates_card_a2a_and_metadata(monkeypatch, config):
    agent_card = DummyAgentCard(
        name="remote-agent",
        description="remote-desc",
        skills=[DummySkill("skill_one")],
    )

    class FakeA2AClient:
        def __init__(self, config):
            self.config = config
            self.get_agent_card_called = False

        async def get_agent_card(self):
            self.get_agent_card_called = True
            return agent_card

        async def send_message(self, message: str, context_id: None = None):
            # Not used in this test
            if False:  # pragma: no cover
                yield None

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool = A2AAgentTool(
        name=None,
        description=None,
        parameters=None,
        config=config,
    )

    await tool._ensure_initialized()

    assert tool._initialized is True
    assert tool._agent_card is agent_card
    assert tool._a2a is not None
    assert tool._a2a.get_agent_card_called is True

    # Name and description are taken from the card when not provided
    assert tool.name == "remote-agent"
    assert "remote-desc" in tool.description
    assert "skill_one" in tool.description

    # Parameters schema is set as expected
    assert tool.parameters["type"] == "object"
    assert "message" in tool.parameters["properties"]
    assert tool.parameters["properties"]["message"]["type"] == "string"
    assert "message" in tool.parameters["required"]

    # Underlying callable is attached
    assert hasattr(tool, "_func")
    assert callable(tool._func)


@pytest.mark.unit
@pytest.mark.a2a
async def test_ensure_initialized_does_not_override_existing_name_or_description(
    monkeypatch, config
):
    agent_card = DummyAgentCard(
        name="remote-agent",
        description="remote-desc",
    )

    class FakeA2AClient:
        def __init__(self, config):
            self.config = config

        async def get_agent_card(self):
            return agent_card

        async def send_message(self, message: str, context_id: None = None):
            if False:  # pragma: no cover
                yield None

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool = A2AAgentTool(
        name="local-name",
        description="local-desc",
        parameters=None,
        config=config,
    )

    await tool._ensure_initialized()

    # Pre-existing name/description are preserved
    assert tool.name == "local-name"
    assert tool.description == "local-desc"


@pytest.mark.unit
@pytest.mark.a2a
async def test_call_remote_returns_last_event_content_until_final(monkeypatch, config):
    """
    New behavior: tool returns the *last* ev.content received before FINAL_ANSWER / INPUT_REQUIRED.
    No concatenation is performed.
    """

    class FakeA2AClient:
        def __init__(self, config):
            self.config = config

        async def get_agent_card(self):
            return DummyAgentCard(name="remote-agent", description="desc")

        async def send_message(self, message: str, context_id: None = None):
            yield UpdateEvent(content="Hello ")
            yield UpdateEvent(content="Hello world")
            yield FinalAnswerEvent(content="Hello world")

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool: A2AAgentTool = await A2AAgentTool.from_config(config)

    result = await tool._func("hi there")  # type: ignore[attr-defined]
    assert result == "Hello world"


@pytest.mark.unit
@pytest.mark.a2a
async def test_call_remote_no_events_returns_empty_string(monkeypatch, config):
    class FakeA2AClient:
        def __init__(self, config):
            self.config = config

        async def get_agent_card(self):
            return DummyAgentCard(name="remote-agent", description="desc")

        async def send_message(self, message: str, context_id: None = None):
            if False:
                yield None

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool: A2AAgentTool = await A2AAgentTool.from_config(config)

    result = await tool._func("hello")  # type: ignore[attr-defined]
    assert result == ""


@pytest.mark.unit
@pytest.mark.a2a
async def test_call_remote_exceptions_bubble(monkeypatch, config):
    class FakeA2AClient:
        def __init__(self, config):
            self.config = config

        async def get_agent_card(self):
            return DummyAgentCard(name="remote-agent", description="desc")

        async def send_message(self, message: str, context_id: None = None):
            raise RuntimeError("boom")
            if False:  # pragma: no cover
                yield None

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool: A2AAgentTool = await A2AAgentTool.from_config(config)

    with pytest.raises(RuntimeError, match="boom"):
        await tool._func("hello")  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.a2a
async def test_from_config_without_agent_card_calls_ensure_initialized(
    monkeypatch, config
):
    called = {"count": 0}

    async def fake_ensure_initialized(self):
        called["count"] += 1
        self._initialized = True
        self._agent_card = DummyAgentCard(name="remote-agent", description="desc")
        self._attach_func()

    monkeypatch.setattr(
        A2AAgentTool, "_ensure_initialized", fake_ensure_initialized, raising=False
    )

    tool: A2AAgentTool = await A2AAgentTool.from_config(config)
    assert isinstance(tool, A2AAgentTool)
    assert tool.config is config
    assert called["count"] == 1


@pytest.mark.unit
@pytest.mark.a2a
def test_from_config_with_agent_card_is_sync_and_sets_metadata(config):
    agent_card = DummyAgentCard(
        name="remote-agent",
        description="Remote description",
        skills=[DummySkill("skill_one")],
    )

    tool: A2AAgentTool = A2AAgentTool.from_config(config, agent_card=agent_card)

    assert tool._initialized is True
    assert tool._agent_card is agent_card
    assert tool.name == "remote-agent"
    assert "Remote description" in tool.description
    assert "skill_one" in tool.description
    assert callable(tool._func)


@pytest.mark.unit
@pytest.mark.a2a
async def test_create_agent_description_includes_all_skills(config):
    tool = A2AAgentTool(
        name="agent",
        description="desc",
        parameters={},
        config=config,
    )

    skills = [DummySkill("skill_a"), DummySkill("skill_b")]
    card = DummyAgentCard(description="base-desc", skills=skills)

    description = tool._create_agent_description(card)

    assert description.startswith("base-desc")
    assert "skill_a" in description
    assert "skill_b" in description


@pytest.mark.unit
@pytest.mark.a2a
async def test_a2a_agent_tool_model_dump_openai_compatible(monkeypatch, config):
    """
    Same intent as before: validate Tool.model_dump() emits an OpenAI-compatible
    tool spec *after* initialization.
    """
    agent_card = DummyAgentCard(
        name="remote-agent",
        description="Remote description",
        skills=[DummySkill("skill_one")],
    )

    class FakeA2AClient:
        def __init__(self, config):
            self.config = config

        async def get_agent_card(self):
            return agent_card

        async def send_message(self, message: str, context_id: None = None):
            if False:  # pragma: no cover
                yield None

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool: A2AAgentTool = await A2AAgentTool.from_config(config)

    spec = tool.model_dump()

    assert spec["type"] == "function"
    assert "function" in spec

    fn = spec["function"]
    assert fn["name"] == "remote-agent"
    assert "Remote description" in fn["description"]
    assert "skill_one" in fn["description"]

    params = fn["parameters"]
    assert isinstance(params, dict)
    assert params.get("type") == "object"

    props = params.get("properties", {})
    assert set(props.keys()) == {"message"}

    message_prop = props["message"]
    assert message_prop.get("type") == "string"
    assert "Message to send to the agent" in message_prop.get("description", "")

    assert params.get("required") == ["message"]


@pytest.mark.unit
@pytest.mark.a2a
def test_a2a_agent_tool_is_not_namespace(config):
    tool = A2AAgentTool(
        name="some-tool",
        description="desc",
        parameters={},
        config=config,
    )
    assert tool.is_namespace is False


@pytest.mark.unit
@pytest.mark.a2a
async def test_a2a_agent_tool_is_single_callable_and_ainvoke_works(monkeypatch, config):
    class FakeA2AClient:
        def __init__(self, config):
            self.config = config

        async def get_agent_card(self):
            return DummyAgentCard(name="remote-agent", description="desc")

        async def send_message(self, message: str, context_id: None = None):
            yield UpdateEvent(content="Hello world")
            yield FinalAnswerEvent(content="Hello world")

    monkeypatch.setattr(MODULE, "A2AClient", FakeA2AClient)

    tool: A2AAgentTool = await A2AAgentTool.from_config(config)

    assert tool.is_namespace is False

    result = await tool.ainvoke(message="hi there")
    assert result == "Hello world"
