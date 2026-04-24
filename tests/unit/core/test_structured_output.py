from __future__ import annotations

import pytest
from pydantic import BaseModel, model_serializer

from ant_ai.agent.agent import Agent
from ant_ai.core.events import FinalAnswerEvent
from ant_ai.core.message import Message, ToolCall, ToolFunction
from ant_ai.core.types import State
from ant_ai.llm.protocol import ChatLLM
from ant_ai.tools.tool import Tool


class Location(BaseModel):
    city: str
    country: str


class DummyResponse:
    def __init__(self, content: str, tool_calls=None):
        self.message = Message(role="assistant", content=content)
        self.tool_calls = tool_calls or []


def _agent(llm) -> Agent:
    return Agent(name="test-agent", system_prompt="sys", llm=llm, tools=[])


def _state(content: str = "question") -> State:
    s = State()
    s.add_message(Message(role="user", content=content))
    return s


def _tool_call(name: str = "lookup", call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, function=ToolFunction(name=name, arguments="{}"))


class _LookupTool(Tool):
    name: str = "lookup"
    description: str = "lookup"
    parameters: dict = {}

    async def ainvoke(self, **kwargs):
        return "result"

    @model_serializer
    def _serialize(self):
        return {"name": self.name, "description": self.description}


def _agent_with_tool(llm) -> Agent:
    return Agent(name="test-agent", system_prompt="sys", llm=llm, tools=[_LookupTool()])


@pytest.mark.unit
async def test_no_schema_llm_called_once():
    call_count = 0

    class CountingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            return DummyResponse("plain text answer")

    result = await _agent(CountingLLM()).ainvoke(_state())

    assert call_count == 1
    assert result == "plain text answer"


@pytest.mark.unit
async def test_with_schema_no_tools_llm_called_once():
    """No-tools agent: response_format is injected natively — only one LLM call."""
    call_count = 0

    class NativeSchemaLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            # Simulate real LLM behaviour: return valid JSON when response_format is set.
            if response_format is not None:
                return DummyResponse('{"city": "Tokyo", "country": "Japan"}')
            return DummyResponse("The city is Tokyo and the country is Japan.")

    result = await _agent(NativeSchemaLLM()).ainvoke(_state(), response_schema=Location)

    assert call_count == 1
    parsed = Location.model_validate_json(result)
    assert parsed.city == "Tokyo"
    assert parsed.country == "Japan"


@pytest.mark.unit
async def test_with_schema_no_tools_passes_response_format_to_llm():
    """When act_step is None, LLM receives response_format on the first (and only) call."""
    received_formats: list = []

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            received_formats.append(response_format)
            return DummyResponse('{"city": "Paris", "country": "France"}')

    await _agent(RecordingLLM()).ainvoke(_state(), response_schema=Location)

    assert len(received_formats) == 1
    assert received_formats[0] is Location


@pytest.mark.unit
async def test_with_tools_and_schema_repair_call_when_free_text():
    """With tools present, if LLM returns free text on final step, a repair call is made."""
    call_count = 0

    class ThreeCallLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DummyResponse("", tool_calls=[_tool_call()])
            if call_count == 2:
                return DummyResponse("The city is Tokyo and the country is Japan.")
            # Structuring repair call — return valid JSON
            return DummyResponse('{"city": "Tokyo", "country": "Japan"}')

    result = await _agent_with_tool(ThreeCallLLM()).ainvoke(
        _state(), response_schema=Location
    )

    assert call_count == 3
    parsed = Location.model_validate_json(result)
    assert parsed.city == "Tokyo"


@pytest.mark.unit
async def test_with_tools_and_schema_skips_repair_when_json_valid():
    """With tools present, if LLM already returns valid JSON, no repair call is needed."""
    call_count = 0

    class TwoCallLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DummyResponse("", tool_calls=[_tool_call()])
            # Final answer — already valid JSON
            return DummyResponse('{"city": "Tokyo", "country": "Japan"}')

    result = await _agent_with_tool(TwoCallLLM()).ainvoke(
        _state(), response_schema=Location
    )

    assert call_count == 2
    parsed = Location.model_validate_json(result)
    assert parsed.city == "Tokyo"


@pytest.mark.unit
async def test_structuring_call_receives_final_answer_as_input():
    """The repair/structuring call gets only the free-text answer as input, not the full history."""
    calls: list[dict] = []

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            calls.append(
                {"messages": list(messages), "response_format": response_format}
            )
            if len(calls) == 1:
                return DummyResponse("", tool_calls=[_tool_call()])
            if len(calls) == 2:
                return DummyResponse("The answer is Paris, France.")
            return DummyResponse('{"city": "Paris", "country": "France"}')

    await _agent_with_tool(RecordingLLM()).ainvoke(
        _state("What city?"), response_schema=Location
    )

    assert len(calls) == 3

    # Main reasoning calls include the user question
    main_contents = [m.content for m in calls[1]["messages"]]
    assert any("What city?" in (c or "") for c in main_contents)

    # Structuring/repair call — user message is the free-text final answer only
    structuring_contents = [m.content for m in calls[2]["messages"]]
    assert "The answer is Paris, France." in structuring_contents
    assert not any("What city?" in (c or "") for c in structuring_contents)


@pytest.mark.unit
async def test_structuring_call_has_no_tools():
    """The repair/structuring call is sent without tools and with response_format set."""
    calls: list[dict] = []

    class RecordingLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            calls.append({"tools": tools, "response_format": response_format})
            if len(calls) == 1:
                return DummyResponse("", tool_calls=[_tool_call()])
            if len(calls) == 2:
                return DummyResponse("Berlin, Germany.")
            return DummyResponse('{"city": "Berlin", "country": "Germany"}')

    await _agent_with_tool(RecordingLLM()).ainvoke(_state(), response_schema=Location)

    assert len(calls) == 3
    # Structuring/repair call must have no tools and response_format set
    assert not calls[2]["tools"]
    assert calls[2]["response_format"] is Location


@pytest.mark.unit
async def test_structured_output_after_tool_call():
    """Full tool → reasoning → repair flow with location schema."""
    call_count = 0

    class MultiStepLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DummyResponse("", tool_calls=[_tool_call()])
            if call_count == 2:
                return DummyResponse("The capital is Oslo, Norway.")
            return DummyResponse('{"city": "Oslo", "country": "Norway"}')

    agent = Agent(
        name="test-agent",
        system_prompt="sys",
        llm=MultiStepLLM(),
        tools=[_LookupTool()],
    )

    result = await agent.ainvoke(_state("Capital of Norway?"), response_schema=Location)

    assert call_count == 3
    parsed = Location.model_validate_json(result)
    assert parsed.city == "Oslo"
    assert parsed.country == "Norway"


@pytest.mark.unit
async def test_final_answer_event_contains_structured_json():
    """FinalAnswerEvent content is valid JSON matching the schema."""
    call_count = 0

    class NativeSchemaLLM(ChatLLM):
        async def ainvoke(
            self, messages, *, ctx=None, tools=None, response_format=None
        ):
            nonlocal call_count
            call_count += 1
            if response_format is not None:
                return DummyResponse('{"city": "Rome", "country": "Italy"}')
            return DummyResponse("It is in Rome, Italy.")

    events = [
        e
        async for e in _agent(NativeSchemaLLM()).stream(
            _state(), response_schema=Location
        )
    ]

    final = next(e for e in events if isinstance(e, FinalAnswerEvent))
    assert final.content == '{"city": "Rome", "country": "Italy"}'
    # Must be parseable
    Location.model_validate_json(final.content)
    assert call_count == 1  # native schema, one call
