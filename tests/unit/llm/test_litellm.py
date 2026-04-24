from __future__ import annotations

import pytest

from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk
from ant_ai.llm.integrations.lite_llm import LiteLLMChat, to_chatllm_response


@pytest.mark.unit
def test_to_litellm_messages(sample_messages):
    result: list[dict[str, str]] = LiteLLMChat._to_litellm_messages(sample_messages)
    assert result == [m.model_dump(exclude={"kind"}) for m in sample_messages]


@pytest.mark.unit
def test_invoke_calls_completion_and_returns_message(
    monkeypatch,
    sample_messages,
    litellm_response,
):
    import ant_ai.llm.integrations.lite_llm as wrapper_module

    def fake_completion(
        *, model, messages, api_base=None, api_key=None, tools=None, **kwargs
    ):
        assert model == "litellm-model"
        assert messages[0]["role"] == "system"
        # tools is omitted (None) when no tools are passed — passing tools=[]
        # causes errors on some endpoints, so we only send tools when non-empty.
        assert tools is None
        return litellm_response("LiteLLM sync response!")

    monkeypatch.setattr(wrapper_module, "completion", fake_completion)

    chat = LiteLLMChat(model="litellm-model")
    result: ChatLLMResponse = chat.invoke(sample_messages)

    assert isinstance(result, ChatLLMResponse)
    assert result.message.role == "assistant"
    assert result.message.content == "LiteLLM sync response!"


@pytest.mark.unit
def test_invoke_passes_tools_through(monkeypatch, sample_messages, litellm_response):
    import ant_ai.llm.integrations.lite_llm as wrapper_module

    tools = [{"type": "function", "function": {"name": "hello", "parameters": {}}}]

    def fake_completion(*, model, messages, tools=None, **kwargs):
        assert tools == [
            {"type": "function", "function": {"name": "hello", "parameters": {}}}
        ]
        return litellm_response("ok")

    monkeypatch.setattr(wrapper_module, "completion", fake_completion)

    chat = LiteLLMChat(model="litellm-model")
    _: ChatLLMResponse = chat.invoke(sample_messages, tools=tools)


@pytest.mark.unit
async def test_ainvoke_calls_acompletion_and_returns_message(
    monkeypatch,
    sample_messages,
    litellm_response,
):
    import ant_ai.llm.integrations.lite_llm as wrapper_module

    async def fake_acompletion(*, model, messages, **kwargs):
        assert model == "litellm-model"
        assert messages[0]["role"] == "system"
        return litellm_response("LiteLLM async response!")

    monkeypatch.setattr(wrapper_module, "acompletion", fake_acompletion)

    chat = LiteLLMChat(model="litellm-model")
    chat.default_params = {}

    result: ChatLLMResponse = await chat.ainvoke(sample_messages)

    assert isinstance(result, ChatLLMResponse)
    assert result.message.role == "assistant"
    assert result.message.content == "LiteLLM async response!"


@pytest.mark.unit
async def test_stream_yields_message_chunks_and_skips_empty_deltas(
    monkeypatch,
    sample_messages,
    collect_stream_chunks,
    litellm_stream_chunk,
    make_async_stream,
):
    import ant_ai.llm.integrations.lite_llm as wrapper_module

    parts = ["Lit", None, "", "eLLM ", "stream"]

    async def fake_acompletion(*, model, messages, stream=False, **kwargs):
        assert model == "litellm-stream-model"
        assert stream is True
        return make_async_stream(litellm_stream_chunk(p) for p in parts)

    monkeypatch.setattr(wrapper_module, "acompletion", fake_acompletion)

    chat = LiteLLMChat(model="litellm-stream-model")
    chat.default_params = {}

    chunks_list, combined = await collect_stream_chunks(chat.stream(sample_messages))
    assert all(isinstance(c, ChatLLMStreamChunk) for c in chunks_list)
    assert all(c.delta.role == "assistant" for c in chunks_list)
    assert combined.replace(" ", "") == "LiteLLMstream"


@pytest.mark.unit
def test_to_chatllm_response_maps_tool_calls(litellm_response, dummy_tool_call):
    """Validates the tool call mapping in to_chatllm_response()."""
    tool_calls = [
        dummy_tool_call(call_id="call_1", name="do_thing", arguments='{"x":1}'),
        dummy_tool_call(call_id="call_2", name=None, arguments=None),
    ]
    resp = litellm_response("hello", tool_calls=tool_calls)

    out: ChatLLMResponse = to_chatllm_response(resp)

    assert out.message.role == "assistant"
    assert out.message.content == "hello"
    assert len(out.tool_calls) == 2

    assert out.tool_calls[0].id == "call_1"
    assert out.tool_calls[0].function.name == "do_thing"
    assert out.tool_calls[0].function.arguments == '{"x":1}'

    assert out.tool_calls[1].id == "call_2"
    assert out.tool_calls[1].function.name == ""
    assert out.tool_calls[1].function.arguments == ""
