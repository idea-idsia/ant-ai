from __future__ import annotations

import pytest

from ant_ai.core.response import ChatLLMResponse, ChatLLMStreamChunk
from ant_ai.llm.integrations.openai_llm import OpenAIChat


@pytest.mark.unit
def test_to_openai_messages(sample_messages, expected_dict_messages):
    result = OpenAIChat._to_openai_messages(sample_messages)
    assert result == expected_dict_messages


@pytest.mark.unit
def test_to_openai_messages_accepts_tools_param(sample_messages):
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    result = OpenAIChat._to_openai_messages(sample_messages, tools=tools)
    assert isinstance(result, list)
    assert result[0]["role"] == "system"


@pytest.mark.unit
def test_invoke_uses_client_and_returns_message(
    monkeypatch,
    sample_messages,
    openai_sync_response,
):
    chat = OpenAIChat(model="some-model", api_key="dummy")

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return openai_sync_response("Sync response!")

    monkeypatch.setattr(
        chat.client.chat.completions,
        "create",
        fake_create,
        raising=False,
    )

    result: ChatLLMResponse = chat.invoke(sample_messages)

    assert isinstance(result, ChatLLMResponse)
    assert result.message.role == "assistant"
    assert result.message.content == "Sync response!"
    assert captured["model"] == "some-model"
    assert captured["messages"][0]["role"] == "system"


@pytest.mark.unit
async def test_ainvoke_uses_async_client_and_returns_message(
    monkeypatch,
    sample_messages,
    openai_sync_response,
):
    chat = OpenAIChat(model="async-model", api_key="dummy")

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return openai_sync_response("Async response!")

    monkeypatch.setattr(
        chat.async_client.chat.completions,
        "create",
        fake_create,
        raising=False,
    )

    result: ChatLLMResponse = await chat.ainvoke(sample_messages)

    assert isinstance(result, ChatLLMResponse)
    assert result.message.role == "assistant"
    assert result.message.content == "Async response!"
    assert captured["model"] == "async-model"


@pytest.mark.unit
async def test_stream_yields_message_chunks_and_skips_none(
    monkeypatch,
    sample_messages,
    collect_stream_chunks,
    openai_stream_chunk,
    make_async_stream,
):
    chat = OpenAIChat(model="stream-model", api_key="dummy")

    parts = ["Hel", None, "lo ", "world", None]

    async def fake_create(**kwargs):
        assert kwargs["model"] == "stream-model"
        assert kwargs["stream"] is True
        return make_async_stream(openai_stream_chunk(p) for p in parts)

    monkeypatch.setattr(
        chat.async_client.chat.completions,
        "create",
        fake_create,
        raising=False,
    )

    chunks_list, combined = await collect_stream_chunks(chat.stream(sample_messages))

    assert combined == "Hello world"
    for c in chunks_list:
        assert isinstance(c, ChatLLMStreamChunk)
        assert c.delta.role == "assistant"
        assert c.delta.delta  # never empty due to skip logic


@pytest.mark.unit
async def test_stream_with_all_empty_produces_no_chunks(
    monkeypatch,
    sample_messages,
    collect_stream_chunks,
    openai_stream_chunk,
    make_async_stream,
):
    chat = OpenAIChat(model="stream-model", api_key="dummy")

    parts = [None, None]

    async def fake_create(**kwargs):
        return make_async_stream(openai_stream_chunk(p) for p in parts)

    monkeypatch.setattr(
        chat.async_client.chat.completions,
        "create",
        fake_create,
        raising=False,
    )

    chunks_list, combined = await collect_stream_chunks(chat.stream(sample_messages))
    assert chunks_list == []
    assert combined == ""


@pytest.mark.unit
async def test_stream_yields_tool_call_fragments_in_order(
    monkeypatch,
    sample_messages,
    openai_tool_call_chunk,
    make_async_stream,
):
    chat = OpenAIChat(model="stream-model", api_key="dummy")

    fragments = [
        openai_tool_call_chunk(
            index=0, call_id="call-1", name="my_tool", arguments='{"a"'
        ),
        openai_tool_call_chunk(index=0, arguments=": 1}"),
    ]

    async def fake_create(**kwargs):
        return make_async_stream(fragments)

    monkeypatch.setattr(
        chat.async_client.chat.completions,
        "create",
        fake_create,
        raising=False,
    )

    chunks: list[ChatLLMStreamChunk] = [c async for c in chat.stream(sample_messages)]

    assert [c.tool_calls for c in chunks] == [
        {"index": 0, "id": "call-1", "name": "my_tool", "arguments": '{"a"'},
        {"index": 0, "id": None, "name": None, "arguments": ": 1}"},
    ]


def _bad_request(code: str | None):
    import httpx
    import openai

    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.openai.com/v1/chat")
    )
    return openai.BadRequestError(
        "provider text with the prompt in it",
        response=response,
        body={
            "code": code,
            "message": "provider text",
            "type": "invalid_request_error",
        },
    )


@pytest.mark.unit
async def test_context_length_exceeded_raises_the_library_exception(
    monkeypatch, sample_messages
):
    import openai

    from ant_ai.llm import ContextWindowExceededError

    chat = OpenAIChat(model="some-model", api_key="dummy")

    def overflow(**kwargs):
        raise _bad_request("context_length_exceeded")

    async def aoverflow(**kwargs):
        overflow()

    monkeypatch.setattr(chat.client.chat.completions, "create", overflow)
    monkeypatch.setattr(chat.async_client.chat.completions, "create", aoverflow)

    with pytest.raises(ContextWindowExceededError) as info:
        chat.invoke(sample_messages)
    assert info.value.model == "some-model"
    assert isinstance(info.value.__cause__, openai.BadRequestError)
    assert "prompt" not in str(info.value)

    with pytest.raises(ContextWindowExceededError):
        await chat.ainvoke(sample_messages)

    with pytest.raises(ContextWindowExceededError):
        async for _ in chat.stream(sample_messages):
            pass


@pytest.mark.unit
def test_other_bad_requests_pass_through_untouched(monkeypatch, sample_messages):
    """Only the context-window code is translated; a different 400 is still the
    provider's error, since we cannot say anything truer about it."""
    import openai

    chat = OpenAIChat(model="some-model", api_key="dummy")

    def bad_value(**kwargs):
        raise _bad_request("invalid_value")

    monkeypatch.setattr(chat.client.chat.completions, "create", bad_value)

    with pytest.raises(openai.BadRequestError):
        chat.invoke(sample_messages)
