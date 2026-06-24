from __future__ import annotations

import pytest
from pydantic import ValidationError

from ant_ai.core.message import (
    Message,
    ToolCall,
    ToolCallMessage,
    ToolCallResultMessage,
    ToolFunction,
)
from ant_ai.core.types import State
from ant_ai.hooks.builtins.history_compression import (
    _SUMMARY_PREFIX,
    HistoryCompressionHook,
)


class _DummyResponse:
    def __init__(self, content: str = "summary text"):
        self.message = Message(role="assistant", content=content)
        self.tool_calls = []


class _CapturingLLM:
    """Records calls and returns a configurable summary."""

    def __init__(self, summary: str = "summary text"):
        self.calls: list[list[Message]] = []
        self._summary = summary

    async def ainvoke(self, messages, *, ctx=None, tools=None, response_format=None):
        self.calls.append(list(messages))
        return _DummyResponse(self._summary)


def _state(*contents: str) -> State:
    s = State()
    for c in contents:
        s.add_message(Message(role="user", content=c))
    return s


@pytest.mark.unit
def test_raises_when_no_threshold_set():
    with pytest.raises(ValidationError, match="max_messages or max_token_ratio"):
        HistoryCompressionHook(llm=_CapturingLLM())


@pytest.mark.unit
def test_raises_when_token_ratio_without_context_window():
    with pytest.raises(ValidationError, match="context_window"):
        HistoryCompressionHook(llm=_CapturingLLM(), max_token_ratio=0.75)


@pytest.mark.unit
def test_valid_with_max_messages_only():
    hook = HistoryCompressionHook(llm=_CapturingLLM(), max_messages=10)
    assert hook.max_messages == 10


@pytest.mark.unit
def test_valid_with_token_ratio_and_context_window():
    hook = HistoryCompressionHook(
        llm=_CapturingLLM(), max_token_ratio=0.75, context_window=128_000
    )
    assert hook.max_token_ratio == 0.75


@pytest.mark.unit
async def test_compresses_when_max_messages_exceeded():
    llm = _CapturingLLM("old history summarised")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=2)

    state = _state("m1", "m2", "m3", "m4", "m5")
    await hook.before_model(state, ctx=None)

    assert llm.calls, "LLM should have been called for summarisation"
    # 1 summary message + 2 kept
    assert len(state.messages) == 3
    assert state.messages[0].content == f"{_SUMMARY_PREFIX}old history summarised"
    assert state.messages[1].content == "m4"
    assert state.messages[2].content == "m5"


@pytest.mark.unit
async def test_no_compression_below_max_messages():
    llm = _CapturingLLM()
    hook = HistoryCompressionHook(llm=llm, max_messages=10, keep_last=2)

    state = _state("m1", "m2", "m3")
    await hook.before_model(state, ctx=None)

    assert not llm.calls, "LLM should not be called when below threshold"
    assert len(state.messages) == 3


@pytest.mark.unit
async def test_compresses_at_exact_max_messages_boundary():
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=4, keep_last=2)

    state = _state("m1", "m2", "m3", "m4")
    await hook.before_model(state, ctx=None)

    assert llm.calls


@pytest.mark.unit
async def test_compresses_when_token_ratio_exceeded():
    llm = _CapturingLLM("token summary")
    # 10 messages × 100 chars = 1000 chars → ~250 tokens; window=300 → ratio≈0.83 > 0.75
    hook = HistoryCompressionHook(
        llm=llm, max_token_ratio=0.75, context_window=300, keep_last=2
    )

    state = State()
    for _i in range(10):
        state.add_message(Message(role="user", content="x" * 100))

    await hook.before_model(state, ctx=None)

    assert llm.calls, "LLM should have been called when ratio exceeded"
    assert len(state.messages) == 3  # summary + 2 kept


@pytest.mark.unit
async def test_no_compression_when_token_ratio_below_threshold():
    llm = _CapturingLLM()
    # 2 messages × 10 chars = 20 chars → ~5 tokens; window=10000 → tiny ratio
    hook = HistoryCompressionHook(
        llm=llm, max_token_ratio=0.75, context_window=10_000, keep_last=2
    )

    state = _state("hello", "world")
    await hook.before_model(state, ctx=None)

    assert not llm.calls


@pytest.mark.unit
async def test_keeps_last_n_messages_verbatim():
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=3, keep_last=3)

    # Exactly keep_last messages — nothing to compress
    state = _state("a", "b", "c")
    await hook.before_model(state, ctx=None)

    assert not llm.calls
    assert [m.content for m in state.messages] == ["a", "b", "c"]


@pytest.mark.unit
async def test_keeps_last_n_messages_after_compression():
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=3)

    state = _state("old1", "old2", "recent1", "recent2", "recent3")
    await hook.before_model(state, ctx=None)

    assert llm.calls
    kept = [m.content for m in state.messages[1:]]
    assert kept == ["recent1", "recent2", "recent3"]


@pytest.mark.unit
async def test_summary_injected_as_system_message():
    llm = _CapturingLLM("the prior context")
    hook = HistoryCompressionHook(llm=llm, max_messages=3, keep_last=1)

    state = _state("m1", "m2", "m3")
    await hook.before_model(state, ctx=None)

    summary_msg = state.messages[0]
    assert summary_msg.role == "system"
    assert summary_msg.content == f"{_SUMMARY_PREFIX}the prior context"


@pytest.mark.unit
async def test_llm_receives_messages_to_compress():
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=4, keep_last=2)

    state = _state("compressed1", "compressed2", "kept1", "kept2")
    await hook.before_model(state, ctx=None)

    assert llm.calls
    prompt_content = llm.calls[0][0].content
    assert "compressed1" in prompt_content
    assert "compressed2" in prompt_content
    # kept messages should NOT be sent to the summariser
    assert "kept1" not in prompt_content
    assert "kept2" not in prompt_content


@pytest.mark.unit
async def test_compression_context_set_when_compressed():
    """state._compression_context holds the baseline after compression fires."""
    llm = _CapturingLLM("summary text")
    hook = HistoryCompressionHook(llm=llm, max_messages=4, keep_last=2)

    # 4 messages: compressed1, compressed2, kept1, user_N (last = "current user msg")
    state = _state("compressed1", "compressed2", "kept1", "user_N")
    await hook.before_model(state, ctx=None)

    assert llm.calls, "LLM should have been called"
    # Baseline = state.messages[:-1] = [summary, kept1]  (excludes user_N)
    assert state._compression_context is not None
    assert len(state._compression_context) == 2
    assert state._compression_context[0].role == "system"
    assert _SUMMARY_PREFIX in (state._compression_context[0].content or "")
    assert state._compression_context[1].content == "kept1"


@pytest.mark.unit
async def test_compression_context_none_when_no_compression():
    """state._compression_context is None when compression does not fire."""
    llm = _CapturingLLM()
    hook = HistoryCompressionHook(llm=llm, max_messages=10, keep_last=2)

    state = _state("m1", "m2", "m3")
    await hook.before_model(state, ctx=None)

    assert not llm.calls
    assert state._compression_context is None


@pytest.mark.unit
async def test_compression_context_excludes_current_user_message():
    """The last message (current user turn) is excluded from the context baseline."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=3)

    # 5 messages, keep_last=3 → keep = ["keep2", "keep3", "user_N"]
    # Baseline should be [summary, "keep2", "keep3"] (user_N excluded)
    state = _state("old", "keep1", "keep2", "keep3", "user_N")
    await hook.before_model(state, ctx=None)

    assert state._compression_context is not None
    contents = [m.content for m in state._compression_context]
    assert "user_N" not in contents
    assert "keep2" in contents
    assert "keep3" in contents


# ---------------------------------------------------------------------------
# Tool-call sequence integrity
# ---------------------------------------------------------------------------


def _make_tool_group(
    call_id: str = "call_1",
) -> tuple[ToolCallMessage, ToolCallResultMessage]:
    """Return a matched (ToolCallMessage, ToolCallResultMessage) pair."""
    call = ToolCallMessage(
        tool_calls=[
            ToolCall(id=call_id, function=ToolFunction(name="mytool", arguments="{}"))
        ]
    )
    result = ToolCallResultMessage(
        tool_call_id=call_id, name="mytool", content="result"
    )
    return call, result


def _assert_valid_tool_sequence(messages: list[Message]) -> None:
    """Assert no orphaned tool-result messages exist in the sequence."""
    for i, msg in enumerate(messages):
        if msg.role == "tool":
            assert i > 0, f"tool message at index {i} has no preceding message"
            prev = messages[i - 1]
            assert prev.role in ("assistant", "tool"), (
                f"tool message at index {i} is orphaned: preceded by role='{prev.role}'"
            )


@pytest.mark.unit
async def test_orphaned_tool_result_at_boundary_is_fixed():
    """Boundary that would orphan a ToolCallResultMessage is slid left to include its ToolCallMessage."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=3)

    # [user1, tool_call, tool_result, user2, user3]
    # Initial split at index 2 → keep starts with tool_result (role="tool") → orphan!
    # Fix: boundary moves left to index 1 → keep = [tool_call, tool_result, user2, user3]
    call, result = _make_tool_group()
    state = State()
    state.add_message(Message(role="user", content="user1"))
    state.add_message(call)
    state.add_message(result)
    state.add_message(Message(role="user", content="user2"))
    state.add_message(Message(role="user", content="user3"))

    await hook.before_model(state, ctx=None)

    assert llm.calls, "LLM should have been called"
    _assert_valid_tool_sequence(state.messages)


@pytest.mark.unit
async def test_multiple_tool_results_kept_intact():
    """When multiple tool results follow one call, all are slid into keep together."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=3)

    # [user1, tool_call, result1, result2, user2]
    # Initial split at 2 → keep = [result1, result2, user2] → orphaned
    # Fix: boundary slides to 1 → keep = [tool_call, result1, result2, user2]
    call = ToolCallMessage(
        tool_calls=[ToolCall(id="c1", function=ToolFunction(name="t", arguments="{}"))]
    )
    result1 = ToolCallResultMessage(tool_call_id="c1", name="t", content="r1")
    result2 = ToolCallResultMessage(tool_call_id="c1", name="t", content="r2")
    state = State()
    state.add_message(Message(role="user", content="user1"))
    state.add_message(call)
    state.add_message(result1)
    state.add_message(result2)
    state.add_message(Message(role="user", content="user2"))

    await hook.before_model(state, ctx=None)

    assert llm.calls
    _assert_valid_tool_sequence(state.messages)
    # tool_call + both results all land in keep
    roles = [m.role for m in state.messages]
    tool_idx = roles.index("assistant") if "assistant" in roles else -1
    assert tool_idx != -1
    assert roles[tool_idx + 1] == "tool"
    assert roles[tool_idx + 2] == "tool"


@pytest.mark.unit
async def test_tool_group_fully_in_to_compress_is_unaffected():
    """A tool-call group that falls entirely in to_compress requires no boundary adjustment."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=2)

    # [user1, tool_call, tool_result, user2, user3]
    # Initial split at 3 → to_compress = [user1, tool_call, tool_result], keep = [user2, user3]
    # keep[0].role = "user" → no adjustment needed
    call, result = _make_tool_group()
    state = State()
    state.add_message(Message(role="user", content="user1"))
    state.add_message(call)
    state.add_message(result)
    state.add_message(Message(role="user", content="user2"))
    state.add_message(Message(role="user", content="user3"))

    await hook.before_model(state, ctx=None)

    assert llm.calls
    _assert_valid_tool_sequence(state.messages)
    # keep_last=2 preserved: last two user messages
    kept_contents = [m.content for m in state.messages[-2:]]
    assert kept_contents == ["user2", "user3"]


@pytest.mark.unit
async def test_no_compression_when_boundary_adjustment_empties_to_compress():
    """If boundary adjustment leaves nothing to compress, the hook is a no-op."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=5, keep_last=4)

    # [tool_call, result, user1, user2, user3]
    # Initial split at 1 → messages[1].role = "tool" → slide to 0
    # to_compress = [] → no-op
    call, result = _make_tool_group()
    state = State()
    state.add_message(call)
    state.add_message(result)
    state.add_message(Message(role="user", content="user1"))
    state.add_message(Message(role="user", content="user2"))
    state.add_message(Message(role="user", content="user3"))

    original = list(state.messages)
    await hook.before_model(state, ctx=None)

    assert not llm.calls, "LLM should not be called when nothing is left to compress"
    assert state.messages == original


# ---------------------------------------------------------------------------
# keep_last=0 — compress everything
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_keep_last_zero_compresses_all_messages():
    """keep_last=0 should compress the entire history into a single summary message."""
    llm = _CapturingLLM("full history summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=3, keep_last=0)

    state = _state("m1", "m2", "m3")
    await hook.before_model(state, ctx=None)

    assert llm.calls, "LLM should have been called"
    assert len(state.messages) == 1
    assert state.messages[0].role == "system"
    assert state.messages[0].content == f"{_SUMMARY_PREFIX}full history summary"


@pytest.mark.unit
async def test_keep_last_zero_sends_full_history_to_summariser():
    """With keep_last=0 all messages are sent to the summariser, not just a subset."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=3, keep_last=0)

    state = _state("alpha", "beta", "gamma")
    await hook.before_model(state, ctx=None)

    prompt_content = llm.calls[0][0].content
    assert "alpha" in prompt_content
    assert "beta" in prompt_content
    assert "gamma" in prompt_content


@pytest.mark.unit
async def test_keep_last_zero_no_orphaned_tool_messages():
    """With keep_last=0, tool-call groups are fully included in to_compress — no orphans."""
    llm = _CapturingLLM("summary")
    hook = HistoryCompressionHook(llm=llm, max_messages=3, keep_last=0)

    call, result = _make_tool_group()
    state = State()
    state.add_message(Message(role="user", content="q"))
    state.add_message(call)
    state.add_message(result)

    await hook.before_model(state, ctx=None)

    assert llm.calls
    # Everything compressed → only summary remains; no tool messages to orphan
    assert len(state.messages) == 1
    assert state.messages[0].role == "system"
