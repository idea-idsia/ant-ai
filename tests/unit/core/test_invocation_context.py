from __future__ import annotations

import pytest

from ant_ai.core.types import InvocationContext


@pytest.mark.unit
def test_from_metadata_fills_fields_by_name_and_ignores_unknown():
    ctx = InvocationContext.from_metadata(
        session_id="s1",
        metadata={"user_id": "u-1", "workflow_settings": {"a": 1}, "junk": 1},
    )
    assert ctx.session_id == "s1"
    assert ctx.user_id == "u-1"
    assert ctx.workflow_settings == {"a": 1}
    assert not hasattr(ctx, "junk")


@pytest.mark.unit
def test_from_metadata_session_id_argument_wins_over_metadata():
    ctx = InvocationContext.from_metadata(
        session_id="s1", metadata={"session_id": "spoofed"}
    )
    assert ctx.session_id == "s1"


@pytest.mark.unit
def test_from_metadata_without_metadata():
    ctx = InvocationContext.from_metadata(session_id="s1")
    assert ctx.session_id == "s1"
    assert ctx.user_id is None


@pytest.mark.unit
def test_subclass_fields_filled_from_metadata():
    class MyContext(InvocationContext):
        tenant: str = ""
        tags: list[str] | None = None

    ctx = MyContext.from_metadata(
        session_id="s1", metadata={"tenant": "acme", "tags": ["x"]}
    )
    assert isinstance(ctx, MyContext)
    assert ctx.tenant == "acme"
    assert ctx.tags == ["x"]


@pytest.mark.unit
def test_default_trace_attributes():
    ctx = InvocationContext(session_id="s1", user_id="u-1", llm_settings={"k": "v"})
    assert ctx.trace_attributes() == {"session_id": "s1", "user_id": "u-1"}
