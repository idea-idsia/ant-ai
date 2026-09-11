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


@pytest.mark.unit
def test_outbound_metadata_excludes_session_and_per_callee_settings():
    class MyContext(InvocationContext):
        tenant: str = ""
        tags: list[str] | None = None

    ctx = MyContext(
        session_id="s1",
        user_id="u-1",
        tenant="acme",
        llm_settings={"t": 0},
        workflow_settings={"max_steps": 3},
    )
    out = ctx.outbound_metadata()

    assert out == {"user_id": "u-1", "tenant": "acme"}  # tags=None left out
    rebuilt = MyContext.from_metadata(session_id="s2", metadata=out)
    assert rebuilt == MyContext(session_id="s2", user_id="u-1", tenant="acme")


@pytest.mark.unit
def test_outbound_metadata_can_withhold_fields():
    class MyContext(InvocationContext):
        api_token: str = ""

        def outbound_metadata(self):
            return {
                k: v for k, v in super().outbound_metadata().items() if k != "api_token"
            }

    assert (
        "api_token"
        not in MyContext(session_id="s", api_token="secret").outbound_metadata()
    )
