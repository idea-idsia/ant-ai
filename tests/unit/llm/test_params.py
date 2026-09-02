from __future__ import annotations

import pytest
from pydantic import ValidationError

from ant_ai.core.types import InvocationContext, LLMSettings
from ant_ai.llm.params import resolve_llm_params


@pytest.mark.unit
def test_overrides_omits_unset_fields():
    assert LLMSettings().overrides() == {}
    assert LLMSettings(temperature=0.4).overrides() == {"temperature": 0.4}
    assert LLMSettings(temperature=0.4, reasoning_effort="high").overrides() == {
        "temperature": 0.4,
        "reasoning_effort": "high",
    }


@pytest.mark.unit
def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        LLMSettings.model_validate({"top_p": 0.9})


@pytest.mark.unit
def test_temperature_bounds_enforced():
    with pytest.raises(ValidationError):
        LLMSettings(temperature=2.5)
    with pytest.raises(ValidationError):
        LLMSettings(temperature=-0.1)


@pytest.mark.unit
def test_reasoning_effort_literal_enforced():
    with pytest.raises(ValidationError):
        LLMSettings.model_validate({"reasoning_effort": "ultra"})


@pytest.mark.unit
def test_resolve_no_ctx_returns_default_params_copy():
    default = {"extra_body": {"x": 1}}
    out = resolve_llm_params(default, LLMSettings(), None)
    assert out == {"extra_body": {"x": 1}}
    assert out is not default  # copy, not alias


@pytest.mark.unit
def test_resolve_layers_precedence_request_wins():
    default = {"temperature": 0.1, "num_retries": 3}
    base = LLMSettings(temperature=0.5)
    ctx = InvocationContext(session_id="s1", llm_settings=LLMSettings(temperature=0.9))

    out = resolve_llm_params(default, base, ctx)

    assert out == {"temperature": 0.9, "num_retries": 3}


@pytest.mark.unit
def test_resolve_base_overrides_default_params():
    out = resolve_llm_params({"temperature": 0.1}, LLMSettings(temperature=0.5), None)
    assert out["temperature"] == 0.5


@pytest.mark.unit
def test_resolve_ctx_without_llm_settings_is_ignored():
    ctx = InvocationContext(session_id="s1")
    out = resolve_llm_params({}, LLMSettings(temperature=0.3), ctx)
    assert out == {"temperature": 0.3}


@pytest.mark.unit
def test_resolve_does_not_mutate_default_params():
    default = {"temperature": 0.1}
    resolve_llm_params(
        default,
        LLMSettings(temperature=0.5),
        InvocationContext(
            session_id="s1", llm_settings=LLMSettings(reasoning_effort="low")
        ),
    )
    assert default == {"temperature": 0.1}
