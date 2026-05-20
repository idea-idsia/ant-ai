from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ant_ai.skills.protocol import AgentSkill


def _make_skill(**overrides) -> AgentSkill:
    defaults = {
        "name": "my-skill",
        "description": "A valid skill.",
        "instructions": "Do something.",
        "skill_dir": Path("/tmp/my-skill"),
    }
    return AgentSkill(**(defaults | overrides))


@pytest.mark.unit
def test_valid_name_accepted():
    assert _make_skill(name="my-skill").name == "my-skill"


@pytest.mark.unit
def test_single_char_name_accepted():
    assert _make_skill(name="a").name == "a"


@pytest.mark.unit
def test_name_with_numbers_accepted():
    assert _make_skill(name="skill123").name == "skill123"


@pytest.mark.unit
def test_uppercase_name_rejected():
    with pytest.raises(ValidationError):
        _make_skill(name="My-Skill")


@pytest.mark.unit
def test_leading_hyphen_rejected():
    with pytest.raises(ValidationError):
        _make_skill(name="-bad")


@pytest.mark.unit
def test_trailing_hyphen_rejected():
    with pytest.raises(ValidationError):
        _make_skill(name="bad-")


@pytest.mark.unit
def test_consecutive_hyphens_rejected():
    with pytest.raises(ValidationError):
        _make_skill(name="bad--name")


@pytest.mark.unit
def test_name_too_long_rejected():
    with pytest.raises(ValidationError):
        _make_skill(name="a" * 65)


@pytest.mark.unit
def test_name_max_length_accepted():
    assert len(_make_skill(name="a" * 64).name) == 64


@pytest.mark.unit
def test_empty_description_rejected():
    with pytest.raises(ValidationError):
        _make_skill(description="")


@pytest.mark.unit
def test_description_too_long_rejected():
    with pytest.raises(ValidationError):
        _make_skill(description="x" * 1025)


@pytest.mark.unit
def test_description_max_length_accepted():
    assert len(_make_skill(description="x" * 1024).description) == 1024


@pytest.mark.unit
def test_compatibility_too_long_rejected():
    with pytest.raises(ValidationError):
        _make_skill(compatibility="x" * 501)


@pytest.mark.unit
def test_compatibility_max_length_accepted():
    skill = _make_skill(compatibility="x" * 500)
    assert skill.compatibility is not None and len(skill.compatibility) == 500


@pytest.mark.unit
def test_optional_fields_default_to_none_or_empty():
    skill = _make_skill()
    assert skill.license is None
    assert skill.compatibility is None
    assert skill.metadata == {}
    assert skill.allowed_tools == []
    assert skill.scripts == []


@pytest.mark.unit
def test_all_optional_fields_set():
    skill = _make_skill(
        license="MIT",
        compatibility="Requires Python 3.14",
        metadata={"author": "me", "version": "1.0"},
        allowed_tools=["Bash", "Read"],
        scripts=[Path("/tmp/my-skill/scripts/run.py")],
    )
    assert skill.license == "MIT"
    assert skill.compatibility == "Requires Python 3.14"
    assert skill.metadata == {"author": "me", "version": "1.0"}
    assert skill.allowed_tools == ["Bash", "Read"]
    assert len(skill.scripts) == 1
