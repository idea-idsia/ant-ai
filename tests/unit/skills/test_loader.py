from __future__ import annotations

from pathlib import Path

import pytest

from ant_ai.skills.loader import SkillLoader

DATA_DIR = Path(__file__).parent / "data" / "skills"


def _write_skill_md(skill_dir: Path, content: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content)


def _minimal_skill_md(name: str = "my-skill", description: str = "A skill.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\nDo something."


@pytest.mark.unit
def test_load_returns_empty_for_missing_dir(tmp_path):
    assert SkillLoader(tmp_path / "nonexistent").load() == []


@pytest.mark.unit
def test_load_skips_top_level_files(tmp_path):
    (tmp_path / "not-a-dir.md").write_text("hello")
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_folder_without_skill_md(tmp_path):
    (tmp_path / "some-skill").mkdir()
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_parses_name_description_instructions(tmp_path):
    _write_skill_md(tmp_path / "my-skill", _minimal_skill_md())
    skills = SkillLoader(tmp_path).load()
    assert len(skills) == 1
    assert skills[0].name == "my-skill"
    assert skills[0].description == "A skill."
    assert skills[0].instructions == "Do something."


@pytest.mark.unit
def test_load_skips_skill_md_without_frontmatter(tmp_path):
    _write_skill_md(tmp_path / "no-fm", "Just instructions, no frontmatter.")
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_skill_md_with_invalid_yaml(tmp_path):
    _write_skill_md(tmp_path / "bad-yaml", "---\n: :\n---\n\nInstructions.")
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_skill_md_missing_name(tmp_path):
    _write_skill_md(
        tmp_path / "no-name", "---\ndescription: A skill.\n---\n\nInstructions."
    )
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_skill_md_missing_description(tmp_path):
    _write_skill_md(tmp_path / "no-desc", "---\nname: no-desc\n---\n\nInstructions.")
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_skill_with_invalid_name(tmp_path):
    _write_skill_md(
        tmp_path / "bad-name-skill",
        "---\nname: Bad--Name\ndescription: A skill.\n---\n\nInstructions.",
    )
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_skill_when_name_does_not_match_directory(tmp_path):
    _write_skill_md(
        tmp_path / "actual-dir",
        "---\nname: different-name\ndescription: A skill.\n---\n\nInstructions.",
    )
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_skips_skill_with_description_too_long(tmp_path):
    _write_skill_md(
        tmp_path / "long-desc",
        f"---\nname: long-desc\ndescription: {'x' * 1025}\n---\n\nInstructions.",
    )
    assert SkillLoader(tmp_path).load() == []


@pytest.mark.unit
def test_load_collects_scripts_from_scripts_subdir(tmp_path):
    skill_dir = tmp_path / "scripted"
    _write_skill_md(skill_dir, _minimal_skill_md(name="scripted"))
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('hello')")
    (scripts_dir / "helper.sh").write_text("#!/bin/bash\necho hi")

    skills = SkillLoader(tmp_path).load()
    assert len(skills) == 1
    assert {p.name for p in skills[0].scripts} == {"run.py", "helper.sh"}


@pytest.mark.unit
def test_load_scripts_empty_when_no_scripts_dir(tmp_path):
    _write_skill_md(tmp_path / "no-scripts", _minimal_skill_md(name="no-scripts"))
    assert SkillLoader(tmp_path).load()[0].scripts == []


@pytest.mark.unit
def test_load_multiple_skills_sorted_by_folder_name(tmp_path):
    for name in ["zebra-skill", "alpha-skill", "middle-skill"]:
        _write_skill_md(tmp_path / name, _minimal_skill_md(name=name))
    skills = SkillLoader(tmp_path).load()
    assert [s.name for s in skills] == ["alpha-skill", "middle-skill", "zebra-skill"]


@pytest.mark.unit
def test_load_skill_dir_is_absolute(tmp_path):
    _write_skill_md(tmp_path / "my-skill", _minimal_skill_md())
    assert SkillLoader(tmp_path).load()[0].skill_dir.is_absolute()


@pytest.mark.unit
def test_load_parses_optional_fields(tmp_path):
    content = (
        "---\n"
        "name: full-skill\n"
        "description: A full skill.\n"
        "license: Apache-2.0\n"
        "compatibility: Requires Python 3.14\n"
        "metadata:\n"
        "  author: example-org\n"
        "  version: '1.0'\n"
        "allowed-tools: Bash Read\n"
        "---\n\n"
        "Instructions here."
    )
    _write_skill_md(tmp_path / "full-skill", content)
    s = SkillLoader(tmp_path).load()[0]
    assert s.license == "Apache-2.0"
    assert s.compatibility == "Requires Python 3.14"
    assert s.metadata == {"author": "example-org", "version": "1.0"}
    assert s.allowed_tools == ["Bash", "Read"]


@pytest.mark.unit
def test_load_allowed_tools_space_separated(tmp_path):
    content = (
        "---\nname: my-skill\ndescription: A skill.\n"
        "allowed-tools: Bash(git:*) Bash(jq:*) Read\n---\n\nInstructions."
    )
    _write_skill_md(tmp_path / "my-skill", content)
    assert SkillLoader(tmp_path).load()[0].allowed_tools == [
        "Bash(git:*)",
        "Bash(jq:*)",
        "Read",
    ]


@pytest.mark.unit
def test_load_list_of_paths_merges_skills(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    _write_skill_md(dir_a / "alpha-skill", _minimal_skill_md(name="alpha-skill"))
    _write_skill_md(dir_b / "beta-skill", _minimal_skill_md(name="beta-skill"))
    skills = SkillLoader([dir_a, dir_b]).load()
    assert [s.name for s in skills] == ["alpha-skill", "beta-skill"]


@pytest.mark.unit
def test_load_accepts_a_direct_skill_folder(tmp_path):
    """A path that is itself a skill folder (SKILL.md inside) loads as that skill."""
    skill_dir = tmp_path / "greet"
    _write_skill_md(skill_dir, _minimal_skill_md(name="greet"))
    skills = SkillLoader(skill_dir).load()
    assert [s.name for s in skills] == ["greet"]
    assert skills[0].skill_dir == skill_dir.resolve()


@pytest.mark.unit
def test_load_list_of_direct_skill_folders_merges(tmp_path):
    _write_skill_md(tmp_path / "greet", _minimal_skill_md(name="greet"))
    _write_skill_md(tmp_path / "farewell", _minimal_skill_md(name="farewell"))
    skills = SkillLoader([tmp_path / "greet", tmp_path / "farewell"]).load()
    assert sorted(s.name for s in skills) == ["farewell", "greet"]


@pytest.mark.unit
def test_load_direct_skill_folder_that_fails_validation_is_empty(tmp_path):
    skill_dir = tmp_path / "actual-dir"
    _write_skill_md(
        skill_dir,
        "---\nname: different-name\ndescription: A skill.\n---\n\nInstructions.",
    )
    assert SkillLoader(skill_dir).load() == []


@pytest.mark.unit
def test_fixture_skill_loads_with_scripts():
    """End-to-end: load the csv-to-markdown fixture from disk and verify scripts are collected."""
    skills = SkillLoader(DATA_DIR).load()
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "csv-to-markdown"
    script_names = {p.name for p in skill.scripts}
    assert "csv_to_markdown.py" in script_names
    assert skill.skill_dir.is_absolute()
