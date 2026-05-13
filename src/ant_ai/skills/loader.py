from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from ant_ai.skills.protocol import AgentSkill
from ant_ai.tools.tool import Tool

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


class SkillLoader:
    """
    Loads Agent Skills from a directory of skill folders.

    Each skill must be a sub-directory containing at least a ``SKILL.md`` file
    with YAML frontmatter providing ``name`` and ``description`` fields, followed
    by Markdown instructions as the body.

    Folders that are missing ``SKILL.md``, have invalid frontmatter, or fail
    spec validation are silently skipped.

    Args:
        skills_dir: Path to the directory that contains skill sub-folders.
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self._skills_dir = Path(skills_dir).resolve()

    def load(self) -> list[AgentSkill]:
        """
        Walk the skills directory and parse each valid skill folder.

        Returns:
            A list of :class:`AgentSkill` instances, one per valid skill folder
            found. Results are sorted by folder name.
        """
        skills: list[AgentSkill] = []
        if not self._skills_dir.is_dir():
            return skills

        for entry in sorted(self._skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            skill = self._parse_skill(entry, skill_md)
            if skill is not None:
                skills.append(skill)

        return skills

    def _parse_skill(self, skill_dir: Path, skill_md: Path) -> AgentSkill | None:
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except OSError:
            return None

        match = _FRONTMATTER_RE.match(raw)
        if not match:
            return None

        try:
            frontmatter: dict = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None

        name = frontmatter.get("name")
        description = frontmatter.get("description")
        if not name or not description:
            return None

        instructions = match.group(2).strip()
        scripts = self._collect_scripts(skill_dir)

        raw_allowed = frontmatter.get("allowed-tools", "")
        allowed_tools = str(raw_allowed).split() if raw_allowed else []

        raw_metadata = frontmatter.get("metadata") or {}
        metadata = (
            {str(k): str(v) for k, v in raw_metadata.items()}
            if isinstance(raw_metadata, dict)
            else {}
        )

        try:
            return AgentSkill(
                name=str(name),
                description=str(description),
                instructions=instructions,
                skill_dir=skill_dir.resolve(),
                scripts=scripts,
                license=frontmatter.get("license"),
                compatibility=frontmatter.get("compatibility"),
                metadata=metadata,
                allowed_tools=allowed_tools,
            )
        except ValidationError:
            return None

    def _collect_scripts(self, skill_dir: Path) -> list[Path]:
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        return sorted(p for p in scripts_dir.iterdir() if p.is_file())


def make_use_skill_tool(skills: list[AgentSkill]) -> Tool:
    """
    Build the ``use_skill`` activation tool for a list of skills.

    When the LLM calls ``use_skill(skill_name="some-skill")``, the full
    ``SKILL.md`` instructions are returned as the tool result, injecting them
    into the conversation context (activation stage of progressive disclosure).

    Args:
        skills: The skills the agent knows about.

    Returns:
        A single :class:`~ant_ai.tools.tool.Tool` that accepts a ``skill_name``
        argument and returns the matching skill's instructions.
    """
    _index: dict[str, str] = {s.name: s.instructions for s in skills}
    known = ", ".join(f'"{n}"' for n in _index)
    description = (
        f"Load the full instructions for a skill by name. "
        f"Known skills: {known}. "
        "Call this before using any skill to receive its complete instructions."
    )

    def use_skill(skill_name: str) -> str:
        """Load the full instructions for a named skill.

        Args:
            skill_name: The exact name of the skill to activate.
        """
        instructions = _index.get(skill_name)
        if instructions is None:
            return f"Unknown skill '{skill_name}'. Available skills: {known}"
        return instructions

    return Tool._from_function(use_skill, name="use_skill", description=description)
