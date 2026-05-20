from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from ant_ai.skills.protocol import AgentSkill

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


class SkillLoader:
    """
    Loads Agent Skills from one or more directories of skill folders.

    Each skill must be a sub-directory containing at least a ``SKILL.md`` file
    with YAML frontmatter providing ``name`` and ``description`` fields, followed
    by Markdown instructions as the body.

    Folders that are missing ``SKILL.md``, have invalid frontmatter, or fail
    spec validation are silently skipped.

    Args:
        skills_dir: Path (or list of paths) to directories that contain skill sub-folders.
    """

    def __init__(self, skills_dir: str | Path | list[str | Path]) -> None:
        if isinstance(skills_dir, list):
            self._skills_dirs: list[Path] = [Path(d).resolve() for d in skills_dir]
        else:
            self._skills_dirs = [Path(skills_dir).resolve()]

    def load(self) -> list[AgentSkill]:
        """
        Walk each skills directory and parse every valid skill folder.

        Returns:
            A list of `AgentSkill` instances across all directories, sorted by folder name within each directory.
        """
        skills: list[AgentSkill] = []
        for skills_dir in self._skills_dirs:
            skills.extend(self._load_dir(skills_dir))
        return skills

    def _load_dir(self, skills_dir: Path) -> list[AgentSkill]:
        skills: list[AgentSkill] = []
        if not skills_dir.is_dir():
            return skills

        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_md: Path = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            skill: AgentSkill | None = self._parse_skill(entry, skill_md)
            if skill is not None:
                skills.append(skill)

        return skills

    def _parse_skill(self, skill_dir: Path, skill_md: Path) -> AgentSkill | None:
        try:
            raw: str = skill_md.read_text(encoding="utf-8")
        except OSError:
            return None

        match: re.Match[str] | None = _FRONTMATTER_RE.match(raw)
        if not match:
            return None

        try:
            frontmatter: dict = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None

        name: str | None = frontmatter.get("name")
        description: str | None = frontmatter.get("description")
        if not name or not description:
            return None
        if str(name) != skill_dir.name:
            return None

        instructions: str = match.group(2).strip()
        scripts: list[Path] = self._collect_scripts(skill_dir)

        raw_allowed: str | None = frontmatter.get("allowed-tools", "")
        allowed_tools: list[str] = str(raw_allowed).split() if raw_allowed else []

        raw_metadata: dict | None = frontmatter.get("metadata") or {}
        metadata: dict[str, str] = (
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
        scripts_dir: Path = skill_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        return sorted(p for p in scripts_dir.iterdir() if p.is_file())
