from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AgentSkill(BaseModel):
    """
    Represents a parsed Agent Skill conforming to the agentskills.io specification.

    See https://agentskills.io/specification for the full format spec.
    """

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9](-?[a-z0-9]+)*$",
        description=(
            "Skill identifier. Lowercase letters, numbers, and hyphens only. "
            "Must not start or end with a hyphen, and must not contain consecutive hyphens."
        ),
    )
    description: str = Field(
        min_length=1,
        max_length=1024,
        description="One-line description of what the skill does and when to use it.",
    )
    instructions: str = Field(
        description="Full Markdown body of SKILL.md, injected into context on activation.",
    )
    skill_dir: Path = Field(
        description="Resolved absolute path to the skill folder on disk.",
    )
    scripts: list[Path] = Field(
        default_factory=list,
        description=(
            "Files found in scripts/. Informational only — not auto-registered as tools. "
            "SKILL.md instructions reference these by path for the agent to invoke."
        ),
    )
    license: str | None = Field(
        default=None,
        description="License name or reference to a bundled license file.",
    )
    compatibility: str | None = Field(
        default=None,
        max_length=500,
        description="Environment requirements (intended product, system packages, etc.).",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata from SKILL.md frontmatter.",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Pre-approved tools the skill may use (from the allowed-tools field).",
    )
