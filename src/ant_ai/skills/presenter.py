from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ant_ai.skills.protocol import AgentSkill

_HOW_TO_USE = """\
**How to Use Skills (Progressive Disclosure):**
1. Recognize when a skill applies: check if the user's task matches a skill's description
2. Read full instructions: use your file reading tool on the path shown above
3. Follow the skill's instructions: SKILL.md contains step-by-step workflows and examples
4. Access supporting files via absolute paths"""


class SkillPresenter(Protocol):
    """Protocol for formatting skills into an agent's system prompt."""

    def system_prompt(self, skills: list[AgentSkill]) -> str: ...


class MarkdownSkillPresenter:
    """Injects skills as a Markdown block in the system prompt.

    The agent activates a skill by reading its SKILL.md via its native file tool — no custom activation tool is registered.
    """

    def system_prompt(self, skills: list[AgentSkill]) -> str:
        if not skills:
            return ""
        lines: list[str] = [
            "## Skills System",
            "",
            "You have access to a skills library that provides specialized capabilities and domain knowledge.",
            "",
            "**Available Skills:**",
            "",
        ]
        for skill in skills:
            skill_md: Path = skill.skill_dir / "SKILL.md"
            lines.append(f"- **{skill.name}**: {skill.description}")
            if skill.allowed_tools:
                lines.append(f"  -> Allowed tools: {', '.join(skill.allowed_tools)}")
            lines.append(
                f"  -> Read `{skill_md}` for full instructions (pass `limit=1000`)"
            )
        lines += ["", _HOW_TO_USE]
        return "\n".join(lines)
