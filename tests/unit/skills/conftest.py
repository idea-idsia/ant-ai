from __future__ import annotations

from pathlib import Path

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.llm.protocol import ChatLLM

DATA_DIR: Path = Path(__file__).parent / "data" / "skills"


@pytest.fixture
def make_skill(tmp_path):
    """Factory fixture that writes a skill to a real directory and returns the skills dir."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    def _factory(
        name: str = "my-skill",
        description: str = "Does something useful.",
        instructions: str = "Follow these steps carefully.",
        allowed_tools: list[str] | None = None,
    ) -> Path:
        skill_dir = skills_dir / name
        skill_dir.mkdir(exist_ok=True)
        frontmatter = f"name: {name}\ndescription: {description}"
        if allowed_tools:
            frontmatter += f"\nallowed-tools: {' '.join(allowed_tools)}"
        (skill_dir / "SKILL.md").write_text(
            f"---\n{frontmatter}\n---\n{instructions}\n"
        )
        return skills_dir

    return _factory


@pytest.fixture
def make_agent():
    """Factory fixture that creates an Agent with a given skills path and LLM."""

    def _factory(skills: Path | None, llm: ChatLLM) -> Agent:
        return Agent(
            name="test",
            system_prompt="You are a helpful agent.",
            llm=llm,
            skills=skills,
        )

    return _factory
