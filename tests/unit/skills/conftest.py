from __future__ import annotations

from pathlib import Path

import pytest

from ant_ai.agent.agent import Agent
from ant_ai.llm.protocol import ChatLLM
from ant_ai.skills.protocol import AgentSkill

DATA_DIR: Path = Path(__file__).parent / "data" / "skills"


@pytest.fixture
def make_skill():
    """Factory fixture that returns an AgentSkill with sensible defaults."""

    def _factory(
        name: str = "my-skill",
        description: str = "Does something useful.",
        instructions: str = "Follow these steps carefully.",
    ) -> AgentSkill:
        return AgentSkill(
            name=name,
            description=description,
            instructions=instructions,
            skill_dir=Path("/tmp") / name,
        )

    return _factory


@pytest.fixture
def make_agent():
    """Factory fixture that creates an Agent with given skills and LLM."""

    def _factory(skills: list[AgentSkill], llm: ChatLLM) -> Agent:
        return Agent(
            name="test",
            system_prompt="You are a helpful agent.",
            llm=llm,
            skills=skills,
        )

    return _factory
