from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SkipValidation

from ant_ai.core.message import Message
from ant_ai.skills.loader import SkillLoader
from ant_ai.skills.presenter import MarkdownSkillPresenter, SkillPresenter

if TYPE_CHECKING:
    from ant_ai.skills.protocol import AgentSkill


class AgentSkills(BaseModel):
    """The skills of an agent, announced in its system prompt."""

    skills: Path | list[Path] | None = Field(
        default=None,
        description="Path (or list of paths) to skills directories (agentskills.io standard).",
    )
    skill_presenter: Annotated[SkillPresenter, SkipValidation] = Field(
        default_factory=MarkdownSkillPresenter,
        description="Controls how skills are injected into the system prompt.",
    )

    _skills: list[AgentSkill] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_skills(self) -> None:
        self._skills = (
            SkillLoader(self.skills).load() if self.skills is not None else []
        )

    def _system_message_with_skills(self, prompt: str) -> Message:
        """Return the system message of `prompt`, with skill discovery appended if present."""
        if not self._skills:
            return Message(role="system", content=prompt)
        skills_block: str = self.skill_presenter.system_prompt(self._skills)
        content: str = "\n\n".join([prompt, skills_block])
        return Message(role="system", content=content)
