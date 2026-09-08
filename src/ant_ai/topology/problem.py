"""What a check reports, and what is raised when checks fail.

Its own module, and a leaf one: `preflight` reports these about a pipeline and
`state` reports them about a rewrite, and the two must not have to import each
other in order to say the same kind of thing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["Problem", "TopologyConfigurationError"]


class Problem(BaseModel):
    """One thing wrong with a configuration or an edit, and what to do about it.

    `hint` is a separate field rather than a longer message because it is the
    half a caller acts on, and an exception that states a problem without
    stating the fix has only moved the confusion.
    """

    code: str = Field(description="Stable identifier, e.g. 'E001'.")
    level: Literal["error", "warning"]
    message: str = Field(description="What is wrong, in one sentence.")
    hint: str = Field(default="", description="How to fix it.")

    def render(self) -> str:
        return f"[{self.code}] {self.message}" + (
            f"\n  Fix: {self.hint}" if self.hint else ""
        )


class TopologyConfigurationError(ValueError):
    """Raised when a topology cannot possibly do what it was configured to do.

    Carries the problems rather than only their rendered text, so a caller
    building configurations programmatically — an ablation sweep — can branch on
    `code` instead of matching strings.
    """

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        body = "\n".join(p.render() for p in problems)
        super().__init__(
            f"This topology cannot run as configured:\n{body}\n\n"
            "Construct `Ensemble(...)` directly to bypass these checks."
        )
