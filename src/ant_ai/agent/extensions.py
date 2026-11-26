from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ant_ai.agent.planning import planning_kit
from ant_ai.hooks.layer import HookLayer
from ant_ai.hooks.protocol import AgentHook
from ant_ai.memory.protocol import Memory

if TYPE_CHECKING:
    from ant_ai.tools.tool import Tool


class AgentExtensions(BaseModel):
    """The hooks, long-term memory and plan of an agent."""

    hooks: list[AgentHook] = Field(
        default_factory=list,
        description="Lifecycle hooks for the agent.",
    )
    memory: Memory | None = Field(
        default=None,
        description="Long-term memory kept across conversations; its tools are "
        "added to the agent's (see Memory).",
    )
    planning: bool = Field(
        default=False,
        description="Let the agent keep a plan of its task, saved in "
        "`.plans/<agent>/<session_id>.md`.",
    )

    _hook_layer: HookLayer = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build_extensions(self, agent_name: str) -> list[Tool]:
        """Build the hook chain and return the tools memory and planning add."""
        tools: list[Tool] = [] if self.memory is None else [self.memory]
        hooks: list[AgentHook] = list(self.hooks)
        if self.planning:
            planning_tool, planning_hook = planning_kit(agent_name)
            tools.append(planning_tool)
            hooks.append(planning_hook)
        self._hook_layer = HookLayer(hooks=hooks)
        return tools
