from __future__ import annotations

from ant_ai.hooks.builtins.planning import PlanningHook
from ant_ai.tools.builtins.planning_tool import PlanningTool, PlanStore


def planning_kit(agent_name: str) -> tuple[PlanningTool, PlanningHook]:
    """The tool and hook that let the agent `agent_name` keep a plan in one store."""
    store = PlanStore()
    return (
        PlanningTool(agent=agent_name, store=store),
        PlanningHook(agent=agent_name, store=store),
    )
