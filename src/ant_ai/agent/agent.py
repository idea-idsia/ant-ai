from __future__ import annotations

from ant_ai.agent.base import BaseAgent
from ant_ai.agent.loop.react import ReActLoop
from ant_ai.steps.llm_step import LLMStep
from ant_ai.steps.tool_step import ToolStep
from ant_ai.tools.tool import Tool


class Agent(BaseAgent):
    """
    Conversational ReAct agent: Reason → Act → Reason → … → Answer.
    """

    def _make_loop(self) -> ReActLoop:
        return ReActLoop(
            reason_step=LLMStep(
                llm=self.llm,
                system_message=self.system_message,
                serialized_tools=self._registry.to_serialized(),
            ),
            act_step=ToolStep(registry=self._registry)
            if self._registry.tools
            else None,
            hooks=self._hook_layer,
            max_retries=self.max_retries,
        )

    def add_tool(self, tool: Tool) -> None:
        """Register a tool at runtime and hot-reload the loop's step."""
        self._registry.register(tool)
        self._loop.register_tool(self._registry)
