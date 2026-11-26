from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ant_ai.core.types import InvocationContext
from ant_ai.tools.tool import Tool, ToolError, tool

if TYPE_CHECKING:
    from ant_ai.agent.base import BaseAgent

_DEFAULT_PROMPT = (
    "You are a subagent: another agent gave you one task. Do it with the tools "
    "you have, then answer with everything the other agent needs, since your "
    "answer is all it will see of your work. Be concise and do not ask questions."
)

_DEFAULT_DESCRIPTION = (
    "Create a subagent for one self-contained part of the work, such as "
    "exploring a part of the code base, and get back its answer. It sees only "
    "the task you write, so include every detail it needs, and it can use only "
    "the tools you list, chosen among your own. Its steps stay out of this "
    "conversation."
)


class SubagentSpec(BaseModel):
    name: str = Field(description="A short name for the subagent, e.g. `explorer`.")
    task: str = Field(
        description="The complete task. The subagent sees nothing else of the "
        "conversation, so include every detail it needs."
    )
    tools: list[str] = Field(
        description="Names of your own tools the subagent may use; it gets no other."
    )


class SubagentTool(Tool):
    """
    Lets an agent create subagents while it works.

    Put it in the `tools` of an agent. The model calls it with a name, a task
    and the names of the tools the subagent may use, chosen among the agent's
    own: the subagent runs on the same LLM, with `system_prompt`, and its
    final answer is the result of the call, as for `BaseAgent.as_tool()`. A
    subagent cannot create subagents of its own nor keep a plan.

    Example:

        lead = Agent(name="lead", system_prompt="...", llm=llm,
                     tools=[FilesystemTool(), ShellTool(), SubagentTool()])
    """

    name: str | None = "subagent"
    description: str | None = _DEFAULT_DESCRIPTION
    system_prompt: str = Field(
        default=_DEFAULT_PROMPT, description="System prompt of every subagent."
    )
    max_steps: int = Field(default=10, ge=1, description="Step budget of each run.")

    def _bind(self, parent: BaseAgent) -> Tool:
        """Return the tool the model calls, creating subagents of `parent`."""

        async def run(
            name: str,
            task: str,
            tools: list[str],
            ctx: InvocationContext | None = None,
        ) -> str:
            from ant_ai.agent.agent import Agent

            own: dict[str, Tool] = parent._delegable_tools()
            for tool_name in tools:
                if tool_name not in own:
                    choices: str = ", ".join(own) or "no tool"
                    raise ToolError(
                        f"unknown tool '{tool_name}': choose from {choices}"
                    )
            subagent = Agent(
                name=name,
                description=f"Subagent of {parent.name}.",
                system_prompt=self.system_prompt,
                llm=parent.llm,
                tools=[own[tool_name] for tool_name in dict.fromkeys(tools)],
            )
            return await subagent.as_tool(
                name="subagent", max_steps=self.max_steps
            ).ainvoke(message=task, ctx=ctx)

        return tool(
            name=self.name,
            description=self.description,
            args_model=SubagentSpec,
        )(run)
