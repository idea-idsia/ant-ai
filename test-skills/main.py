# main.py
import asyncio
import subprocess

from ant_ai import Agent, InvocationContext, Message, State, tool
from ant_ai.core import FinalAnswerEvent, ToolCallingEvent
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.skills.loader import SkillLoader


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr


skills = SkillLoader(".agents/skills").load()

agent = Agent(
    name="Assistant",
    system_prompt="You are a helpful assistant.",
    llm=LiteLLMChat("gpt-5-nano"),
    tools=[run_command],
    skills=skills,
)


async def main():
    ctx = InvocationContext(session_id="skills-demo")
    state = State()
    state.add_message(
        Message(role="user", content="Is there a skill for ggplot development?")
    )

    async for event in agent.stream(state, ctx=ctx):
        if isinstance(event, ToolCallingEvent):
            for tc in event.message.tool_calls:
                print(f"[tool] {tc.function.name} {tc.function.arguments}")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\n{event.content}")


asyncio.run(main())
