---
title: Example
---

# Single-agent example: Code generation agent

This example builds a code-generation agent with file-system tools and runs it directly via `agent.stream()`.

## 1. Define tools

```python
# tools.py
from ant_ai import tool

@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    with open(path) as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Written to {path}"
```

## 2. Create the agent

```python
# agent.py
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat
from myapp.tools import read_file, write_file

agent = Agent(
    name="Developer",
    llm=LiteLLMChat("gpt-4o-mini"),
    system_prompt="You are a senior Python developer. Write clean, well-tested code.",
    description="A 10x Developer",
    tools=[read_file, write_file],
)
```

## 3. Run it

```python
# main.py
import asyncio
from ant_ai import Message, State, InvocationContext
from myapp.agent import agent


async def main():
    ctx = InvocationContext(session_id="demo-session")
    state = State()
    state.add_message(Message(role="user", content="Write a function that checks if a number is prime and save it to prime.py."))

    async for event in agent.stream(state, ctx=ctx):
        if event.content:
            print(f"[{event.kind}] {event.content}")


asyncio.run(main())
```

Example output:

```
[update] I'll write a prime-checking function and save it to prime.py.
[tool_calling] write_file
[tool_result] Written to prime.py
[final_answer] Done! The function has been saved to prime.py.
```
