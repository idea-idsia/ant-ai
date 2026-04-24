---
title: Single-agent
---

# Single-agent setup

A single `ant-ai` agent pairs an LLM with tools and, optionally, a Workflow that controls how it moves through a task.

## Creating an agent

Instantiate [`Agent`][ant_ai.agent.agent.Agent] with a name, system prompt, LLM, and an optional list of tools.

```python
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat

agent = Agent(
    name="Developer",
    system_prompt="You are a senior Python developer. Write clean, well-tested code.",
    llm=LiteLLMChat("gpt-4o-mini"),
    description="Writes Python code on demand.",
)
```

`LiteLLMChat` accepts any model string supported by [LiteLLM](https://docs.litellm.ai/docs/providers) (e.g. `"gpt-4o"`, `"claude-opus-4-6"`, `"gemini/gemini-2.0-flash"`).
It also supports vLLM by setting the venv file content:

```
LITELLM_API_KEY=dev-local-key
LITELLM_API_BASE=http://localhost:8000/v1
```

## Defining tools

### Function decorator

The `@tool` decorator turns a plain function into a tool. Type annotations are used to generate the JSON schema the LLM receives.

```python
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

Pass tools to the agent at construction time or add them later:

```python
agent = Agent(..., tools=[read_file, write_file])

# or dynamically
agent.add_tool(read_file)
```

### Class-based tools (namespaces)

Group related tools under a single class. Each public method becomes a separate tool exposed to the LLM as `ClassName.method_name`.

```python
from ant_ai import Tool

class FilesystemTools(Tool):
    """Tools for reading and writing files."""

    def read_file(self, path: str) -> str:
        """Read the contents of a file."""
        with open(path) as f:
            return f.read()

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        with open(path, "w") as f:
            f.write(content)
        return f"Written to {path}"

agent = Agent(..., tools=[FilesystemTools()])
```

### MCP tools

Connect to any [Model Context Protocol](https://modelcontextprotocol.io/) server and import its tools directly:

```python
from ant_ai.tools import mcp_tools_from_url

tools = await mcp_tools_from_url("http://localhost:8000/mcp", namespace="remote")
agent = Agent(..., tools=tools)
```

## Streaming a response

[`Agent.stream()`][ant_ai.agent.agent.Agent.stream] drives the agent until it produces a final answer, yielding [`Event`][ant_ai.core.events.Event] objects at each step — LLM output, tool calls, tool results, and completion.

```python
import asyncio
from ant_ai import Message, State, InvocationContext
from ant_ai.core import FinalAnswerEvent, UpdateEvent

async def main():
    ctx = InvocationContext(session_id="my-session")
    state = State()
    state.add_message(Message(role="user", content="Write a hello-world function."))

    async for event in agent.stream(state, ctx=ctx):
        if isinstance(event, UpdateEvent):
            print("update:", event.content)
        elif isinstance(event, FinalAnswerEvent):
            print("final:", event.content)

asyncio.run(main())
```

Key event classes (all in [`ant_ai.core.events`][ant_ai.core.events]):

| Class                                                                          | Meaning                         |
| ------------------------------------------------------------------------------ | ------------------------------- |
| [`UpdateEvent`][ant_ai.core.events.UpdateEvent]                              | Intermediate LLM output         |
| [`ToolCallingEvent`][ant_ai.core.events.ToolCallingEvent]                    | Agent is about to call a tool   |
| [`ToolResultEvent`][ant_ai.core.events.ToolResultEvent]                      | Tool returned a result          |
| [`FinalAnswerEvent`][ant_ai.core.events.FinalAnswerEvent]                    | Agent produced its final answer |
| [`MaxStepsReachedEvent`][ant_ai.core.events.MaxStepsReachedEvent]            | Loop hit the step limit         |

## Adding a Workflow

A [`Workflow`][ant_ai.workflow.workflow.Workflow] is a directed graph of _nodes_. Each node is an async generator that receives the agent, the current [`State`][ant_ai.core.types.State], and an [`InvocationContext`][ant_ai.core.types.InvocationContext], and yields events plus an updated state.

### Defining nodes

```python
from collections.abc import AsyncGenerator
from ant_ai import Agent, Message, InvocationContext, State
from ant_ai.workflow import NodeYield

async def generate(
    agent: Agent, state: State, ctx: InvocationContext
) -> AsyncGenerator[NodeYield]:
    state.add_message(Message(role="user", content="Write a Python function that reverses a string."))

    async for event in agent.stream(state, ctx=ctx):
        yield event   # forward events to the caller

    yield state       # always yield the updated state at the end
```

### Routing between nodes

A _conditional edge_ is an async function that inspects the state and returns the name of the next node (or `END`):

```python
from typing import Literal
from ant_ai.workflow import END

async def should_revise(
    agent: Agent, state: State, ctx: InvocationContext
) -> Literal["revise", "END"]:
    if "TODO" in (state.last_message.content or ""):
        return "revise"
    return END
```

### Assembling the workflow

```python
from ant_ai.workflow import Workflow, START, END

workflow = Workflow()
workflow.add_node("generate", generate)
workflow.add_node("validate", validate)
workflow.add_node("revise", revise)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")
workflow.add_conditional_edge("validate", should_revise)
workflow.add_edge("revise", "validate")   # retry loop
```

The workflow enforces that each non-conditional node has exactly one outgoing edge. Conditional edges replace static edges on the same source node.

### Running the workflow

```python
async for event in workflow.stream(agent, ctx=ctx, state=initial_state):
    print(event)

# or get the final state only
final_state = await workflow.ainvoke(agent, ctx=ctx, state=initial_state)
```

### State

[`State`][ant_ai.core.types.State] carries both the conversation history (`messages`) and an `artefacts` list for passing structured data between nodes. Subclass it to add domain-specific fields:

```python
state = State()
state.add_message(Message(role="user", content="..."))
state.artefacts.append({"generated_code": "def foo(): ..."})
```

## Full example

See [Example](example.md) for a complete single-agent setup: a code-generation agent with file-system tools running via `agent.stream()`.
