---
title: Memory
---

# Agent memory

The `memory` field on [`Agent`][ant_ai.agent.agent.Agent] gives the agent a long-term memory that persists knowledge across conversations. The built-in backend, [`Mem0Memory`](#built-in-backend-mem0), keeps facts in the [mem0](https://app.mem0.ai/) cloud service, searched by meaning and kept apart per user.

## How it works

A memory is a tool namespace: setting `memory` on `Agent` registers the backend's tools, which the LLM calls like any other tool. They are named after the backend class, e.g. `Mem0Memory_search`/`Mem0Memory_add`.

The model decides when to call them — memory is not retrieved automatically before every turn or written back automatically at the end. The tool descriptions nudge the model to recall before answering questions that might depend on stored context, and to save facts as soon as it learns them. Recall therefore depends on the model choosing to call the tools — for applications that need exhaustive capture, reinforce this in the agent's `system_prompt` (see the [full example](#full-example) below), since a weaker or less instruction-following model may not always call them.

The tools are absent when `memory` is `None`, so agents without memory are unaffected.

## Built-in backend: mem0

[`Mem0Memory`][ant_ai.memory.backends.mem0.Mem0Memory] wraps the [mem0](https://app.mem0.ai/) cloud client. It requires the `mem0` extra, and a `MEM0_API_KEY` environment variable or an explicit `api_key` argument.

```python
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.memory.backends.mem0 import Mem0Memory

agent = Agent(
    name="Assistant",
    llm=LiteLLMChat("gpt-5-mini"),
    system_prompt="You are a helpful assistant.",
    memory=Mem0Memory(),  # picks up MEM0_API_KEY from the environment
)
```

The model gets two tools:

- `Mem0Memory_search(query)` — search the memory for facts relevant to `query`, returning at most `top_k` of them (5 by default).
- `Mem0Memory_add(facts)` — persist one or more durable facts for future conversations.

### Scoping memories to a user

Pass `user_id` through [`InvocationContext`][ant_ai.core.types.InvocationContext] to keep each user's memories isolated:

```python
from ant_ai import InvocationContext, Message, State

ctx = InvocationContext(session_id="session-abc", user_id="alice")
state = State()
state.add_message(Message(role="user", content="My favourite language is Python."))

async for event in agent.stream(state, ctx=ctx):
    ...
```

On the next invocation with the same `user_id`, the agent can recall that preference (once the model chooses to call `Mem0Memory_search`).

When `user_id` is absent `Mem0Memory` falls back to `session_id`, which gives run-scoped memory (useful for long single-session tasks).

`Mem0Memory` requires scoping information: calling it with no `ctx` raises `ValueError`, rather than silently pooling memory across every user and session. If a tool call triggers this, the LLM receives a clean `"ERROR: Mem0Memory requires scoping..."` tool result. `ctx` doesn't require running behind a server — any stable identifier works, even in a local script (`InvocationContext(session_id="local-script")`).

### A2A: passing `user_id` from metadata

When running behind an [A2A](../multi-agent/index.md) server, pass `user_id` in the task metadata:

```python
task_client.send_task(
    message="...",
    metadata={"user_id": "alice"},
)
```

The [`A2AExecutor`][ant_ai.a2a.executor.A2AExecutor] forwards it into `InvocationContext` automatically.

### Full example

```python
import asyncio
from ant_ai import Agent, Message, State, InvocationContext
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.memory.backends.mem0 import Mem0Memory
from ant_ai.core import FinalAnswerEvent

agent = Agent(
    name="Assistant",
    llm=LiteLLMChat("gpt-5-mini"),
    system_prompt=(
        "You are a helpful assistant with long-term memory, exposed via the "
        "Mem0Memory_search and Mem0Memory_add tools. Save durable facts "
        "about the user as soon as you learn them, and search memory before "
        "answering questions that might depend on something you already "
        "know about the user."
    ),
    memory=Mem0Memory(),
)


async def chat(user_id: str, text: str) -> str:
    ctx = InvocationContext(session_id="s1", user_id=user_id)
    state = State()
    state.add_message(Message(role="user", content=text))
    result = ""
    async for event in agent.stream(state, ctx=ctx):
        if isinstance(event, FinalAnswerEvent):
            result = event.content
    return result


async def main():
    # First session — agent learns the preference
    await chat("alice", "I am from Italy")

    # mem0 cloud indexing is async; wait for the memory to become searchable.
    await asyncio.sleep(10)

    # Second session — agent remembers without being told again
    reply = await chat("alice", "What's the capital of my country?")
    print(reply)


asyncio.run(main())
```

## Writing your own backend

Subclass [`Memory`][ant_ai.memory.protocol.Memory] and define the tools that fit your storage as public methods. Their docstrings are the tool descriptions the model reads, and their parameters, except `ctx`, are the tool's arguments:

```python
from ant_ai.core.types import InvocationContext
from ant_ai.memory import Memory


class NotesMemory(Memory):
    notes: dict[str, list[str]] = {}

    def recall(self, ctx: InvocationContext | None = None) -> list[str]:
        """Everything you saved about this user in earlier conversations."""
        return self.notes.get(ctx.user_id if ctx else "", [])

    def save(self, note: str, ctx: InvocationContext | None = None) -> str:
        """Save one durable fact about the user for future conversations."""
        self.notes.setdefault(ctx.user_id if ctx else "", []).append(note)
        return "Saved."
```

A method that declares `ctx` receives the current [`InvocationContext`][ant_ai.core.types.InvocationContext], which `ToolStep` injects when the model calls the tool; it is never shown to the model. Use `ctx.user_id` to keep each user's memories apart. Methods may be sync or async, and private methods (`_name`) are not exposed.
