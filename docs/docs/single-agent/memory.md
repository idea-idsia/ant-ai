---
title: Memory
---

# Agent memory

The `memory` field on [`Agent`][ant_ai.agent.agent.Agent] connects a pluggable long-term memory backend that persists knowledge across conversations.

## How it works

On every call to `agent.stream()`:

1. **Retrieve** — before the first LLM step the agent queries the memory store with the latest user message and prepends any matching memories as `system` messages.
2. **Consolidate** — once a final answer is produced (or the step limit is reached) the user message and assistant reply are written back to the store.

Both steps are no-ops when `memory` is `None`, so existing agents are unaffected.

## The `Memory` interface

[`Memory`][ant_ai.memory.protocol.Memory] is the abstract base class. Implement two async methods to connect any backend:

```python
from ant_ai.memory import Memory
from ant_ai.core.message import Message
from ant_ai.core.types import InvocationContext

class MyMemory(Memory):
    async def retrieve(
        self, query: str, *, top_k: int = 5, **kwargs
    ) -> list[Message]:
        # Return relevant memories as system messages
        ...

    async def update(self, messages: list[Message], **kwargs) -> None:
        # Persist messages for future retrieval
        ...
```

Both methods receive `ctx: InvocationContext` via `**kwargs`. Use `ctx.user_id` for cross-session scoping.

## Built-in backend: mem0

[`Mem0Memory`][ant_ai.memory.backends.mem0.Mem0Memory] wraps the [mem0](https://app.mem0.ai/) cloud client. It requires a `MEM0_API_KEY` environment variable or an explicit `api_key` argument.

```python
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.memory.backends.mem0 import Mem0Memory

agent = Agent(
    name="Assistant",
    llm=LiteLLMChat("gpt-5-mini"),
    system_prompt="You are a helpful assistant.",
    memory=Mem0Memory(),           # picks up MEM0_API_KEY from the environment
)
```

## Scoping memories to a user

Pass `user_id` through [`InvocationContext`][ant_ai.core.types.InvocationContext] to keep each user's memories isolated:

```python
from ant_ai import InvocationContext, Message, State

ctx = InvocationContext(session_id="session-abc", user_id="alice")
state = State()
state.add_message(Message(role="user", content="My favourite language is Python."))

async for event in agent.stream(state, ctx=ctx):
    ...
```

On the next invocation with the same `user_id`, the agent will recall that preference automatically.

When `user_id` is absent the backend falls back to `session_id`, which gives run-scoped memory (useful for long single-session tasks).

## A2A: passing `user_id` from metadata

When running behind an [A2A](../multi-agent/index.md) server, pass `user_id` in the task metadata:

```python
task_client.send_task(
    message="...",
    metadata={"user_id": "alice"},
)
```

The [`A2AExecutor`][ant_ai.a2a.executor.A2AExecutor] forwards it into `InvocationContext` automatically.

## Full example

```python
import asyncio
from ant_ai import Agent, Message, State, InvocationContext
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.memory.backends.mem0 import Mem0Memory
from ant_ai.core import FinalAnswerEvent

agent = Agent(
    name="Assistant",
    llm=LiteLLMChat("gpt-5-mini"),
    system_prompt="You are a helpful assistant with long-term memory.",
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
