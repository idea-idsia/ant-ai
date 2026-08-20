---
title: Memory
---

# Agent memory

The `memory` field on [`Agent`][ant_ai.agent.agent.Agent] connects a pluggable long-term memory backend that persists knowledge across conversations.

## How it works

Setting `memory` on `Agent` registers two extra tools the LLM can call, just like any other tool — named after the backend class, e.g. `Mem0Memory_search`/`Mem0Memory_add` for the built-in mem0 backend:

- `<Backend>_search(query)` — search the memory store for facts relevant to `query`.
- `<Backend>_add(facts)` — persist one or more durable facts for future conversations.

The model decides when to call them — memory is no longer retrieved automatically before every turn or written back automatically at the end. The tool descriptions explicitly nudge the model to search before answering questions that might depend on stored context, and to save facts proactively as soon as it learns them. This means recall now depends on the model choosing to call `<Backend>_add`/`<Backend>_search` — for applications that need exhaustive capture, reinforce this in the agent's `system_prompt` (see the [full example](#full-example) below), since a weaker or less instruction-following model may not always call it.

Both tools are absent when `memory` is `None`, so existing agents without memory are unaffected.

## The `Memory` and `MemoryTool` classes

[`Memory`][ant_ai.memory.protocol.Memory] is the storage interface — implement two async methods to connect any backend:

```python
from ant_ai.memory import Memory
from ant_ai.core.message import Message


class MyMemory(Memory):
    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs) -> list[Message]:
        # Return relevant memories as system messages
        ...

    async def update(self, messages: list[Message], **kwargs) -> None:
        # Persist messages for future retrieval
        ...
```

[`MemoryTool`][ant_ai.tools.builtins.memory_tool.MemoryTool] extends `Memory` with the `Tool` base class, adding the LLM-facing `search`/`add` methods on top of `retrieve`/`update` — matching the vocabulary mem0's own client uses (`search`/`add`). **A backend must extend `MemoryTool` (not just `Memory`) to be usable as `Agent(memory=...)`** — `BaseAgent.memory` is typed `MemoryTool | None`, since only `Tool`-capable instances can be registered:

```python
from ant_ai.tools.builtins.memory_tool import MemoryTool
from ant_ai.core.message import Message


class MyMemory(MemoryTool):
    async def retrieve(
        self, query: str, *, top_k: int = 5, **kwargs
    ) -> list[Message]: ...

    async def update(self, messages: list[Message], **kwargs) -> None: ...

    # search/add are inherited automatically — no need to redefine them.
```

Both `retrieve` and `update` receive `ctx: InvocationContext` via `**kwargs`. Use `ctx.user_id` for cross-session scoping.

## Built-in backend: mem0

[`Mem0Memory`][ant_ai.memory.backends.mem0.Mem0Memory] wraps the [mem0](https://app.mem0.ai/) cloud client (`Mem0Memory(MemoryTool)`). It requires a `MEM0_API_KEY` environment variable or an explicit `api_key` argument.

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

On the next invocation with the same `user_id`, the agent can recall that preference (once the model chooses to call `<Backend>_search`).

When `user_id` is absent the backend falls back to `session_id`, which gives run-scoped memory (useful for long single-session tasks).

`Mem0Memory` requires scoping information: calling it with no `ctx` and no explicit `user_id`/`run_id`/`agent_id`/`app_id` raises `ValueError`, rather than silently pooling memory across every user and session. If a tool call triggers this, the LLM receives a clean `"ERROR: Mem0Memory requires scoping..."` tool result. `ctx` doesn't require running behind a server — any stable identifier works, even in a local script (`InvocationContext(session_id="local-script")`).

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
