---
title: Memory
---

# Agent memory

The `memory` field on [`Agent`][ant_ai.agent.agent.Agent] gives the agent a long-term memory that persists knowledge across conversations. Two backends are built in:

- [`Mem0Memory`](#built-in-backend-mem0) keeps facts in the [mem0](https://app.mem0.ai/) cloud service, searched by meaning and kept apart per user.
- [`FileMemory`](#file-based-memory-memorymd) keeps the whole memory in one Markdown file, with no service or API key.

## How it works

A memory is a tool namespace: setting `memory` on `Agent` registers the backend's tools, which the LLM calls like any other tool. They are named after the backend class, e.g. `Mem0Memory_search`/`Mem0Memory_add` or `FileMemory_read`/`FileMemory_write`.

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

!!! warning "The client chooses `user_id`"
    The metadata comes from the client, so a client can send another user's `user_id` and reach their memories. When users must not see each other's memories, authenticate requests and set `user_id` from the authenticated identity, for example in a [custom `from_metadata`](index.md#custom-invocation-context), rather than trusting the value the client sends.

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

## File-based memory: `MEMORY.md`

[`FileMemory`][ant_ai.memory.backends.file.FileMemory] keeps the agent's long-term memory in one Markdown file. It needs no service and no API key. You can read and edit the file by hand, keep it in git, and use it offline and in tests.

```python
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat
from ant_ai.memory.backends.file import FileMemory

agent = Agent(
    name="Assistant",
    llm=LiteLLMChat("gpt-5-mini"),
    system_prompt=(
        "You are a helpful assistant with a long-term memory. Read it with "
        "FileMemory_read at the start of each conversation, and save "
        "durable facts about the user with FileMemory_write as soon as "
        "you learn them."
    ),
    memory=FileMemory(path="./agent_state/MEMORY.md"),
)
```

The memory is meant to stay short, so the model reads and writes it whole, with two tools:

| Tool | What it does |
|------|--------------|
| `FileMemory_read()` | Returns the whole file, or `No memories saved yet.` when there is none. |
| `FileMemory_write(content)` | Replaces the whole file. To save a fact the model writes the memory with the fact added; to forget one, without it. |

Because a write replaces everything, it is refused unless the conversation has read the memory as it is now. A model cannot erase a memory it has not seen, or undo a change that another conversation, or someone editing the file, made since it read it. It is told to read again and redo its change. A write longer than `max_chars` (10,000 characters by default, `None` for no limit) is refused too, asking the model to shorten the memory.

The file and its parent directories are created on the first write. Each write goes to a temporary file, flushed to disk, that is then renamed over `MEMORY.md`, so a crash, even of the machine, never leaves the file half-written. A new file is readable by its owner only, and an existing one keeps its permissions. Writes take turns, across threads and, on POSIX, across processes sharing the file, by locking its directory, so two conversations that read the same memory cannot both replace it. The lock does not stop someone editing the file by hand at the very moment of a write. Failures, such as a missing permission, raise [`ToolError`][ant_ai.tools.tool.ToolError], and the model gets them as a recoverable `ERROR: ...` result.

!!! note "One file, shared by everyone"
    Unlike `Mem0Memory`, `FileMemory` does not scope memories by `user_id`: whatever `ctx` says, every conversation and user of the memory reads and writes the same file. When users' memories must stay apart, as behind an A2A server, give each user their own agent and `path`.

## Writing your own backend

Subclass [`Memory`][ant_ai.memory.protocol.Memory] and define the tools that fit your storage as public methods. Their docstrings are the tool descriptions the model reads, and their parameters, except `ctx`, are the tool's arguments:

```python
from ant_ai.core.types import InvocationContext
from ant_ai.memory import Memory


class NotesMemory(Memory):
    notes: dict[str, list[str]] = {}

    def recall(self, ctx: InvocationContext | None = None) -> list[str]:
        """Everything you saved about this user in earlier conversations."""
        return self.notes.get(_owner(ctx), [])

    def save(self, note: str, ctx: InvocationContext | None = None) -> str:
        """Save one durable fact about the user for future conversations."""
        self.notes.setdefault(_owner(ctx), []).append(note)
        return "Saved."


def _owner(ctx: InvocationContext | None) -> str:
    # Without a user, keep the notes to the session, so that sessions without a
    # user_id do not share them.
    if ctx is None:
        raise ValueError("NotesMemory needs an InvocationContext")
    return f"user:{ctx.user_id}" if ctx.user_id else f"session:{ctx.session_id}"
```

A method that declares `ctx` receives the current [`InvocationContext`][ant_ai.core.types.InvocationContext], which `ToolStep` injects when the model calls the tool; it is never shown to the model. Use `ctx.user_id` to keep each user's memories apart. Methods may be sync or async, and private methods (`_name`) are not exposed.
