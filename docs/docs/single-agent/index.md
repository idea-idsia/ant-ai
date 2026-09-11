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
To point it at your own endpoint (vLLM, a proxy, …) pass the credential and URL directly, or set them in the environment:

```python
llm = LiteLLMChat(
    "gpt-4o-mini",
    api_key=os.environ["MY_DEPLOYMENT_API_KEY"],  # keep the secret under your own name
    api_base="http://localhost:8000/v1",
)
```

```
LITELLM_API_KEY=dev-local-key        # fallback when api_key is not given
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

### Reporting failures

Raise to report a failure; don't return an error string. Any exception a tool raises becomes an `ERROR: ...` result the model can recover from, and the call is flagged `is_error=True` on the [`ToolResultEvent`][ant_ai.core.events.ToolResultEvent] and the [`ToolCallResultMessage`][ant_ai.core.message.ToolCallResultMessage] — the same flag Anthropic's `tool_result` and MCP's `CallToolResult` carry. A tool that returns is a success: the framework doesn't inspect the content, so an error *string* would be counted as one.

Use [`ToolError`][ant_ai.tools.tool.ToolError] for expected failures whose message is written for the model; let anything else propagate as-is:

```python
from ant_ai import tool, ToolError


@tool
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    if not Path(path).is_file():
        raise ToolError(f"file not found: {path}")
    return Path(path).read_text()
```

MCP tools follow the same contract: a server answering with `isError` raises `ToolError` on this side. The flag is carried through events, A2A history and the agent's state, but is not sent to the model.

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

### Receiving the invocation context (`ctx`)

Any tool — `@tool`-decorated function or namespace method — can declare a `ctx: InvocationContext` parameter to receive the real [`InvocationContext`][ant_ai.core.types.InvocationContext] passed to `agent.stream(...)`/`agent.ainvoke(...)`. It's injected automatically at call time and is hidden from the LLM-facing JSON schema — the model never sees or supplies it.

```python
from ant_ai import tool
from ant_ai.core.types import InvocationContext


@tool
def whoami(ctx: InvocationContext) -> str:
    """Return the current user's id."""
    return ctx.user_id or "anonymous"
```

Use this for anything scoped to the caller rather than the conversation — per-user data lookups, multi-tenant isolation, audit logging. [`MemoryTool`][ant_ai.tools.builtins.memory_tool.MemoryTool] (see [Agent memory](memory.md)) is a built-in example: its `search`/`add` tool methods take `ctx` this way to scope memories per user.

### Custom invocation context

Subclass `InvocationContext` to carry your own request-scoped fields — a tenant, a feature flag, tags for your tracing backend — through the whole run. Tools receive the subclass instance, and the A2A and ACP entry points build it for you from the incoming request:

```python
from typing import Any

from ant_ai.a2a import A2AServer
from ant_ai.core.types import InvocationContext


class MyContext(InvocationContext):
    tenant: str = ""
    tags: list[str] | None = None

    def trace_attributes(self) -> dict[str, Any]:
        # Bound to every trace event for the run; the Langfuse sink forwards `tags`.
        return {**super().trace_attributes(), "tags": self.tags}


@tool
def whoami(ctx: MyContext) -> str:
    return f"{ctx.user_id} @ {ctx.tenant}"


server = A2AServer(
    agent=agent, workflow=workflow, agent_card=card, context_class=MyContext
)
```

Two hooks control how the subclass behaves:

- `from_metadata(session_id=..., metadata=...)` builds the context from the request metadata (the A2A message `metadata`, i.e. `request_metadata` on [`A2AClient.send_message`][ant_ai.a2a.client.A2AClient.send_message]). Fields are filled **by name** — a `tenant` key fills `tenant` — and unknown keys are ignored. Override it to map a different wire shape onto your fields.
- `trace_attributes()` returns the fields bound to the run's trace and sent with `workflow.start`. The default is `session_id` and `user_id`. Extend it to surface your own — and keep secrets out, since these reach whatever observability backend is configured.
- `outbound_metadata()` is what the context forwards when an agent calls *another* agent through [`A2AAgentTool`][ant_ai.a2a.agent.A2AAgentTool] — every set field except `session_id` (it travels as the A2A `context_id`) and the per-callee `llm_settings`/`workflow_settings`. Whether a remote agent *receives* it is decided per agent by `A2AConfig.trusted`: a trusted agent gets the trace context and this metadata; mark third-party agents `trusted=False` and they get the message and the session id only. The callee's `from_metadata` rebuilds it.

When calling the agent directly, construct the subclass yourself: `agent.ainvoke(..., ctx=MyContext(session_id="s1", tenant="acme"))`.

### What the caller sees when a run fails

A failure inside an A2A run reaches the caller as a bare `InternalError` — the exception text is **never** forwarded, since it can carry prompt content or internal detail and the caller may not be the operator. The one exception is an A2A error you raise yourself: any `a2a.utils.errors.A2AError` (`InternalError("…")`, `InvalidParamsError("…")`, …) passes through with its message, so the place that knows what the caller should hear can say it — a tool, a hook, or an LLM wrapper:

```python
import litellm
from a2a.types import InternalError


class MyLLM(LiteLLMChat):
    async def ainvoke(self, messages, **kw):
        try:
            return await super().ainvoke(messages, **kw)
        except litellm.ContextWindowExceededError as e:
            raise InternalError(
                "This conversation has grown too long; start a new one."
            ) from e
```

### Asking the user for input

A tool can return a [`ClarificationNeededOutput`][ant_ai.core.result.ClarificationNeededOutput] instead of a result to say it needs something only a person can supply. The built-in [`HumanInputNeededTool`][ant_ai.tools.builtins.human_input.HumanInputNeededTool] does exactly that, and your own tools can too:

```python
from ant_ai.core.result import ClarificationNeededOutput


@tool
def deploy(env: str) -> str | ClarificationNeededOutput:
    """Deploy to an environment."""
    if env == "prod":
        return ClarificationNeededOutput(question="Deploy to prod — are you sure?")
    ...
```

The agent emits a [`ClarificationNeededEvent`][ant_ai.core.events.ClarificationNeededEvent] with the question and **stops**. The clarified tool call is answered in the transcript with its own question, so the conversation is left well-formed — `assistant(tool_calls)` → `tool` — and resuming is just adding the user's reply and running again:

```python
state = State(messages=[Message(role="user", content="ship it to prod")])
async for event in agent.stream(state, ctx=ctx):
    if isinstance(event, ClarificationNeededEvent):
        answer = input(event.content)  # ask a person
        state.add_message(Message(role="user", content=answer))
        # then call agent.stream(state, ctx=ctx) again
```

Over A2A the task enters `input-required`; the caller resumes by sending the answer with the same `context_id`, and the agent rebuilds the transcript from task history.

Use a clarification when the run genuinely cannot proceed without the answer. If the tool just needs to *tell* the user something and let the model carry on — a precondition it can't satisfy, say — raise a [`ToolError`](#reporting-failures) instead: the model sees the message, does what it can, and relays it in its answer.

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

| Class                                                               | Meaning                         |
| ------------------------------------------------------------------- | ------------------------------- |
| [`UpdateEvent`][ant_ai.core.events.UpdateEvent]                   | Intermediate LLM output         |
| [`ToolCallingEvent`][ant_ai.core.events.ToolCallingEvent]         | Agent is about to call a tool   |
| [`ToolResultEvent`][ant_ai.core.events.ToolResultEvent]           | Tool returned a result          |
| [`FinalAnswerEvent`][ant_ai.core.events.FinalAnswerEvent]         | Agent produced its final answer |
| [`MaxStepsReachedEvent`][ant_ai.core.events.MaxStepsReachedEvent] | Loop hit the step limit         |

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
    state.add_message(
        Message(role="user", content="Write a Python function that reverses a string.")
    )

    async for event in agent.stream(state, ctx=ctx):
        yield event  # forward events to the caller

    yield state  # always yield the updated state at the end
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
workflow.add_edge("revise", "validate")  # retry loop
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
