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

Every [`ChatLLM`][ant_ai.llm.protocol.ChatLLM] backend is constructed the same way: `Backend(model, *, api_key=None, api_base=None)`.
To point one at your own endpoint (vLLM, a proxy, …) pass the credential and URL directly, or set them in the environment under the backend's provider name:

```python
llm = LiteLLMChat(
    "gpt-4o-mini",
    api_key=os.environ["MY_DEPLOYMENT_API_KEY"],  # keep the secret under your own name
    api_base="http://localhost:8000/v1",
)
```

| Backend       | `api_key` fallback | `api_base` fallback |
| ------------- | ------------------ | ------------------- |
| `LiteLLMChat` | `LITELLM_API_KEY`  | `LITELLM_API_BASE`  |
| `OpenAIChat`  | `OPENAI_API_KEY`   | `OPENAI_BASE_URL`   |

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

### Limiting tool results

Everything a tool returns goes into the conversation, so one long result can fill the model's context. Before a result is serialized, [`ToolStep`][ant_ai.steps.tool_step.ToolStep] passes it through the tool's [`limit_output()`][ant_ai.tools.tool.Tool.limit_output]. By default, a result longer than `max_output_chars` characters (100,000) keeps its start, with a note that tells the model how much was cut. Set `max_output_chars=None` for no limit, also on a tool loaded with `mcp_tools_from_url()`. The trace keeps the full result.

A tool that knows its results overrides `limit_output()`, with the helpers [`keep_start()`][ant_ai.tools.tool.keep_start] and [`keep_end()`][ant_ai.tools.tool.keep_end], which also tell the model how much was cut and, with `hint`, how to get the rest:

```python
from ant_ai import Tool
from ant_ai.tools import keep_end


class Logs(Tool):
    def tail(self, service: str) -> str:
        """Return the log of a service."""
        ...

    def limit_output(self, output):
        return keep_end(output, 20_000, hint="ask for a shorter time range")
```

For a class-based tool, `limit_output()` applies to all its methods. [`ShellTool`][ant_ai.tools.builtins.shell_tool.ShellTool] and [`FilesystemTool`][ant_ai.tools.builtins.filesystem_tool.FilesystemTool] use the same limit but cut in their own way: the shell keeps the end of `stdout` and `stderr`, where a failing command usually says why; the filesystem keeps the start of a file or of the search matches and the first entries of a listing. Their notes point the model to a filtered command, `read_file` with `offset` and `limit`, or a subfolder.

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

### Agents as tools (subagents)

An agent in the `tools` of another agent becomes one of its tools, in the same process. Each call runs the agent on a new conversation that holds only the task, so the files it reads and the tools it calls stay out of the caller's conversation: the caller sees one tool call and the final answer.

```python
from ant_ai.agent import Agent
from ant_ai.tools.builtins import FilesystemTool

explorer = Agent(
    name="explorer",
    description="Finds where something is done in the code base and answers with file and line.",
    system_prompt="Answer the question you are given. Cite files and lines.",
    llm=llm,
    tools=[FilesystemTool(workspace_root=".")],
)
lead = Agent(name="lead", system_prompt="...", llm=llm, tools=[explorer])
```

The tool takes the name and description of the agent and has one argument, `message`, as an [`A2AAgentTool`][ant_ai.a2a.agent.A2AAgentTool]: the model has to write the complete task, since the subagent sees nothing else of the conversation. Each run has 10 steps. To change these settings, pass [`as_tool()`][ant_ai.agent.base.BaseAgent.as_tool] instead of the agent: `explorer.as_tool(name="find", max_steps=20)`. The caller's `ctx` is passed on, so the run shares the session and the trace. If the subagent runs out of steps or asks for a clarification, the call fails like a raised [`ToolError`](#reporting-failures). An agent with `planning=True` is refused: only the agent that talks to the user keeps a plan. For agents in other processes, see [Multi-agent setup](../multi-agent/index.md).

### Subagents created at runtime

With `as_tool()`, the subagents are written in code beforehand. On an open task the agent cannot know in advance which helpers it needs: add a [`SubagentTool`][ant_ai.agent.subagent.SubagentTool] and the model creates them while it works.

```python
from ant_ai.agent import Agent, SubagentTool
from ant_ai.tools.builtins import FilesystemTool, ShellTool

lead = Agent(
    name="lead",
    system_prompt="...",
    llm=llm,
    tools=[FilesystemTool(workspace_root="."), ShellTool(cwd="."), SubagentTool()],
)
```

The model calls the tool with a name for the subagent, the task and the names of the tools it may use, chosen among the agent's own: a subagent can never do more than the agent that creates it. The subagent runs on the same LLM and, as with `as_tool()`, gets one task and returns one answer. Every subagent has the same system prompt, a short default you can replace with `SubagentTool(system_prompt="...")`; `max_steps` is the step budget of each run (10 by default) and `name` the name of the tool (`subagent`). A subagent cannot create subagents of its own nor keep a plan, so the `SubagentTool` and the planning tools are never offered to it.

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

Use this for anything scoped to the caller rather than the conversation — per-user data lookups, multi-tenant isolation, audit logging. [`Mem0Memory`][ant_ai.memory.backends.mem0.Mem0Memory] (see [Agent memory](memory.md)) is a built-in example: its `search`/`add` tool methods take `ctx` this way to scope memories per user.

!!! warning "Scoping is only as safe as the identity behind it"
    Behind an A2A server, `ctx.user_id` comes from the request metadata, which the client writes. Unless the server sets it from an authenticated identity, any client can claim another user's `user_id` and read or change their data. Before relying on `user_id` to keep users apart, authenticate requests and overwrite or reject a `user_id` that does not belong to the caller, for example in your own `from_metadata`.

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

A failure inside an A2A run reaches the caller as a bare `InternalError` — the exception text is **never** forwarded, since it can carry prompt content or internal detail and the caller may not be the operator. Two things pass through with a message:

- [`ContextWindowExceededError`][ant_ai.llm.ContextWindowExceededError], which `LiteLLMChat` and `OpenAIChat` raise in place of their provider's own error when the conversation no longer fits the model. Its message is written by the library, not copied from the provider, so the caller is told the conversation is too long and nothing else. You can also catch it yourself around `Agent.run()` or in a tool, without importing the provider.
- Any `a2a.utils.errors.A2AError` you raise yourself (`InternalError("…")`, `InvalidParamsError("…")`, …), so the place that knows what the caller should hear can say it — a tool, a hook, or an LLM wrapper.

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

Over A2A the task stays `input-required`; the caller continues it by sending the answer with the task's `task_id`, as A2A specifies, and the agent resumes with the task's history and the turns it references.

Use a clarification when the run genuinely cannot proceed without the answer. If the tool just needs to *tell* the user something and let the model carry on — a precondition it can't satisfy, say — raise a [`ToolError`](#reporting-failures) instead: the model sees the message, does what it can, and relays it in its answer.

### Keeping a plan

On a task with many steps, `planning=True` lets the agent write its plan, a list of items each `pending`, `in_progress` or `completed`, and keeps it in front of the model:

```python
agent = Agent(
    name="worker",
    system_prompt="For a task with several steps, write a plan with write_todos and keep it up to date.",
    llm=llm,
    planning=True,
)
```

The plan is saved in `.plans/<agent>/<session_id>.md`, so it is still there on the next message of the conversation, also over A2A, and it is cleared once every item is completed. When the conversation does not show it, because a new `State` was started for the same `session_id` or [`HistoryCompressionHook`][ant_ai.hooks.builtins.history_compression.HistoryCompressionHook] has summarised it away, it is added to the end of what the model receives. Without a `session_id` the plan is not saved (see [Keeping a conversation](#keeping-a-conversation)), and agents left with the default name `BaseAgent` share one folder of plans.

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

### Keeping a conversation

A run belongs to a conversation only when its `ctx` has a `session_id`: the plan of `planning=True` is saved under it, the trace is grouped by it, and every event of the run carries it in `event.session_id` (events that already have one, such as those of a remote A2A agent, keep theirs). A run without a `ctx` has no session. To keep a conversation from a direct call, create the id yourself and pass the same `ctx` and `state` again on the next message:

```python
from uuid import uuid4

ctx = InvocationContext(session_id=str(uuid4()))
async for event in agent.stream(state, ctx=ctx):
    ...

state.add_message(Message(role="user", content="And now add a test."))
async for event in agent.stream(state, ctx=ctx):
    ...
```

The same works for `ainvoke` and for a [`Workflow`](#adding-a-workflow).

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
