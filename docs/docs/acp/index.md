---
title: ACP
---

# IDE integration

The [Agent Client Protocol](https://agentclientprotocol.com/) (ACP) is how code editors talk to agents. Where [A2A](../multi-agent/index.md) connects agents to each other, ACP connects an agent to a person's editor — Zed, VSCode, the Gemini CLI, or any other ACP client.

[`ACPServer`][ant_ai.acp.server.ACPServer] serves an [`Agent`][ant_ai.agent.agent.Agent] and its [`Workflow`][ant_ai.workflow.workflow.Workflow] over the protocol. The agent itself needs no changes: the workflow that answers `agent.stream()` also answers a prompt typed in the editor.

## Core concepts

| Concept                                       | Description                                                                    |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| [`ACPServer`][ant_ai.acp.server.ACPServer]     | Serves an agent over ACP, either on stdio or over WebSocket.                     |
| [`ACPAdapter`][ant_ai.acp.adapter.ACPAdapter]  | Implements the protocol: sessions, prompts, per-session MCP servers, commands.   |
| **ACP tools**                                  | Tools that call back into the editor — read files, run terminals, push a plan.   |
| [`ACPCommand`][ant_ai.acp.commands.ACPCommand] | A slash command advertised in the editor's prompt UI.                            |

## Serving an agent

Most editors spawn the agent as a subprocess and talk to it over stdin/stdout:

```python
from ant_ai.acp import ACPServer

server = ACPServer(agent=agent, workflow=workflow)
server.serve_stdio()
```

The editor is then pointed at the script that calls it:

```json
{
    "acp.agents": {
        "ant-ai": { "command": "python", "args": ["my_agent.py"] }
    }
}
```

In this mode stdout _is_ the protocol channel, so never `print()` from a tool or handler — `loguru` writes to stderr and is safe.

To keep the agent on a remote machine instead, `server.serve()` exposes it at `ws://host:port/acp/ws`. `starlette_app()` and `fastapi_app()` return the ASGI application, whose routes sit on paths disjoint from A2A's — so one process can serve both protocols.

## What the editor sees

Events from the workflow are translated into ACP session updates as they stream, so the editor renders structured UI rather than a wall of text: content deltas arrive as message chunks, reasoning as collapsible thoughts, and each tool call appears as an entry that flips to _completed_ when its result comes back.

## Tools that reach into the IDE

Because the session runs inside the user's editor, the agent can ask it to do things. These come as ordinary tools:

```python
from ant_ai.acp import ACP_FILESYSTEM_TOOLS, ACP_TERMINAL_TOOLS

agent = Agent(..., tools=[*ACP_FILESYSTEM_TOOLS, *ACP_TERMINAL_TOOLS])
```

| Bundle                 | What the agent can do                                                    |
| ---------------------- | -------------------------------------------------------------------------- |
| `ACP_SESSION_TOOLS`    | Ask for the session's working directory.                                   |
| `ACP_FILESYSTEM_TOOLS` | List a directory, read a file (optionally a line range), write a file.     |
| `ACP_TERMINAL_TOOLS`   | Run a command in the IDE terminal, or drive a long-running one by id.      |
| `ACP_PLAN_TOOLS`       | Push a checklist of steps into the editor's UI, and update it as it works. |

`ACP_ALL_TOOLS` is all four. Relative paths are resolved against the session's working directory, so the model can pass a bare filename. Clients declare what they support when they connect; a tool whose capability is missing raises, and the model sees the error as a tool result and can route around it.

!!! warning
    Terminal commands run on the user's machine with the user's permissions. Constrain what the agent may run through its system prompt.

## Slash commands

Commands passed to the server are advertised to the editor and appear in its prompt UI:

```python
from ant_ai.acp import ACPCommand

ACPServer(
    agent=agent,
    workflow=workflow,
    commands=[
        ACPCommand(
            name="plan",
            description="Draft a plan before acting",
            kind="prompt",
            template="Draft a short plan for the task below, then wait for approval:\n\n{args}",
        )
    ],
)
```

A `kind="prompt"` command expands its template (`{args}` is the text after the command) and runs a normal agent turn. A `kind="code"` command instead runs an async `handler` with no model turn at all — it receives an [`ACPCommandContext`][ant_ai.acp.commands.ACPCommandContext] holding the live session: its message history, working directory, and agent. That is enough for a `/compact` that rewrites the transcript in place, or a command that swaps in a differently configured agent for the rest of the session.

## Sessions

Each session keeps its own history, working directory, and agent instance, in memory for the life of the process; clients can create, load, list, fork, and close them. MCP servers passed when a session opens are loaded and attached to a copy of the agent, so tools stay scoped to that session and never leak into another.
