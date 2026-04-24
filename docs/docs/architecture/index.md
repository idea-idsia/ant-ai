---
title: Architecture
---

# Architecture Overview

## Components

| Module                           | Component                                                                                               | Responsibility                                                                                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <nobr>`ant_ai.agent`</nobr>    | [`Agent`][ant_ai.agent.agent.Agent]                                                                   | Core reasoning unit. Runs the ReAct loop: queries the LLM, executes tool calls, and streams events until a final answer is reached.                                               |
| <nobr>`ant_ai.workflow`</nobr> | [`Workflow`][ant_ai.workflow.workflow.Workflow]                                                       | Directed graph of nodes (actions) connected by static edges or conditional routers. Orchestrates what the agent does and in what order.                                           |
| <nobr>`ant_ai.tools`</nobr>    | [`Tool`][ant_ai.tools.tool.Tool]                                                                      | Callables exposed to the LLM via JSON schema. Defined with the `@tool` decorator or as a `Tool` subclass for grouped namespaces.                                                  |
|                                  | [`ToolRegistry`][ant_ai.tools.registry.ToolRegistry]                                                  | Built automatically from the agent's tool list. Expands namespace tools into individually callable entries.                                                                       |
| <nobr>`ant_ai.a2a`</nobr>      | [`Colony`][ant_ai.a2a.colony.Colony]                                                                        | Multi-agent coordinator. Registers agents with their workflows and A2A cards, wires collaboration edges, and produces ASGI apps for deployment.                                   |
|                                  | [`A2AExecutor`][ant_ai.a2a.executor.A2AExecutor]                                                      | ASGI request handler. Receives incoming A2A requests, initialises `InvocationContext` and `State`, drives `Workflow.stream()`, and translates events to A2A task updates. |
|                                  | [`A2AAgentTool`][ant_ai.a2a.agent.A2AAgentTool]                                                       | A `Tool` that calls a remote agent over HTTP. Added to source agents automatically by `Colony.collab()`.                                                                            |
| <nobr>`ant_ai.core`</nobr>     | [`AgentEvent`][ant_ai.core.events.AgentEvent] / [`WorkflowEvent`][ant_ai.core.events.WorkflowEvent] | Typed events emitted by the agent and workflow respectively. All progress — LLM output, tool calls, node transitions — is observable through this stream.                         |

A high level view of the interaction of these components is shown below:

```mermaid
---
title: Component relationships
---
flowchart TD
    Client(["Client"])
    Remote(["Remote agent"])

    subgraph ext_llm["LLM provider"]
        LLM["LiteLLM / OpenAI"]
    end

    subgraph hm["ant_ai"]
        Colony["Colony"]
        A2AS["A2AServer"]
        Exec["A2AExecutor"]
        Agent["Agent"]

        subgraph reg["ToolRegistry"]
            Tool["Tool"]
            A2AT["A2AAgentTool"]
        end

        subgraph wf["Workflow"]
            direction LR
            S(("START")) --> N1["Node"]
            N1 -->|router| N2["Node"]
            N1 -->|router| N3["Node"]
            N2 --> E(("END"))
            N3 --> E
        end

        Colony -.->|"wires & deploys"| A2AS
        Colony -.->|"wires & deploys"| Agent
        A2AS --> Exec
        Exec --> wf
        wf --> Agent
        Agent --> reg
    end

    subgraph ext_a2a["a2a SDK"]
        SDK["RequestHandler"]
    end

    Client -->|"A2A request"| A2AS
    Exec -.->|"extends"| SDK
    Agent -.->|"LLM calls"| LLM
    A2AT -->|"A2A call"| Remote
    Exec -->|"streamed updates"| Client
```

## Flow of a Request

In the A2A integration, [`A2AExecutor.execute()`][ant_ai.a2a.executor.A2AExecutor.execute] acts as the streaming entrypoint: it ensures there is an active [`Task`](https://a2a-protocol.org/latest/sdk/python/api/a2a.html#a2a.types.Task), builds an [`InvocationContext`][ant_ai.core.types.InvocationContext] and initial [`State`][ant_ai.core.types.State] (including converted history), then consumes the asynchronous event stream produced by [`Workflow.stream()`][ant_ai.workflow.workflow.Workflow]. The workflow is a graph of actions (nodes) connected by static edges or routers; each node execution emits [`WorkflowEvent`][ant_ai.core.events.WorkflowEvent] updates (started/completed/update) and may delegate to the [`Agent`][ant_ai.agent.agent.Agent] for LLM-driven logic (including tool calls), which surfaces as [`AgentEvent`][ant_ai.core.events.AgentEvent] (`update`, `tool_calling`, `tool_result`, `final`). Every event, whether originating from the workflow or the agent, is translated via [`HVEventToA2A.apply()`][ant_ai.a2a.translator.HVEventToA2A.apply] and applied to the task through [`TaskUpdater`](https://a2a-protocol.org/latest/sdk/python/api/a2a.server.tasks.html#a2a.server.tasks.TaskUpdater), resulting in incremental task updates being streamed back to the client; this makes progress observable end-to-end until the workflow emits a final `COMPLETED` event and the task reaches a terminal state.

```mermaid
---
title: Architecture (event streaming)
---
sequenceDiagram
    autonumber
    participant Client as User/Agent
    participant Exec as A2AExecutor
    participant WF as Workflow
    participant Act as Action
    participant Agent as Agent
    participant LLM as LLM

    Client->>Exec: Send request
    Exec->>WF: Start workflow stream

        WF-->>Exec: WorkflowEvent (START)
        Exec-->>Client: Stream update

        loop For each action/node (many)
            WF-->>Exec: WorkflowEvent (Action STARTED)
            Exec-->>Client: Stream update

            WF->>Act: Run action
            Act->>Agent: Run agent logic

            loop ReAct iterations (many)
                Agent->>LLM: Query LLM
                LLM-->>Agent: Response
                Agent-->>Exec: AgentEvent (update)
                Exec-->>Client: Stream update

                opt Tools (many)
                    Agent-->>Exec: AgentEvent (tool call)
                    Exec-->>Client: Stream update
                    Agent-->>Exec: AgentEvent (tool result)
                    Exec-->>Client: Stream update
                end
            end

            Agent-->>Exec: AgentEvent (final)
            Exec-->>Client: Stream update

            WF-->>Exec: WorkflowEvent (Action COMPLETED)
            Exec-->>Client: Stream update

            WF->>WF: Select next action (edge/router)
        end

        WF-->>Exec: WorkflowEvent (COMPLETED)
        Exec-->>Client: Stream update
```
