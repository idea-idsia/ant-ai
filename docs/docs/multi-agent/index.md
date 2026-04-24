---
title: Multi-agent
---

# Multi-agent setup

A [`Colony`][ant_ai.a2a.colony.Colony] connects multiple agents over the [A2A protocol](https://a2a-protocol.org/), so they can delegate tasks to each other at runtime. Each agent in the colony runs as its own ASGI service and is reachable by the others as a tool.

## Core concepts

| Concept                | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| **Agent**              | An LLM actor with tools and a workflow (see [Single-agent](../single-agent/index.md)). |
| **AgentCard**          | A2A metadata: the agent's URL, skills, and capabilities.                               |
| **Colony**             | Registry that wires agents together and produces ASGI apps for deployment.             |
| **Collaboration edge** | A directed link allowing one agent to call another as a tool.                          |

## Registering agents

Create a `Colony` and register each agent with `colony.agent()`. Every agent needs an `AgentCard` that declares its URL and skills.

```python
from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill
from ant_ai.a2a import Colony

colony = Colony(db_url="postgresql+asyncpg://user:pass@host/db")

card = AgentCard(
        name="codegen",
        description="Writes Python code.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url="http://codegen-agent:9001/")],
        skills=[AgentSkill(id="develop", name="Write python code", ...)],
    )

colony.agent(
    "codegen",
    agent=codegen_agent,
    workflow=codegen_workflow,
    card=card,
)
```

If you don't need a persistent task store, omit `db_url` and the colony will use an in-memory store.

## Setting up collaborations

`colony.collab(source, target)` adds the target agent as a callable tool on the source agent. The source can then invoke the target during its ReAct loop, exactly like any other tool.

```python
colony.collab("codegen", "testgen")   # codegen can call testgen
colony.collab("quality", "codegen")   # quality can call codegen

# bidirectional shorthand
colony.collab("codegen", "quality", mutual=True)
```

Under the hood this creates an [`A2AAgentTool`][ant_ai.a2a.agent.A2AAgentTool] configured with the target's URL (from `AgentCard.supported_interfaces[0].url`), and attaches it to the source agent.

## Deploying agents

`colony.asgi(agent_name=...)` returns a FastAPI (or Starlette) ASGI application for the named agent, ready to be served with any ASGI server.

```python
import uvicorn

asgi_app = colony.asgi(agent_name="codegen", use_fastapi=True)
_, port = colony.get_agent_host("codegen")
uvicorn.run(asgi_app, host="0.0.0.0", port=port)
```

Each agent is a separate process. Run them independently (e.g. one container each):

```bash
python -m myapp start codegen   # listens on port 9001
python -m myapp start testgen   # listens on port 9002
python -m myapp start quality   # listens on port 9003
```

The `Colony` object must be constructed identically in every process so that each agent knows the full collaboration graph and can route calls correctly.

## AgentCard URL convention

Use DNS names that resolve inside your deployment environment (Docker Compose service names, Kubernetes service names, etc.):

```
http://codegen-agent:9001/
http://testgen-agent:9002/
http://quality-agent:9003/
```

## End-to-end request flow

1. An external client sends a request to one agent (the _entry point_).
2. The receiving agent's `Workflow` runs; nodes invoke the `Agent` via `agent.stream()`.
3. When the agent decides to call a collaborator, it issues an `A2AAgentTool` call, which makes an HTTP request to the target agent's A2A endpoint.
4. The target agent runs its own workflow, streams events back, and returns a final answer.
5. The source agent incorporates the result as a tool result and continues its ReAct loop.

See [Architecture](../architecture/index.md) for the full event-streaming sequence diagram.

## Full example

See the [Multi-agent example](example.md) for a complete software-engineering colony with three collaborating agents.
