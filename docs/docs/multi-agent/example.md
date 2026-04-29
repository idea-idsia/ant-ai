---
title: Example
---

# Multi-agent example: Software Engineering Colony

This example walks through a three-agent colony that collaborates to write, test, and review Python code.

## Agents

| Agent     | Port | Responsibility       | Can call  |
| --------- | ---- | -------------------- | --------- |
| `codegen` | 9001 | Writes Python code   | `testgen` |
| `testgen` | 9002 | Writes pytest tests  | `codegen` |
| `quality` | 9003 | Reviews code quality | `codegen` |

The collaboration graph:

```
codegen ──► testgen
quality ──► codegen
```

## 1. Define the agents

Each agent is created with a name, an LLM, and a system prompt. Tools provided by collaborators are added automatically by `Colony.collab()` — no manual wiring needed.

```python
# agents/codegen/codegen.py
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from ant_ai import Agent
from ant_ai.llm.integrations import LiteLLMChat

def build_codegen_agent() -> tuple[Agent, AgentCard]:
    agent = Agent(
        name="Developer",
        llm=LiteLLMChat("gpt-4o-mini"),
        system_prompt="You are a senior Python developer. Write clean, well-tested code.",
        description="A 10x Developer",
    )

    card = AgentCard(
        name="codegen",
        description="A 10x Software Developer",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url="http://codegen-agent:9001/")],
        skills=[AgentSkill(
            id="develop",
            name="Write python code",
            description="Returns high-quality Python code",
            tags=["code", "develop"],
        )],
    )

    return agent, card
```

Define `build_testgen_agent()` and `build_quality_agent()` the same way, pointing to ports 9002 and 9003 respectively.

## 2. Define workflows

A workflow controls the multi-step reasoning each agent performs. The `codegen` workflow generates code, validates it, and fixes it in a loop until it passes.

```python
# agents/codegen/workflow.py
from collections.abc import AsyncGenerator
from typing import Literal

from ant_ai import Agent, Message, InvocationContext, State
from ant_ai.workflow import END, START, NodeYield, Workflow


async def generate_code(
    agent: Agent, state: State, ctx: InvocationContext
) -> AsyncGenerator[NodeYield]:
    state.add_message(Message(role="user", content="Write the requested Python code."))
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


async def validate_code(
    agent: Agent, state: State, ctx: InvocationContext
) -> AsyncGenerator[NodeYield]:
    state.add_message(Message(role="user", content="Is the code above correct? Reply with VALID or NOT VALID."))
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


async def fix_code(
    agent: Agent, state: State, ctx: InvocationContext
) -> AsyncGenerator[NodeYield]:
    state.add_message(Message(role="user", content="Fix the issues identified above."))
    async for event in agent.stream(state, ctx=ctx):
        yield event
    yield state


async def code_validation_result(
    agent: Agent, state: State, ctx: InvocationContext
) -> Literal["fix_code", "END"]:
    last_message = state.last_message.content or ""
    if "NOT" in last_message:
        return "fix_code"
    return END


def build_codegen_workflow() -> Workflow:
    workflow = Workflow()
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("validate_code", validate_code)
    workflow.add_node("fix_code", fix_code)

    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "validate_code")
    workflow.add_conditional_edge("validate_code", code_validation_result)
    workflow.add_edge("fix_code", "validate_code")   # retry loop

    return workflow
```

The execution path:

```
START → generate_code → validate_code ─── VALID ──► END
                              │
                           NOT VALID
                              │
                           fix_code ──────────────► validate_code
```

Define analogous workflows for `testgen` (generate → validate → fix tests) and `quality` (review → done).

## 3. Assemble the colony

```python
# colonies/se_hive.py
from ant_ai.a2a import Colony

from myapp.agents.codegen.codegen import build_codegen_agent
from myapp.agents.codegen.workflow import build_codegen_workflow
from myapp.agents.testgen.testgen import build_testgen_agent
from myapp.agents.testgen.workflow import build_testgen_workflow
from myapp.agents.quality.quality import build_quality_agent
from myapp.agents.quality.workflow import build_quality_workflow


def build_se_hive() -> Colony:
    codegen, codegen_card = build_codegen_agent()
    testgen, testgen_card = build_testgen_agent()
    quality, quality_card = build_quality_agent()

    colony = Colony(db_url="postgresql+asyncpg://user:pass@host/db")

    colony.agent("codegen", agent=codegen, card=codegen_card, workflow=build_codegen_workflow())
    colony.agent("testgen", agent=testgen, card=testgen_card, workflow=build_testgen_workflow())
    colony.agent("quality", agent=quality, card=quality_card, workflow=build_quality_workflow())

    colony.collab("codegen", "testgen")   # codegen gets a tool to call testgen
    colony.collab("quality", "codegen")   # quality gets a tool to call codegen

    return colony
```

## 4. Deploy

Each agent runs as a separate process. A single CLI entry point selects which agent to start:

```python
# cli.py
import typer, uvicorn
from ant_ai.a2a import Colony
from myapp.colonies.se_hive import build_se_hive

app = typer.Typer()

@app.command()
def start(agent: str):
    colony: Colony = build_se_hive()
    asgi_app = colony.asgi(agent_name=agent, use_fastapi=True)
    _, port = colony.get_agent_host(agent)
    uvicorn.run(asgi_app, host="0.0.0.0", port=port)
```

```bash
python -m myapp start codegen   # → 0.0.0.0:9001
python -m myapp start testgen   # → 0.0.0.0:9002
python -m myapp start quality   # → 0.0.0.0:9003
```

### Docker Compose

```yaml
services:
    codegen-agent:
        build: .
        command: python -m myapp start codegen
        ports:
            - "9001:9001"

    testgen-agent:
        build: .
        command: python -m myapp start testgen
        ports:
            - "9002:9002"

    quality-agent:
        build: .
        command: python -m myapp start quality
        ports:
            - "9003:9003"

    db:
        image: postgres:17
        environment:
            POSTGRES_DB: ant_ai
            POSTGRES_USER: user
            POSTGRES_PASSWORD: pass
```

The `Colony` uses service names (`codegen-agent`, `testgen-agent`, `quality-agent`) as hostnames, matching the URLs in each `AgentCard.supported_interfaces`.

## How a request flows

1. A client sends `"Write a function that sorts a list"` to the `codegen` agent.
1. `codegen` runs `generate_code` → `validate_code`. In the process, it can optionally call `testgen` to generate tests.
1. `testgen` receives the code from `codegen`, generates tests, validates them, and returns the result.
1. `codegen` incorporates the test output and produces its final answer.
1. Independently, `quality` can be called to review any code, and may in turn call `codegen` to fix issues it finds.
