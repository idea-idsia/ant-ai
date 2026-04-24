<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/idea-idsia/ant-ai/main/docs/assets/ant_h_white.png">
  <img alt="ANT AI" src="https://raw.githubusercontent.com/idea-idsia/ant-ai/main/docs/assets/ant_h_dark.png" height="100">
</picture>

![Python](https://img.shields.io/badge/python-3.14%2B-4584b6?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?logo=MIT&logoColor=white-lightgrey)
[![Coverage](https://codecov.io/gh/idea-idsia/ant-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/idea-idsia/ant-ai)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526cfe?logo=materialformkdocs&logoColor=white)](https://idea-idsia.github.io/ant-ai/)

**A lightweight Python framework for building tool-driven AI agents and multi-agent systems.**

</div>

---

`ant-ai` is a lightweight Python framework for building multi-agent systems: graph-based workflow orchestration, first-class agent-to-agent (A2A) communication via the [A2A protocol](https://github.com/a2aproject/A2A), MCP tool integration, lifecycle hooks for guardrails, and built-in observability — all on top of an LLM-agnostic core.

## Why ANT AI

**Multi-agent by design.** Agents communicate and delegate natively via the [A2A protocol](https://github.com/a2aproject/A2A) — build systems that grow without rewrites.

**No lock-in.** Swap LLMs, tools, or observability backends in one line. Your logic stays untouched.

**Structured, not scripted.** Model complex behavior as graphs. Know exactly what runs, when, and why.

**Production-ready out of the box.** Hooks, guardrails, and full observability via [Langfuse](https://langfuse.com/) and OpenTelemetry — without extra setup.

## Installation

Requires Python 3.14+. Install with [uv](https://docs.astral.sh/uv/):

```sh
uv add ant-ai
```

Or clone and sync for local development:

```sh
git clone https://github.com/idea-idsia/ant-ai
cd ant-ai
uv sync --all-extras
```

## Quickstart

### Single agent

```python
from ant_ai import Agent, Message, State, tool
from ant_ai.llm.integrations import LiteLLMChat

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22°C in {city}"

llm = LiteLLMChat(model="gpt-4o-mini")

agent = Agent(
    name="WeatherAgent",
    system_prompt="You are a helpful weather assistant.",
    llm=llm,
    tools=[get_weather],
)

state = State(messages=[Message(role="user", content="What's the weather in Lugano?")])
answer = agent.invoke(state)
print(answer)
```

### Streaming events

```python
from ant_ai.core import FinalAnswerEvent

async for event in agent.stream(state):
    if isinstance(event, FinalAnswerEvent):
        print(event.content)
```

### Structured output

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature: int
    condition: str

answer = agent.invoke(state, response_schema=WeatherReport)
# answer is a JSON string matching WeatherReport
```

## Development

```sh
# Install dev dependencies and pre-commit hooks
uv sync --all-extras
uv run pre-commit install

# Run tests
uv run pytest

# Serve docs locally
uv run mkdocs serve
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributing guide, branching model, and review process.

## License

This software is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.
