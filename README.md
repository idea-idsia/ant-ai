<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/idea-idsia/ant-ai/main/docs/assets/ant_h_white.png">
  <img alt="ANT AI" src="https://raw.githubusercontent.com/idea-idsia/ant-ai/main/docs/assets/ant_h_dark.png" height="100">
</picture>

**A lightweight Python framework for building tool-driven AI agents and multi-agent systems.**

[![PyPI](https://img.shields.io/pypi/v/ant-ai?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/ant-ai/)
[![Python](https://img.shields.io/badge/Python-3.14%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/idea-idsia/ant-ai?label=Coverage&logo=codecov)](https://codecov.io/gh/idea-idsia/ant-ai)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![DOI](https://zenodo.org/badge/1219140757.svg)](https://doi.org/10.5281/zenodo.21276625)

[**Documentation**](https://idea.idsia.ch/ant-ai/) · [**Install**](https://idea.idsia.ch/ant-ai/docs/install/) · [**Quickstart**](#quickstart) · [**Architecture**](https://idea.idsia.ch/ant-ai/docs/architecture/) · [**Contributing**](CONTRIBUTING.md)

</div>

---

Agents that talk to each other, tools that just work, and a graph you can actually reason about — `ant-ai` is a lightweight Python framework for building multi-agent systems, from a single tool-using agent to a whole colony of them.

## Why ANT AI

| | |
| --- | --- |
| 🐜 **Multi-agent by design** | Agents communicate and delegate over the [A2A protocol](https://github.com/a2aproject/A2A) — no custom glue code required. |
| 🧩 **Editor-native** | Serve any agent to Zed, VSCode, or the Gemini CLI over the [Agent Client Protocol](https://agentclientprotocol.com/), with filesystem, terminal, and slash-command support built in. |
| 🔌 **No lock-in** | Swap LLMs, tools, or observability backends without touching your agent logic. |
| 📐 **Structured, not scripted** | Model complex behavior as graphs — know exactly what runs, when, and why. |
| 🔭 **Observable from day one** | Built-in tracing via [Langfuse](https://langfuse.com/) and lifecycle hooks for guardrails. |

## Installation

Requires Python 3.14+. Install with [uv](https://docs.astral.sh/uv/):

```sh
uv add ant-ai
```

Or grab everything at once:

```sh
uv add "ant-ai[all]"
```

Need just a piece — `openai`, `langfuse`, `mem0`, `guardrails-ai`, `datafog`, `viz`? See the [install guide](https://idea.idsia.ch/ant-ai/docs/install/) for the full list of extras.

Or clone and sync for local development:

```sh
git clone git@github.com:idea-idsia/ant-ai.git
cd ant-ai
uv sync --all-packages --all-groups --all-extras
```

## Quickstart

### Your first agent

An agent is an LLM, a system prompt, and a set of tools. Decorate a function with `@tool` and it becomes callable by the model:

```python
from ant_ai import Agent, Message, State, tool
from ant_ai.llm.integrations import LiteLLMChat


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22°C in {city}"


agent = Agent(
    name="WeatherAgent",
    system_prompt="You are a helpful weather assistant.",
    llm=LiteLLMChat(model="gpt-4o-mini"),
    tools=[get_weather],
)

state = State(messages=[Message(role="user", content="What's the weather in Lugano?")])
print(agent.invoke(state))
```

### Streaming

`agent.stream()` interleaves live `ContentDeltaEvent`s (token by token) before the terminal event they build up to — match on it for live text, or ignore it and just take the final answer:

```python
from ant_ai.core import ContentDeltaEvent, FinalAnswerEvent

async for event in agent.stream(state):
    if isinstance(event, ContentDeltaEvent):
        print(event.delta, end="", flush=True)
    elif isinstance(event, FinalAnswerEvent):
        print()
```

### Structured output

Pass a Pydantic model as `response_schema` and the final answer comes back as JSON matching it:

```python
from pydantic import BaseModel


class WeatherReport(BaseModel):
    city: str
    temperature: int
    condition: str


answer = agent.invoke(state, response_schema=WeatherReport)
# answer is a JSON string matching WeatherReport
```

### A colony of agents

A `Colony` wires agents together over A2A: each one runs as its own ASGI service, and a collaboration edge makes one agent callable as a tool by another.

```python
from ant_ai.a2a import Colony

colony = Colony()
colony.agent(
    "codegen", agent=codegen_agent, workflow=codegen_workflow, card=codegen_card
)
colony.agent(
    "testgen", agent=testgen_agent, workflow=testgen_workflow, card=testgen_card
)

colony.collab("codegen", "testgen")  # codegen can now call testgen as a tool

asgi_app = colony.asgi(agent_name="codegen", use_fastapi=True)
```

### Serving an agent to your editor

`ACPServer` exposes an agent over the Agent Client Protocol, so it runs inside Zed, VSCode, or the Gemini CLI — no changes to the agent itself:

```python
from ant_ai.acp import ACP_ALL_TOOLS, ACPServer

agent = Agent(..., tools=[*ACP_ALL_TOOLS])  # read files, run terminals, push plans

ACPServer(agent=agent, workflow=workflow).serve_stdio()
```

## Documentation

| Guide | What it covers |
| --- | --- |
| [Installation](https://idea.idsia.ch/ant-ai/docs/install/) | Extras, installing from source, verifying your setup. |
| [Single-agent](https://idea.idsia.ch/ant-ai/docs/single-agent/) | Tools, streaming, memory, skills, and visualizing workflows. |
| [Multi-agent](https://idea.idsia.ch/ant-ai/docs/multi-agent/) | Colonies, agent cards, collaboration edges, deployment. |
| [ACP](https://idea.idsia.ch/ant-ai/docs/acp/) | Serving an agent to an editor, IDE tools, slash commands. |
| [Architecture](https://idea.idsia.ch/ant-ai/docs/architecture/) | How agents, workflows, and events fit together end-to-end. |

Runnable scripts live in [`examples/`](examples/) — an ACP agent for your editor, a stdio→WebSocket proxy, and an interactive ACP test client.

## Development

```sh
# Install dev dependencies and pre-commit hooks
uv sync --all-packages --all-groups --all-extras
uv run pre-commit install

# Run the test suite (skipping tests that need vLLM or external services)
uv run pytest -m "not vllm and not external"

# Serve the docs locally
uv run mkdocs serve
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributing guide, branching model, and review process.

## License

This software is licensed under the MIT license. See the [LICENSE](LICENSE) file for details.

## Citation

If you use `ant-ai` in your research, please cite it. See [CITATION.cff](CITATION.cff) for the machine-readable citation metadata, or use the BibTeX entry below.

```bibtex
@software{Sas_ant-ai_A_lightweight_2026,
author = {Sas, Cezar and Giuffrida, Vincenzo and Mitrović, Sandra and Salani, Matteo},
doi = {10.5281/zenodo.21276625},
license = {MIT},
month = jun,
title = {{ant-ai: A lightweight Python framework for building multi-agent AI systems}},
url = {https://github.com/idea-idsia/ant-ai},
year = {2026}
}
```

## Funding

This project is supported by the following grants.

| Acknowledgement |
| --- |
| Funded by the Swiss State Secretariat for Education, Research and Innovation (SERI), Project number 24.00596. |
| Funded by the European Union under Grant Agreement No. 101189745 (HIVEMIND).<br><img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_funded_en.jpg" alt="Funded by the European Union" height="40"/> |
