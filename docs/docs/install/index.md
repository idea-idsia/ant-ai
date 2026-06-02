---
title: Installation
---

# Installation

`ant-ai` requires **Python 3.14 or later**. [uv](https://docs.astral.sh/uv/) is the recommended package manager.

## From PyPI

```bash
uv add ant-ai
```

### Optional extras

Install all optional extras at once:

```bash
uv add "ant-ai[all]"
```

Or pick individual extras:

| Extra | What it adds |
|-------|-------------|
| `openai` | OpenAI client integration |
| `langfuse` | Observability via [Langfuse](https://langfuse.com/) |
| `mem0` | Long-term memory via [mem0](https://mem0.ai/) |
| `viz` | Workflow graph visualization |

```bash
uv add "ant-ai[openai]"
uv add "ant-ai[langfuse]"
uv add "ant-ai[mem0]"
uv add "ant-ai[viz]"
```

## From the repository

To install directly from source, point `uv` at the Git repository:

```bash
uv add "ant-ai @ git+https://github.com/idea-idsia/ant-ai"
```

## Verifying the installation

```bash
python -c "import ant_ai; print('ok')"
```
