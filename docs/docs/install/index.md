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

`ant-ai` ships an optional extra for the OpenAI client integration:

```bash
uv add "ant-ai[openai]"
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
