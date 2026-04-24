---
title: Installation
---

# Installation

ant_ai requires **Python 3.13 or later**. [uv](https://docs.astral.sh/uv/) is the recommended package manager.

## From PyPI

```bash
uv add ant_ai
```

### Optional extras

ant_ai ships an optional extra for the OpenAI client integration:

```bash
uv add "ant_ai[openai]"
```

## From the repository

To install directly from source, point `uv` at the Git repository:

```bash
uv add "ant_ai @ git+https://github.com/TBD/ant_ai"
```

## Verifying the installation

```bash
python -c "import ant_ai; print('ok')"
```
