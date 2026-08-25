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
| `guardrails-ai` | Guardrail hooks via [Guardrails AI](https://www.guardrailsai.com/) — see [below](#guardrails-ai-extra) for the extra hub-install step |
| `datafog` | `PIIGuardrailHook`, a native PII-scanning guardrail hook — no hub install needed |

```bash
uv add "ant-ai[openai]"
uv add "ant-ai[langfuse]"
uv add "ant-ai[mem0]"
uv add "ant-ai[viz]"
uv add "ant-ai[guardrails-ai]"
uv add "ant-ai[datafog]"
```

### Guardrails AI extra

The `guardrails-ai` extra installs the core library. Validators (e.g. `ToxicLanguage`, `DetectPII`) are distributed via the [Guardrails Hub](https://hub.guardrailsai.com/) and must be installed separately after running `guardrails configure`:

```sh
# Run from outside the project directory so uv doesn't pick up exclude-newer
(cd /tmp && guardrails hub install hub://guardrails/toxic_language hub://guardrails/detect_pii)
# Copy the registry into the project root so imports resolve from here
cp /tmp/.guardrails/hub_registry.json .guardrails/
```

> **Note:** hub validators are installed outside of uv's lockfile and must be reinstalled after `uv sync`.
> The `cd /tmp` wrapper is required because these packages ship without upload dates, which conflicts with the project's `exclude-newer = "P2D"` supply-chain setting. Running from `/tmp` (where no `pyproject.toml` exists) lets uv install them without that constraint while still targeting the active virtualenv.
> The `hub_registry.json` copy is needed because guardrails resolves hub imports from a `.guardrails/hub_registry.json` file relative to the current working directory.

The `datafog` extra powers `PIIGuardrailHook`, a native, dependency-light guardrail hook that scans LLM output for PII and retries/blocks on detection — an alternative to `GuardrailsAIHook` + `DetectPII` that needs no separate hub install. For guardrailing on arbitrary criteria (e.g. "no medical advice"), subclass `LLMGuardrailHook` — an LLM-as-judge base class that needs no extra dependency.

## From the repository

To install directly from source, point `uv` at the Git repository:

```bash
uv add "ant-ai @ git+https://github.com/idea-idsia/ant-ai"
```

## Verifying the installation

```bash
python -c "import ant_ai; print('ok')"
```
