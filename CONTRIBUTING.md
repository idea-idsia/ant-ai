# Contributing to ANT AI

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Branching Model](#branching-model)
- [Issue Tracking](#issue-tracking)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Requests & Review](#pull-requests--review)
- [Releases](#releases)
- [Commit Messages](#commit-messages)

---

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

```sh
# Install all dependencies including dev and extras
uv sync --all-extras

# Install pre-commit hooks (required for all contributors)
uv run pre-commit install
```

Pre-commit hooks run automatically on every commit and enforce linting, formatting, secret detection, and file hygiene.

---

## Branching Model

The repository follows a trunk-based model: all work branches off `main` and merges back via pull request.

### Main branch

| Branch | Description |
|--------|-------------|
| `main` | Stable, production-ready code. Protected and version-tagged. |

### Support branches (short-lived)

Branch prefixes mirror [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) types so that branch names and commit messages stay consistent.

| Branch | Origin | Description |
|--------|--------|-------------|
| `feat/*` | `main` | New features. |
| `fix/*` | `main` | Bug fixes. |
| `docs/*` | `main` | Documentation-only changes. |
| `refactor/*` | `main` | Code restructuring with no behaviour change. |
| `test/*` | `main` | Test additions or corrections. |
| `chore/*` | `main` | Maintenance tasks (deps, config, tooling). |
| `perf/*` | `main` | Performance improvements. |
| `ci/*` | `main` | CI/CD pipeline changes. |
| `junk/*` | `main` | Experimental work. **Never merged.** |

### Workflow

```
main ──────────────────────────────────────────────────── (tagged releases)
  │         ↑ all branches merge back via PR
  ├── feat/*
  ├── fix/*
  ├── docs/*
  ├── refactor/*
  ├── test/*
  ├── chore/*
  ├── perf/*
  ├── ci/*
  └── junk/*  (dead end — never merged)
```

Urgent production fixes use a `fix/` branch — urgency is expressed in the commit message with `!` (e.g. `fix!: correct token expiry calculation`) rather than a separate branch prefix.

**Starting work:**

```sh
git checkout main
git pull
git checkout -b feat/[<issue-number>-]<short-description>   # replace feat/ with the relevant type
```

**Branch naming examples:**

Include the issue number before the description — GitHub will automatically link the branch to the issue.

- `feat/42-tool-retry-policy`
- `fix/17-agent-loop-timeout`
- `docs/31-workflow-sequencing`
- `refactor/8-simplify-agent-loop`
- `ci/15-github-actions`
- `junk/explore-streaming-api`

---

## Issue Tracking

Use the GitHub issue templates — they enforce the required fields and apply the correct label automatically:

| Template | CC label | Use when |
|----------|----------|----------|
| Bug report | `fix` | Something isn't working correctly |
| Feature request | `feat` | Proposing new functionality |
| Documentation | `docs` | Missing, outdated, or incorrect docs |
| Task | `refactor` / `test` / `chore` / `perf` / `ci` | Maintenance or internal work |
| Change request | `change-request` | Significant change needing approval first |

Issue titles are pre-filled with the matching CC prefix (e.g. `fix: `, `feat: `) — keep that prefix so the title can be reused directly in a commit message.

**Significant changes** (architectural, cross-cutting, or with rollback risk) must go through a **change request** before any implementation begins. Once approved, one or more issues are opened to track the work.

### Priority labels

Every issue must carry exactly one priority label — set it after the template auto-applies the type label:

| Label | When to use |
|-------|-------------|
| `priority:critical` | Production is broken |
| `priority:high` | Major functionality affected |
| `priority:medium` | Workaround exists |
| `priority:low` | Cosmetic or minor impact |

---

## Coding Standards

All code must pass the following checks (enforced by CI and pre-commit):

### Linting & Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```sh
uv run ruff check .   # lint
uv run ruff format .  # format
```

Configuration lives in [ruff.toml](ruff.toml). Auto-fixes are applied by pre-commit — run `git add -p` after a failed commit to review them.

### Type Checking

We use [ty](https://github.com/astral-sh/ty) for static type analysis. All public APIs must be fully typed.

```sh
uv run ty check
```

### Key style rules

- Double quotes for strings.
- 4-space indentation.
- Prefer comprehensions over imperative equivalents.
- Keep functions small and single-purpose.
- No commented-out code or debug prints in merged branches.

---

## Testing

Tests live in the [tests/](tests/) directory and use [pytest](https://docs.pytest.org/).

```sh
# Run the full test suite (excluding slow/external tests)
uv run pytest -m "not vllm and not external" --cov --cov-report=term

# Run a single file
uv run pytest tests/test_agents.py -v

# Run tests matching a keyword
uv run pytest -k "tool" -v
```

### Guidelines

- Every new feature or bug fix must include tests.
- Aim for meaningful coverage — prefer testing behaviour over implementation.
- Tag tests that require external services with `@pytest.mark.external` and vLLM-backed tests with `@pytest.mark.vllm`.

---

## Documentation

Public APIs are documented via docstrings and rendered with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Configuration is in [mkdocs.yml](mkdocs.yml).

```sh
uv run mkdocs serve   # live preview
uv run mkdocs build   # static build
```

- All public classes, methods, and functions must have docstrings.
- Follow [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings.
- Update the relevant page in [docs/](docs/) when introducing a new concept or API.

---

## Pull Requests & Review

### Submitting

1. Open the PR — GitHub will auto-populate the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
2. Fill in every section completely — incomplete PRs will be returned.
3. Tick the correct **commit type** and apply the appropriate **bump label** (or confirm none is needed).
4. Ensure all CI jobs pass (`lint`, `type-check`, `run-pytest`).
5. Keep PRs focused: one concern per PR.
6. Request at least **one review** from a project maintainer.
7. Do not force-push once a review has started — add new commits instead.

### Review checklist

Reviewers apply the following criteria to every PR:

- [ ] **Clarity** — code is readable and intent is clear without needing inline comments
- [ ] **Coding conventions** — passes lint, format, and type checks; follows project style
- [ ] **Security impact** — no introduced vulnerabilities; no secrets or credentials committed
- [ ] **Test coverage** — new behaviour is covered; existing tests are not broken
- [ ] **Side effects** — no unintended impact on other components or APIs
- [ ] **Documentation** — public API changes are reflected in docstrings and/or docs pages
- [ ] **Branch target** — PR targets `main`

---

## Releases

Releases are fully automated and triggered by a label on the merged PR.

### Labels

Apply exactly one of these labels before merging:

| Label | Semver segment bumped | When to use |
|-------|-----------------------|-------------|
| `bump:patch` | `x.y.Z` | Backwards-compatible bug fixes |
| `bump:minor` | `x.Y.0` | New backwards-compatible functionality |
| `bump:major` | `X.0.0` | Breaking changes |

PRs without a bump label are merged silently — no release is cut.

### What happens on merge

```
PR merged with bump:* label
        ↓
release.yml — bumps version in pyproject.toml + uv.lock, commits to main, creates git tag
        ↓
publish.yml — triggered by the new GitHub Release → builds → TestPyPI → PyPI
```

The version bump commit uses the message `chore(release): bump version to vX.Y.Z` and is pushed directly to `main` by the `github-actions[bot]`.

### Label management

Labels are defined in [.github/labels.yml](.github/labels.yml) and synced to GitHub automatically on every change to that file. To bootstrap them on a fresh fork, run the **Sync labels** workflow manually from the Actions tab.

---

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `ci`

**Examples:**

```
feat(tools): add retry policy with exponential backoff

fix(agent): prevent infinite loop when tool raises ValueError

fix(auth)!: correct token expiry calculation

docs(workflow): document sequencing behaviour for parallel steps
```

Rules:
- Subject line ≤ 72 characters, imperative mood, no trailing period.
- Use `!` after the type to signal urgent or breaking changes (e.g. `fix!:`, `feat!:`).
- Reference related issues and PRs in the footer using `#<number>`.
  - **Auto-closes** the issue when the PR merges: `Closes #42`, `Fixes #42`, `Resolves #42`
  - **Mentions** without closing: `See #42`, `Related to #42`, `Part of #42`
