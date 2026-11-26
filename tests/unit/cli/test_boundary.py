from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ant_ai

SRC = Path(ant_ai.__file__).parent
CLI = SRC / "cli"


def _module(path: Path) -> str:
    """The dotted name of a source file, e.g. `ant_ai.cli.app`."""
    parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> set[str]:
    """Every name `path` imports, in full: `from ant_ai import cli` gives
    `ant_ai.cli`, and a relative import is resolved against the file."""
    package = _module(path)
    if path.name != "__init__.py":
        package = package.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")
            if node.level:
                base = base[: len(base) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.unit
def test_the_import_check_sees_names_relative_imports_and_private_members():
    names = _imports(SRC / "__init__.py")
    assert all(not n.startswith(".") for n in names)
    cli = _imports(CLI / "app.py")
    assert "ant_ai.cli.history" in cli
    assert "ant_ai.agent.agent.Agent" in cli


@pytest.mark.unit
def test_the_library_does_not_depend_on_the_cli():
    """`ant_ai.cli` can be moved to its own package without touching the library."""
    users = [
        path.relative_to(SRC)
        for path in SRC.rglob("*.py")
        if CLI not in path.parents
        and any(name.startswith("ant_ai.cli") for name in _imports(path))
    ]
    assert users == []


@pytest.mark.unit
def test_the_cli_uses_only_public_names_of_the_library():
    private = [
        f"{path.name}: {name}"
        for path in CLI.glob("*.py")
        for name in _imports(path)
        if name.startswith("ant_ai.")
        and not name.startswith("ant_ai.cli")
        and any(part.startswith("_") for part in name.split("."))
    ]
    assert private == []
