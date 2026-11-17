from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ant_ai

SRC = Path(ant_ai.__file__).parent
CLI = SRC / "cli"


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


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
