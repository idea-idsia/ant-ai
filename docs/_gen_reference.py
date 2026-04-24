from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from types import ModuleType
from typing import LiteralString

import mkdocs_gen_files

PACKAGE = "ant_ai"
OUTPUT_DIR = "reference"
TITLE = "Overview"
EXCLUDE_PRIVATE = True


@dataclass(frozen=True)
class Mod:
    name: str  # e.g. ant_ai.llm.integrations.openai_llm
    is_pkg: bool  # package vs module (folder index vs .md)


def iter_modules(package: str) -> list[Mod]:
    """Return all importable modules/packages inside PACKAGE (excluding private)."""
    pkg: ModuleType = importlib.import_module(package)
    prefix: str = pkg.__name__ + "."

    out: list[Mod] = []
    for m in pkgutil.walk_packages(pkg.__path__, prefix):
        name: str = m.name
        is_pkg: bool = bool(getattr(m, "ispkg", False))

        if EXCLUDE_PRIVATE:
            parts = name.split(".")
            if any(p.startswith("_") for p in parts[1:]):
                continue

        out.append(Mod(name=name, is_pkg=is_pkg))

    # Sort packages before modules for nicer index rendering
    return sorted(out, key=lambda x: (x.name, not x.is_pkg))


def rel_parts(full_name: str) -> list[str]:
    """Drop root package (ant_ai) => ['llm','integrations','openai_llm']"""
    parts: list[str] = full_name.split(".")
    return parts[1:] if parts and parts[0] == PACKAGE else parts


def doc_path(mod: Mod) -> str:
    """Map a module/package to a doc path under docs/."""
    parts: list[str] = rel_parts(mod.name)
    if mod.is_pkg:
        return f"{OUTPUT_DIR}/" + "/".join(parts) + "/index.md"
    return f"{OUTPUT_DIR}/" + "/".join(parts) + ".md"


def edit_path(mod: Mod) -> str:
    """Map a module/package to a source edit path under src/."""
    parts: list[str] = rel_parts(mod.name)
    if mod.is_pkg:
        return "src/" + PACKAGE + "/" + "/".join(parts) + "/__init__.py"
    return "src/" + PACKAGE + "/" + "/".join(parts) + ".py"


def build_tree(mods: list[Mod]) -> dict[str, dict]:
    """
    Build a tree keyed by segment names, storing the Mod at each node as:
      node["__mod__"] = {"mod": Mod(...)}
    """
    root: dict[str, dict] = {}

    for mod in mods:
        parts: list[str] = rel_parts(mod.name)
        if not parts:
            continue

        node: dict[str, dict] = root
        for i, seg in enumerate(parts):
            node: dict = node.setdefault(seg, {})
            if i == len(parts) - 1:
                node["__mod__"] = {"mod": mod}

    return root


def write_index_tree(f, node: dict[str, dict], level: int = 0) -> None:
    """
    Render nested bullets in reference/index.md.

    IMPORTANT: Nested lists must be indented 4 spaces per level for reliable parsing.
    Links are written relative to reference/index.md.
    """
    for seg in sorted(k for k in node if k != "__mod__"):
        child: dict = node[seg]
        mod_entry = child.get("__mod__", {}).get("mod")

        indent: LiteralString = "    " * level  # 4 spaces per nesting level
        if mod_entry is not None:
            rel_link: str = doc_path(mod_entry).replace(OUTPUT_DIR + "/", "")
            f.write(f"{indent}- [`{seg}`]({rel_link})\n")
        else:
            f.write(f"{indent}- `{seg}`\n")

        write_index_tree(f, child, level + 1)


def write_package_children(f, package_mod: Mod, tree: dict[str, dict]) -> None:
    """
    On a package index page (e.g. reference/a2a/index.md), list immediate children:
      - subpackages (link to .../index.md)
      - modules (link to .../.md)
    """
    parts: list[str] = rel_parts(package_mod.name)
    node = tree
    for seg in parts:
        node = node.get(seg, {})
    # Now node is the subtree for this package

    children: list[str] = sorted(k for k in node if k != "__mod__")
    if not children:
        f.write("_No public subpackages or modules found._\n")
        return

    f.write("## Contents\n\n")
    for seg in children:
        child = node[seg]
        mod_entry = child.get("__mod__", {}).get("mod")
        if mod_entry is None:
            continue

        full_doc: str = doc_path(mod_entry)  # starts with "reference/..."
        rel_from_reference_root: str = full_doc.replace(OUTPUT_DIR + "/", "")

        pkg_prefix: str = "/".join(parts) + "/"
        if rel_from_reference_root.startswith(pkg_prefix):
            rel_link: str = rel_from_reference_root[len(pkg_prefix) :]
        else:
            rel_link: str = rel_from_reference_root

        f.write(f"- [`{seg}`]({rel_link})\n")


def main() -> None:
    mods: list[Mod] = iter_modules(PACKAGE)
    tree: dict[str, dict] = build_tree(mods)

    # Reference landing page with nested structure
    index_path = f"{OUTPUT_DIR}/index.md"
    with mkdocs_gen_files.open(index_path, "w") as f:
        f.write("---\n")
        f.write("title: Reference\n")
        f.write("---\n")
        f.write(f"# {TITLE}\n\n")
        f.write(f"API reference for `{PACKAGE}`.\n\n")
        f.write("## Packages and modules\n\n")
        write_index_tree(f, tree, level=0)

    # One page per package/module
    for mod in mods:
        p: str = doc_path(mod)

        with mkdocs_gen_files.open(p, "w") as f:
            full = mod.name
            f.write(f"# `{full}`\n\n")

            if mod.is_pkg:
                # Package pages are navigational (avoid empty mkdocstrings package output)
                write_package_children(f, mod, tree)
            else:
                # Modules use mkdocstrings
                f.write(f"::: {full}\n")

        mkdocs_gen_files.set_edit_path(p, edit_path(mod))


main()
