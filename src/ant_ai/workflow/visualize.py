from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Literal

from graphviz import Digraph

from ant_ai.workflow.workflow import END, START, RouterAction, Workflow

_FORMATS = ("png", "jpg", "pdf", "svg", "latex")

_FILL_SPECIAL = "#d0e8ff"  # START / END
_FILL_NODE = "#ffffff"  # regular nodes
_FILL_ROUTER = "#fff3cd"  # decision diamonds


def _gv_id(name: str) -> str:
    """Stable single-token graphviz node id (no spaces)."""
    return name.replace(" ", "_").replace("-", "_")


def _router_destinations(router: RouterAction) -> list[str]:
    """Return every literal string a router function can return, via AST.

    Uses a depth-first visitor so results appear in source-code order.
    Only direct string literals in ``return`` statements are collected;
    variable references (e.g. ``return END``) are intentionally ignored.
    """
    try:
        src = textwrap.dedent(inspect.getsource(router))
    except OSError:
        return []
    tree = ast.parse(src)

    class _Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.results: list[str] = []

        def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                self.results.append(node.value.value)
            self.generic_visit(node)

    collector = _Collector()
    collector.visit(tree)
    return list(dict.fromkeys(collector.results))  # deduplicate, preserve order


def build_workflow_graph(
    workflow: Workflow,
    *,
    engine: str = "dot",
    rankdir: str = "LR",
) -> Digraph:
    """
    Build and return a ``graphviz.Digraph`` for *workflow*.

    The returned object renders inline in Jupyter notebooks and can be passed
    to :func:`render_workflow` for file export.

    Args:
        workflow: The workflow to visualise.
        engine: Graphviz layout engine (``"dot"``, ``"neato"``, ``"fdp"``…).
        rankdir: Layout direction — ``"LR"`` (left-to-right, default) or ``"TB"``
            (left-to-right).

    Raises:
        ImportError: if the ``graphviz`` package is not installed.
            Install with ``pip install ant-ai[viz]``.
    """
    try:
        import graphviz
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Workflow visualisation requires the graphviz package. "
            "Install it with:  pip install ant-ai[viz]"
        ) from exc

    g = graphviz.Digraph(engine=engine)
    g.attr(rankdir=rankdir, fontname="Helvetica", fontsize="12")
    g.attr("node", fontname="Helvetica", fontsize="12")
    g.attr("edge", fontname="Helvetica", fontsize="10")

    # START / END — ellipse
    for special in (START, END):
        g.node(
            _gv_id(special),
            label=special,
            shape="ellipse",
            style="filled",
            fillcolor=_FILL_SPECIAL,
        )

    # Regular nodes — rectangle
    for name in workflow.nodes:
        g.node(
            _gv_id(name),
            label=name,
            shape="rectangle",
            style="filled,rounded",
            fillcolor=_FILL_NODE,
        )

    # Static edges
    for src, dst in workflow.edges.items():
        g.edge(_gv_id(src), _gv_id(dst))

    # Conditional edges — diamond + AST-extracted destinations
    for src, router in workflow.conditional_edges.items():
        diamond_id = f"__router_{_gv_id(src)}__"
        g.node(
            diamond_id,
            label=getattr(router, "__name__", repr(router)),
            shape="diamond",
            style="filled",
            fillcolor=_FILL_ROUTER,
        )
        g.edge(_gv_id(src), diamond_id)
        for dst in _router_destinations(router):
            g.edge(diamond_id, _gv_id(dst), style="dashed")

    return g


_TIKZ_PREAMBLE = r"""\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}
\begin{document}"""

_TIKZ_EPILOGUE = r"\end{document}"

# Points-per-inch in graphviz plain output; we convert to cm.
_PT_PER_IN = 72.0
_CM_PER_IN = 2.54
# Extra visual scaling so the diagram isn't tiny.
_VIS_SCALE = 1.8


def _pts_to_cm(value: float) -> float:
    return value / _PT_PER_IN * _CM_PER_IN * _VIS_SCALE


def _parse_plain(plain: str) -> tuple[dict[str, dict], list[dict]]:
    """Parse graphviz ``plain`` output into node/edge dicts."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for raw_line in plain.splitlines():
        parts = raw_line.split()
        if not parts:
            continue

        kind = parts[0]

        if kind == "node":
            # node name x y width height label style shape color fillcolor
            # shape is at -3, color at -2, fillcolor at -1
            name = parts[1]
            x = _pts_to_cm(float(parts[2]))
            y = _pts_to_cm(float(parts[3]))
            w = _pts_to_cm(float(parts[4]))
            h = _pts_to_cm(float(parts[5]))
            label = parts[6].strip(
                '"'
            )  # single-word labels; strip any graphviz quoting
            shape = parts[-3]
            nodes[name] = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
                "shape": shape,
            }

        elif kind == "edge":
            # edge tail head n x1 y1 ... [label xl yl] style color
            tail = parts[1]
            head = parts[2]
            n = int(parts[3])
            remaining = parts[4 + 2 * n :]
            style = remaining[0] if remaining else "solid"
            edges.append({"tail": tail, "head": head, "dashed": "dashed" in style})

    return nodes, edges


def _to_tikz_shape(gv_shape: str) -> str:
    if gv_shape == "ellipse":
        return "ellipse"
    if gv_shape == "diamond":
        return "diamond, aspect=2"
    return "rectangle, rounded corners=3pt"


def _to_tikz_fill(gv_shape: str) -> str:
    if gv_shape == "ellipse":
        return "blue!10"
    if gv_shape == "diamond":
        return "yellow!30"
    return "white"


def _build_tikz(workflow: Workflow, engine: str) -> str:
    """Generate a self-contained TikZ/LaTeX document for *workflow*."""
    g = build_workflow_graph(workflow, engine=engine)
    plain = g.pipe(format="plain").decode()
    nodes, edges = _parse_plain(plain)

    tikz_nodes = [
        f"  \\node[draw, {_to_tikz_shape(info['shape'])}, fill={_to_tikz_fill(info['shape'])}, "
        f"minimum width={max(info['w'], 1.2):.2f}cm, minimum height={max(info['h'], 0.5):.2f}cm, align=center] "
        f"({name}) at ({info['x']:.2f},{info['y']:.2f}) {{{info['label'].replace('_', r'\_')}}};"
        for name, info in nodes.items()
    ]

    tikz_edges = [
        f"  \\draw[->, {'dashed, ' if e['dashed'] else ''}>=Stealth] ({e['tail']}) -- ({e['head']});"
        for e in edges
    ]

    body = "\n".join(
        [
            r"\begin{tikzpicture}[",
            r"  every node/.style={font=\small},",
            r"]",
            *tikz_edges,
            *tikz_nodes,
            r"\end{tikzpicture}",
        ]
    )

    return "\n".join([_TIKZ_PREAMBLE, body, _TIKZ_EPILOGUE])


def render_workflow(
    workflow: Workflow,
    path: str | Path,
    format: Literal["png", "jpg", "pdf", "svg", "latex"] = "png",
    *,
    engine: str = "dot",
    rankdir: str = "LR",
) -> Path:
    """
    Render *workflow* to a file.

    Args:
        workflow: The workflow to visualise.
        path: Output path. The file extension is set automatically based on
            *format* (any existing extension is replaced).
        format: ``"png"``, ``"jpg"``, ``"pdf"``, ``"svg"``, or ``"latex"``.
            The ``"latex"`` format produces a self-contained ``.tex`` file
            that can be compiled with ``pdflatex``.
        engine: Graphviz layout engine.
        rankdir: Layout direction — ``"LR"`` (left-to-right, default) or ``"TB"``.

    Returns:
        :class:`~pathlib.Path` of the rendered file.

    Raises:
        ImportError: if the ``graphviz`` package is not installed.
        ValueError: if *format* is not one of the supported values.
    """
    if format not in _FORMATS:
        raise ValueError(f"format must be one of {_FORMATS!r}, got {format!r}")

    out = Path(path)

    if format == "latex":
        tex = _build_tikz(workflow, engine=engine)
        dest = out.with_suffix(".tex")
        dest.write_text(tex, encoding="utf-8")
        return dest

    g: Digraph = build_workflow_graph(workflow, engine=engine, rankdir=rankdir)
    gv_format: Literal["jpeg", "png", "pdf", "svg"] = (
        "jpeg" if format == "jpg" else format
    )
    rendered = g.render(
        filename=str(out.with_suffix("")),
        format=gv_format,
        cleanup=True,
    )
    return Path(rendered)
