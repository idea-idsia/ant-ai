from __future__ import annotations

from pathlib import Path

import pytest

graphviz = pytest.importorskip("graphviz", reason="graphviz package not installed")

from ant_ai.workflow.visualize import (  # noqa: E402
    _parse_plain,
    build_workflow_graph,
    render_workflow,
)
from ant_ai.workflow.workflow import Workflow  # noqa: E402


@pytest.mark.integration
@pytest.mark.graphviz
def test_build_workflow_graph_renders_svg(noop_workflow: Workflow):
    svg = build_workflow_graph(noop_workflow).pipe(format="svg").decode()
    assert "<svg" in svg
    assert "START" in svg
    assert "END" in svg
    assert "A" in svg


@pytest.mark.integration
@pytest.mark.graphviz
def test_build_workflow_graph_conditional_renders_svg(conditional_workflow: Workflow):
    svg = build_workflow_graph(conditional_workflow).pipe(format="svg").decode()
    assert "my_router" in svg
    assert "diamond" in svg.lower() or "polygon" in svg


@pytest.mark.integration
@pytest.mark.graphviz
def test_parse_plain_extracts_node_positions(noop_workflow: Workflow):
    plain = build_workflow_graph(noop_workflow).pipe(format="plain").decode()
    nodes, _ = _parse_plain(plain)

    assert "START" in nodes
    assert "END" in nodes
    assert "A" in nodes

    for info in nodes.values():
        assert isinstance(info["x"], float)
        assert isinstance(info["y"], float)
        assert info["w"] > 0
        assert info["h"] > 0


@pytest.mark.integration
@pytest.mark.graphviz
def test_parse_plain_extracts_node_labels(noop_workflow: Workflow):
    plain = build_workflow_graph(noop_workflow).pipe(format="plain").decode()
    nodes, _ = _parse_plain(plain)
    assert nodes["START"]["label"] == "START"
    assert nodes["END"]["label"] == "END"
    assert nodes["A"]["label"] == "A"


@pytest.mark.integration
@pytest.mark.graphviz
def test_parse_plain_extracts_node_shapes(noop_workflow: Workflow):
    plain = build_workflow_graph(noop_workflow).pipe(format="plain").decode()
    nodes, _ = _parse_plain(plain)
    assert nodes["START"]["shape"] == "ellipse"
    assert nodes["END"]["shape"] == "ellipse"
    assert nodes["A"]["shape"] == "rectangle"


@pytest.mark.integration
@pytest.mark.graphviz
def test_parse_plain_extracts_edges(noop_workflow: Workflow):
    plain = build_workflow_graph(noop_workflow).pipe(format="plain").decode()
    _, edges = _parse_plain(plain)

    pairs = {(e["tail"], e["head"]) for e in edges}
    assert ("START", "A") in pairs
    assert ("A", "END") in pairs


@pytest.mark.integration
@pytest.mark.graphviz
def test_parse_plain_marks_dashed_edges_for_conditional(conditional_workflow: Workflow):
    plain = build_workflow_graph(conditional_workflow).pipe(format="plain").decode()
    _, edges = _parse_plain(plain)

    assert any(e["dashed"] for e in edges), (
        "Expected at least one dashed edge from the router"
    )


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_png(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf", format="png")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_svg(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf", format="svg")
    assert out.exists()
    assert "<svg" in out.read_text()


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_pdf(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf", format="pdf")
    assert out.exists()
    assert out.suffix == ".pdf"
    assert out.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_jpg(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf", format="jpg")
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_latex_writes_tex(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf", format="latex")
    assert out.suffix == ".tex"
    tex = out.read_text()
    assert r"\documentclass{standalone}" in tex
    assert r"\usepackage{tikz}" in tex
    assert r"\begin{tikzpicture}" in tex
    assert r"\end{tikzpicture}" in tex
    assert r"\node" in tex
    assert r"\draw" in tex


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_latex_contains_all_nodes(
    noop_workflow: Workflow, tmp_path: Path
):
    tex = render_workflow(noop_workflow, tmp_path / "wf", format="latex").read_text()
    assert "START" in tex
    assert "END" in tex
    assert "A" in tex


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_latex_correct_shapes(noop_workflow: Workflow, tmp_path: Path):
    tex = render_workflow(noop_workflow, tmp_path / "wf", format="latex").read_text()
    assert "ellipse" in tex
    assert "rectangle" in tex


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_latex_conditional(
    conditional_workflow: Workflow, tmp_path: Path
):
    tex = render_workflow(
        conditional_workflow, tmp_path / "wf_cond", format="latex"
    ).read_text()
    assert "dashed" in tex
    assert "my\\_router" in tex
    assert "diamond" in tex


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_respects_lr_rankdir(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf_lr", format="svg", rankdir="LR")
    assert "<svg" in out.read_text()


@pytest.mark.integration
@pytest.mark.graphviz
def test_render_workflow_extension_replaced(noop_workflow: Workflow, tmp_path: Path):
    out = render_workflow(noop_workflow, tmp_path / "wf.txt", format="png")
    assert out.suffix == ".png"
