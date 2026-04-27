from __future__ import annotations

import sys

import pytest

from ant_ai.workflow.visualize import (
    _gv_id,
    _router_destinations,
    build_workflow_graph,
    render_workflow,
)
from ant_ai.workflow.workflow import Workflow


@pytest.mark.unit
def test_gv_id_passthrough():
    assert _gv_id("simple") == "simple"


@pytest.mark.unit
def test_gv_id_replaces_spaces():
    assert _gv_id("my node") == "my_node"


@pytest.mark.unit
def test_gv_id_replaces_hyphens():
    assert _gv_id("node-a") == "node_a"


@pytest.mark.unit
def test_gv_id_replaces_mixed():
    assert _gv_id("my-cool node") == "my_cool_node"


@pytest.mark.unit
def test_router_destinations_extracts_literals():
    def router(a, state, ctx):
        if True:
            return "node_b"
        return "node_c"

    dests = _router_destinations(router)
    assert dests == ["node_b", "node_c"]


@pytest.mark.unit
def test_router_destinations_deduplicates():
    def router(a, state, ctx):
        if True:
            return "node_b"
        return "node_b"

    dests = _router_destinations(router)
    assert dests == ["node_b"]


@pytest.mark.unit
def test_router_destinations_preserves_order():
    def router(a, state, ctx):
        if True:
            return "z_node"
        elif False:
            return "a_node"
        return "END"  # literal string, not the END variable

    dests = _router_destinations(router)
    assert dests == ["z_node", "a_node", "END"]


@pytest.mark.unit
def test_router_destinations_ignores_non_literal_returns():
    def router(a, state, ctx):
        result = "node_b"
        return result  # variable, not literal

    dests = _router_destinations(router)
    assert dests == []


@pytest.mark.unit
def test_router_destinations_returns_empty_on_no_source(monkeypatch):
    import inspect

    monkeypatch.setattr(
        inspect, "getsource", lambda _: (_ for _ in ()).throw(OSError())
    )

    def router(a, state, ctx):
        return "node_b"

    assert _router_destinations(router) == []


@pytest.mark.unit
def test_build_workflow_graph_raises_without_graphviz(monkeypatch):
    monkeypatch.setitem(sys.modules, "graphviz", None)
    with pytest.raises(ImportError, match="pip install ant-ai"):
        build_workflow_graph(Workflow())


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_dot_source_contains_start_end(noop_workflow):
    src = build_workflow_graph(noop_workflow).source
    assert "START" in src
    assert "END" in src


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_dot_source_contains_nodes(noop_workflow):
    src = build_workflow_graph(noop_workflow).source
    assert "A" in src


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_start_end_are_ellipses(noop_workflow):
    src = build_workflow_graph(noop_workflow).source
    assert src.count("ellipse") >= 2


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_regular_node_is_rectangle(noop_workflow):
    assert "rectangle" in build_workflow_graph(noop_workflow).source


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_conditional_creates_diamond(conditional_workflow):
    assert "diamond" in build_workflow_graph(conditional_workflow).source


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_conditional_dashed_edge(conditional_workflow):
    assert "dashed" in build_workflow_graph(conditional_workflow).source


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_router_label_in_source(conditional_workflow):
    assert "my_router" in build_workflow_graph(conditional_workflow).source


@pytest.mark.unit
@pytest.mark.graphviz
def test_build_workflow_graph_lr_rankdir(noop_workflow):
    assert "LR" in build_workflow_graph(noop_workflow, rankdir="LR").source


@pytest.mark.unit
def test_render_workflow_rejects_invalid_format(noop_workflow):
    with pytest.raises(ValueError, match="format must be one of"):
        render_workflow(noop_workflow, "/tmp/wf", format="bmp")  # type: ignore[arg-type]
