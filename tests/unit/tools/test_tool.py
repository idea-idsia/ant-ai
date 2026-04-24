from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from ant_ai.tools.tool import (
    Tool,
    _build_args_model_from_signature,
    _inline_simple_refs_in_schema,
    mcp_tools_from_url,
    tool as tool_decorator,
)

_tool_module = importlib.import_module("ant_ai.tools.tool")


@pytest.mark.unit
def test_inline_simple_refs_in_schema_inlines_defs_ref():
    schema = {
        "type": "object",
        "properties": {
            "color": {"$ref": "#/$defs/Color", "description": "favorite"},
            "x": {"type": "integer"},
        },
        "$defs": {"Color": {"type": "string", "enum": ["red", "blue"]}},
    }

    out = _inline_simple_refs_in_schema(schema)

    assert out["properties"]["color"]["type"] == "string"
    assert out["properties"]["color"]["enum"] == ["red", "blue"]
    assert out["properties"]["color"]["description"] == "favorite"
    assert out["properties"]["x"] == {"type": "integer"}


@pytest.mark.unit
def test_inline_simple_refs_in_schema_noop_when_missing_defs_or_props():
    assert _inline_simple_refs_in_schema({"type": "object"}) == {"type": "object"}
    assert _inline_simple_refs_in_schema({"properties": {}}) == {"properties": {}}
    assert _inline_simple_refs_in_schema({"$defs": {}}) == {"$defs": {}}


@pytest.mark.unit
def test_inline_simple_refs_in_schema_noop_when_not_a_defs_ref():
    schema = {
        "type": "object",
        "properties": {"color": {"$ref": "#/components/schemas/Color"}},
        "$defs": {"Color": {"type": "string", "enum": ["red", "blue"]}},
    }
    out = _inline_simple_refs_in_schema(schema)
    assert out["properties"]["color"]["$ref"] == "#/components/schemas/Color"


@pytest.mark.unit
def test_build_args_model_from_signature_includes_defaults_and_types():
    def f(x: int, y: str = "hi", *, z: float = 1.5) -> None:
        """doc"""

    Args = _build_args_model_from_signature(f, "FArgs")
    schema = Args.model_json_schema()

    props = schema["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
    assert props["z"]["type"] == "number"

    # required should include only x
    assert "required" in schema
    assert schema["required"] == ["x"]


@pytest.mark.unit
def test_build_args_model_from_signature_skips_self_cls_and_varargs_kwargs():
    class C:
        def m(self, x: int, *args: Any, y: int = 3, **kwargs: Any) -> None:
            pass

    Args = _build_args_model_from_signature(C().m, "MArgs")
    schema = Args.model_json_schema()
    assert set(schema["properties"].keys()) == {"x", "y"}


@pytest.mark.unit
def test_tool_defaults_set_on_init():
    class MyTools(Tool):
        """My tool namespace docs."""

        def add(self, x: int, y: int) -> int:
            """Adds two numbers."""
            return x + y

    t = MyTools()

    assert t.name == "MyTools"
    assert t.description == "My tool namespace docs."
    assert t.parameters == {"type": "object", "properties": {}}
    assert t.is_namespace is True
    assert "add" in t.__class__.__namespace_methods__


@pytest.mark.unit
def test_namespace_expand_builds_method_tools_with_names_and_docs():
    class Math(Tool):
        """Math tools."""

        def add(self, x: int, y: int) -> int:
            """Add x and y."""
            return x + y

        def mul(self, x: int, y: int) -> int:
            return x * y

    ns = Math()
    tools = ns._expand_namespace()

    names = {t.name for t in tools}
    assert {"Math_add", "Math_mul"}.issubset(names)

    add_tool = next(t for t in tools if t.name == "Math_add")
    assert "Add x and y." in (add_tool.description or "")
    assert add_tool.invoke(x=2, y=3) == 5

    mul_tool = next(t for t in tools if t.name == "Math_mul")
    assert mul_tool.invoke(x=3, y=4) == 12


@pytest.mark.unit
def test_namespace_tool_cannot_be_invoked_directly():
    class NS(Tool):
        def hello(self) -> str:
            return "hi"

    ns = NS()
    with pytest.raises(RuntimeError, match="Cannot invoke a namespace Tool directly"):
        ns.invoke()


@pytest.mark.unit
def test_tool_from_function_and_call_and_serializer():
    def ping(host: str, n: int = 1) -> str:
        """Ping something."""
        return f"ping:{host}:{n}"

    t = Tool._from_function(ping)

    assert t.name == "ping"
    assert t.description == "Ping something."
    assert t.parameters["type"] == "object"
    assert set(t.parameters["properties"].keys()) == {"host", "n"}

    # __call__ uses signature binding too
    assert t("example.com") == "ping:example.com:1"
    assert t.invoke(host="x", n=2) == "ping:x:2"

    # serializer shape matches OpenAI function tool format
    dumped = t.model_dump()
    assert dumped["type"] == "function"
    assert dumped["function"]["name"] == "ping"
    assert dumped["function"]["parameters"]["type"] == "object"


@pytest.mark.unit
def test_tool_decorator_returns_tool_instance():
    @tool_decorator
    def hello(name: str) -> str:
        return f"hi {name}"

    assert isinstance(hello, Tool)
    assert hello.invoke(name="Ada") == "hi Ada"


@pytest.mark.unit
def test_tool_decorator_custom_name_description_and_args_model():
    class Args(BaseModel):
        x: int

    @tool_decorator(name="custom", description="Custom desc", args_model=Args)
    def fn(x: int) -> int:
        return x + 1

    assert isinstance(fn, Tool)
    assert fn.name == "custom"
    assert fn.description == "Custom desc"
    assert fn.parameters["properties"]["x"]["type"] == "integer"


@pytest.mark.unit
async def test_ainvoke_awaits_async_callable():
    async def af(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    t: Tool = Tool._from_function(af)
    assert await t.ainvoke(x=4) == 5


@pytest.mark.unit
async def test_ainvoke_runs_sync_callable_in_executor(monkeypatch):
    # make run_in_executor deterministic by intercepting it
    class DummyLoop:
        def __init__(self):
            self.called = False

        async def run_in_executor(self, executor, func):
            self.called = True
            return func()

    dummy_loop = DummyLoop()
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: dummy_loop)

    def sf(x: int) -> int:
        return x * 2

    t: Tool = Tool._from_function(sf)
    out = await t.ainvoke(x=3)

    assert out == 6
    assert dummy_loop.called is True


@pytest.mark.unit
async def test_invoke_returns_coroutine_for_async_tool():
    async def af(x: int) -> int:
        return x + 1

    t: Tool = Tool._from_function(af)
    res = t.invoke(x=1)
    assert inspect.isawaitable(res)
    await res


@pytest.mark.unit
async def test_from_mcp_descriptor_unwraps_structured_content(monkeypatch):
    # fake objects to mimic mcp tool + result
    class FakeMcpTool:
        name = "weather"
        description = "desc"
        inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}

    class FakeResult:
        structuredContent = {"ok": True}
        content = None

    # patch streamablehttp_client and MCPClientSession so no IO happens
    class DummyClientCtx:
        async def __aenter__(self):
            # read/write placeholders
            return object(), object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name: str, arguments: dict[str, Any]):
            assert name == "weather"
            assert arguments == {"q": "x"}
            return FakeResult()

    monkeypatch.setattr(
        _tool_module, "streamable_http_client", lambda url: DummyClientCtx()
    )
    monkeypatch.setattr(
        _tool_module, "MCPClientSession", lambda read, write: DummySession()
    )

    t: Tool = Tool._from_mcp_descriptor(
        url="http://x", mcp_tool=FakeMcpTool(), namespace="ns"
    )
    assert t.name == "ns.weather"
    assert t.parameters["properties"]["q"]["type"] == "string"

    out = await t.ainvoke(q="x")
    assert out == {"ok": True}


@pytest.mark.unit
async def test_mcp_tools_from_url_builds_tools_list(monkeypatch):
    class FakeMcpTool:
        def __init__(self, name: str):
            self.name = name
            self.description = f"desc:{name}"
            self.inputSchema = {"type": "object", "properties": {}}

    class FakeListToolsResult:
        def __init__(self):
            self.tools = [FakeMcpTool("t1"), FakeMcpTool("t2")]

    class DummyClientCtx:
        async def __aenter__(self):
            return object(), object(), object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return FakeListToolsResult()

    monkeypatch.setattr(
        _tool_module, "streamable_http_client", lambda url: DummyClientCtx()
    )
    monkeypatch.setattr(
        _tool_module, "MCPClientSession", lambda read, write: DummySession()
    )

    tools: list[Tool] = await mcp_tools_from_url("http://x", namespace="remote")
    assert [t.name for t in tools] == ["remote.t1", "remote.t2"]
    assert [t.description for t in tools] == ["desc:t1", "desc:t2"]
