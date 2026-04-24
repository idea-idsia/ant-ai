from __future__ import annotations

import asyncio
import functools
import inspect
import typing
from asyncio import AbstractEventLoop
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, cast, overload

import mcp
from mcp import ClientSession as MCPClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    create_model,
    model_serializer,
    model_validator,
)


def _is_public_callable(name: str, value: Any) -> bool:
    if name.startswith("_"):
        return False
    if isinstance(value, (staticmethod, classmethod)):
        return True
    return inspect.isfunction(value)


def _inline_simple_refs_in_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    For simple Pydantic v2 schemas that use `$defs` + `$ref` for enums (and
    similar), inline the referenced definition into the property schema.

    This is mainly to make OpenAI tool schemas nicer to consume:
    - turn
        "color": { "$ref": "#/$defs/Color" }
      with
        "$defs": { "Color": { "type": "string", "enum": [...] } }
      into
        "color": { "type": "string", "enum": [...] }
    """
    defs = schema.get("$defs")
    props = schema.get("properties")

    if not (isinstance(defs, dict) and isinstance(props, dict)):
        return schema

    for prop_name, prop_schema in list(props.items()):
        if not isinstance(prop_schema, dict):
            continue

        ref = cast("dict[str, Any]", prop_schema).get("$ref")
        if not (isinstance(ref, str) and ref.startswith("#/$defs/")):
            continue

        def_name = ref.split("/")[-1]
        def_schema = defs.get(def_name)
        if not isinstance(def_schema, dict):
            continue

        # Start with the definition (e.g. type + enum)
        merged: dict[str, Any] = def_schema.copy()
        # Overlay any extra keys on the property, excluding $ref
        for k, v in prop_schema.items():
            if k != "$ref":
                merged[k] = v

        props[prop_name] = merged

    return schema


def _build_args_model_from_signature(
    func: Callable[..., Any], model_name: str
) -> type[BaseModel]:
    sig = inspect.signature(func)

    type_hints: dict[str, Any] = {}
    if hasattr(func, "__annotations__"):
        try:
            type_hints: dict[str, Any] = typing.get_type_hints(func)
        except Exception:
            type_hints: dict[str, Any] = {}

    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        ann = type_hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default

        fields[name] = (ann, Field(default=default))

    ArgsModel = create_model(model_name, **fields)  # type: ignore

    orig_model_json_schema = ArgsModel.model_json_schema.__func__

    @classmethod
    def _inlining_model_json_schema(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Call the original implementation first
        schema = orig_model_json_schema(cls, *args, **kwargs)
        # Then post-process to inline simple $ref definitions
        return _inline_simple_refs_in_schema(schema)

    # Replace the classmethod on the dynamic model
    ArgsModel.model_json_schema = _inlining_model_json_schema

    return ArgsModel


class Tool(BaseModel):
    """
    Single public abstraction for tools.

    Usage patterns:

    1) Subclass for *namespaces* (methods → tools "ClassName.method"):

    ```python
    class Math(Tool):
        def add(self, x: int, y: int) -> int: ...
        def mul(self, x: int, y: int) -> int: ...
    ```

    2) Decorate functions with @tool:

    ```python
    @tool
    def ping(host: str) -> str: ...
    ```

    3) MCP tools loaded via `mcp_tools_from_url(...)`:

    ```python
    tools = await mcp_tools_from_url("http://localhost:8000/mcp")
    # returns list[Tool] that proxy to MCP tools
    ```

    Internally, a Tool is either:

    - a *namespace* (class-based, many methods)
    - a *single-callable* Python tool (function-based)
    - or a proxy for an MCP tool (created via `mcp_tools_from_url`)

    Notes:
        When defining a namespace tool, so as a subclass of Tool, the documentation of the class itself is not used by the agent. Instead, it's the documentation of each method that is used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = Field(default=None, description="Tool name.")
    description: str | None = Field(
        default=None,
        description="Tool description. Is used by the LLM to decide whether to call or not the specific tool.",
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="The parameters needed by the tool. This is a self-constructed field.",
    )

    _func: Callable[..., Any] | None = PrivateAttr(default=None)
    __namespace_methods__: list[str] = []
    _func_signature: inspect.Signature | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _set_defaults(self) -> Tool:
        if self.name is None:
            self.name: str = self.__class__.__name__

        if self.description is None:
            doc: str = inspect.getdoc(self.__class__) or ""
            self.description = doc.strip() or ""

        if self.parameters is None:
            self.parameters: dict[str, Any] = {"type": "object", "properties": {}}

        return self

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        When you subclass Tool, this inspects the class body and collects
        all public callables as "namespace methods".
        """
        super().__init_subclass__(**kwargs)
        inherited_names: set[str] = {
            name for klass in cls.__mro__[1:] for name in klass.__dict__
        }
        cls.__namespace_methods__: list[str] = [
            name
            for name, value in cls.__dict__.items()
            if _is_public_callable(name, value) and name not in inherited_names
        ]

    def _call_func(self, *args: Any, **kwargs: Any) -> Any:
        """
        Internal: call self._func with positional + keyword args,
        normalized via the underlying function's signature.
        """
        if self._func is None:
            raise RuntimeError("Cannot invoke a namespace Tool directly.")

        # Lazily cache the signature
        if self._func_signature is None:
            self._func_signature = inspect.signature(self._func)

        bound = self._func_signature.bind_partial(*args, **kwargs)
        return self._func(*bound.args, **bound.kwargs)

    @property
    def is_namespace(self) -> bool:
        """
        True if this Tool is a namespace (class with methods),
        False if it's a single-callable tool created by @tool or MCP.
        """
        return (
            bool(getattr(self.__class__, "__namespace_methods__", []))
            and self._func is None
        )

    @classmethod
    def _from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
    ) -> Tool:
        tool_name: str = name or getattr(func, "__name__", None) or ""

        if args_model is None:
            args_model = _build_args_model_from_signature(func, f"{tool_name}_Args")

        schema = args_model.model_json_schema()
        schema.setdefault("type", "object")

        inst = cls(
            name=tool_name,
            description=(description or inspect.getdoc(func) or "").strip(),
            parameters=schema,
        )
        inst._func = func
        return inst

    @classmethod
    def _from_mcp_descriptor(
        cls,
        *,
        url: str,
        mcp_tool: Any,
        namespace: str | None = None,
        unwrap_result: bool = True,
    ) -> Tool:
        full_name: str = f"{namespace}.{mcp_tool.name}" if namespace else mcp_tool.name
        params: dict = dict(mcp_tool.inputSchema or {})
        params.setdefault("type", "object")

        async def _call_mcp(**kwargs: Any) -> Any:
            # Open a short-lived MCP HTTP connection per call.
            # This keeps the Tool self-contained and agent-friendly.
            async with streamable_http_client(url) as client_context:  # pyright: ignore[reportOptionalCall]
                read, write, _aclose = client_context
                async with MCPClientSession(read, write) as session:  # pyright: ignore[reportOptionalCall]
                    await session.initialize()
                    result: mcp.types.CallToolResult = await session.call_tool(
                        mcp_tool.name, arguments=kwargs
                    )

            if not unwrap_result:
                return result

            structured = getattr(result, "structuredContent", None)
            if structured not in (None, {}):
                return structured

            content = getattr(result, "content", None)
            if content and hasattr(content[0], "text"):
                return content[0].text

            return result

        inst = cls(
            name=full_name, description=mcp_tool.description or "", parameters=params
        )
        inst._func = _call_mcp
        return inst

    def _expand_namespace(self) -> list[Tool]:
        """
        For a namespace Tool (class-based), build a list of
        single-callable Tools named "ClassName.method".
        """
        tools: list[Tool] = []
        ns: str = self.name or self.__class__.__name__

        for method_name in self.__class__.__namespace_methods__:
            bound = getattr(self, method_name)
            doc = (
                inspect.getdoc(bound)
                or inspect.getdoc(getattr(self.__class__, method_name))
                or ""
            )

            method_tool: Tool = Tool._from_function(
                func=bound,
                name=f"{ns}_{method_name}",
                description=doc.strip() or f"Method {method_name} on {ns}",
            )
            tools.append(method_tool)

        return tools

    def invoke(self, *args, **kwargs: Any) -> Any:
        """
        Synchronous invocation.

        For plain Python tools, this returns the result directly.
        For MCP-backed tools, this returns a coroutine; you should use
        `await tool.ainvoke(...)` in async code.
        """
        if self._func is None:
            raise RuntimeError("Cannot invoke a namespace Tool directly.")
        return self._call_func(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs: Any) -> Any:
        """
        Asynchronous invocation.

        - If the underlying tool is async → await it directly.
        - If it's sync → run it in a thread executor so we don't block the loop.
        """
        if self._func is None:
            raise RuntimeError("Cannot invoke a namespace Tool directly.")

        if inspect.iscoroutinefunction(self._func):
            # Call it directly; don't go through invoke() to avoid double checks
            result: Any = self._call_func(*args, **kwargs)
            # result should already be a coroutine
            return await cast(Awaitable[Any], result)

        loop: AbstractEventLoop = asyncio.get_running_loop()
        bound: partial[Any] = functools.partial(self._call_func, *args, **kwargs)
        return await loop.run_in_executor(None, bound)

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._func is None:
            raise RuntimeError("Cannot call a namespace Tool directly.")
        return self._call_func(*args, **kwargs)


@overload
def tool(_func: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    _func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_model: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_model: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Tool] | Tool:
    """
    Decorator for *functions* that turns them into Tool instances.

    From the user's POV:

        @tool
        def ping(host: str) -> str: ...
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool._from_function(
            func,
            name=name,
            description=description,
            args_model=args_model,
        )

    if _func is not None:
        return decorator(_func)

    return decorator


async def mcp_tools_from_url(
    url: str,
    *,
    namespace: str | None = None,
    unwrap_result: bool = True,
) -> list[Tool]:
    """
    Connect to an MCP server over HTTP(S) and adapt its tools into `Tool` objects.

    This is the *only* MCP-specific entry point you need from the outside.

    - `url`: MCP HTTP endpoint (streamable), e.g. "http://localhost:8000/mcp"
    - `namespace`: optional prefix for tool names, e.g. "weather.get_forecast"
    - `unwrap_result`:
        * if True: return structuredContent or first text fragment
        * if False: return the full `CallToolResult`

    Example:

        tools = await mcp_tools_from_url("http://localhost:8000/mcp", namespace="remote")
        agent = Agent(tools=tools, ...)  # your agent just sees Tool objects
    """

    async with streamable_http_client(url) as client_context:  # pyright: ignore[reportOptionalCall]
        read, write, _aclose = client_context
        async with MCPClientSession(read, write) as session:  # pyright: ignore[reportOptionalCall]
            await session.initialize()
            tools_resp: mcp.types.ListToolsResult = await session.list_tools()
            mcp_tools: list[mcp.types.Tool] = tools_resp.tools

    tools: list[Tool] = [
        Tool._from_mcp_descriptor(
            url=url,
            mcp_tool=mcp_tool,
            namespace=namespace,
            unwrap_result=unwrap_result,
        )
        for mcp_tool in mcp_tools
    ]

    return tools
