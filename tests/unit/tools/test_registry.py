from __future__ import annotations

import pytest

from ant_ai.tools.registry import ToolRegistry
from ant_ai.tools.tool import Tool, tool as tool_decorator


@tool_decorator
def _echo(msg: str) -> str:
    """Echo the message."""
    return msg


@tool_decorator
def _ping(host: str) -> str:
    """Ping a host."""
    return f"pong:{host}"


@pytest.mark.unit
def test_register_and_contains():
    registry = ToolRegistry()
    registry.register(_echo)
    assert "_echo" in registry
    assert "nonexistent" not in registry


@pytest.mark.unit
def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(_echo)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_echo)


@pytest.mark.unit
def test_getitem_returns_tool():
    registry = ToolRegistry()
    registry.register(_echo)
    tool = registry["_echo"]
    assert tool.name == "_echo"


@pytest.mark.unit
def test_getitem_missing_raises_key_error():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="not found in registry"):
        _ = registry["nonexistent"]


@pytest.mark.unit
def test_remove_removes_tool():
    registry = ToolRegistry()
    registry.register(_echo)
    registry.remove("_echo")
    assert "_echo" not in registry


@pytest.mark.unit
def test_remove_nonexistent_raises_key_error():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="not registered"):
        registry.remove("nonexistent")


@pytest.mark.unit
def test_tools_property_lists_all_registered_tools():
    registry = ToolRegistry()
    registry.register(_echo)
    registry.register(_ping)
    tools = registry.tools
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"_echo", "_ping"}


@pytest.mark.unit
def test_to_serialized_returns_function_dicts():
    registry = ToolRegistry()
    registry.register(_echo)
    serialized = registry.to_serialized()
    assert len(serialized) == 1
    assert serialized[0]["type"] == "function"
    assert "function" in serialized[0]
    assert serialized[0]["function"]["name"] == "_echo"


@pytest.mark.unit
def test_init_with_tools_list_auto_registers():
    registry = ToolRegistry(tools=[_echo, _ping])
    assert "_echo" in registry
    assert "_ping" in registry
    assert len(registry.tools) == 2


# ===========================================================================
# Namespace tool expansion — guards against "Cannot invoke a namespace Tool
# directly" errors that occur when class-based tools are stored as-is.
# ===========================================================================


class _Calc(Tool):
    """Simple calculator namespace for testing."""

    def add(self, x: int, y: int) -> int:
        """Add x and y."""
        return x + y

    def mul(self, x: int, y: int) -> int:
        """Multiply x by y."""
        return x * y


@pytest.mark.unit
def test_namespace_tool_is_expanded_on_register():
    """Registering a namespace tool must produce individual sub-tools, not the namespace itself."""
    registry = ToolRegistry()
    registry.register(_Calc())

    assert "_Calc" not in registry, "The namespace itself must NOT be registered"
    assert "_Calc_add" in registry
    assert "_Calc_mul" in registry


@pytest.mark.unit
def test_namespace_tool_expansion_count():
    """Each public method becomes exactly one registered tool."""
    registry = ToolRegistry()
    registry.register(_Calc())

    assert len(registry.tools) == len(_Calc.__namespace_methods__)


@pytest.mark.unit
def test_namespace_sub_tools_are_invocable():
    """Expanded sub-tools must be callable without raising RuntimeError.

    This directly prevents the regression: 'Cannot invoke a namespace Tool directly'.
    """
    registry = ToolRegistry()
    registry.register(_Calc())

    assert registry["_Calc_add"].invoke(x=2, y=3) == 5
    assert registry["_Calc_mul"].invoke(x=3, y=4) == 12


@pytest.mark.unit
async def test_namespace_sub_tools_are_async_invocable():
    """Expanded sub-tools must also work via ainvoke (the path used by the agent loop)."""
    registry = ToolRegistry()
    registry.register(_Calc())

    assert await registry["_Calc_add"].ainvoke(x=10, y=5) == 15
