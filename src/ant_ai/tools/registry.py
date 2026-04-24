from __future__ import annotations

from pydantic import BaseModel, PrivateAttr

from ant_ai.tools.tool import Tool


class ToolRegistry(BaseModel):
    """
    Maintains a name-keyed mapping of Tool instances.
    """

    _tools: dict[str, Tool] = PrivateAttr(default_factory=dict)

    def __init__(self, tools: list[Tool] | None = None, **data):
        super().__init__(**data)
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add tool to the registry.

        For namespace tools (class-based with multiple methods), each method is
        registered as an individual callable tool named "ClassName_method".

        Args:
            tool: Tool to add to the registry.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.is_namespace:
            for sub_tool in tool._expand_namespace():
                if not sub_tool.name:
                    raise ValueError("Tool must have a name before registration.")
                if sub_tool.name in self._tools:
                    raise ValueError(
                        f"Tool '{sub_tool.name}' is already registered. "
                        "Use a unique name or remove the existing registration first."
                    )
                self._tools[sub_tool.name] = sub_tool
        else:
            if not tool.name:
                raise ValueError("Tool must have a name before registration.")
            if tool.name in self._tools:
                raise ValueError(
                    f"Tool '{tool.name}' is already registered. "
                    "Use a unique name or remove the existing registration first."
                )
            self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        """Remove the tool named name.

        Args:
            name: Name of the tool to remove from the registry.

        Raises:
            KeyError: If no such tool is registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        del self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool '{name}' not found in registry.") from None

    @property
    def tools(self) -> list[Tool]:
        """All registered tools as an ordered list."""
        return list(self._tools.values())

    def to_serialized(self) -> list[dict]:
        """Serialize all tools for inclusion in an LLM API request."""
        return [t.model_dump() for t in self._tools.values()]
