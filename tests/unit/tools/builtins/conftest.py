from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import Field, PrivateAttr

from ant_ai.core.message import Message
from ant_ai.tools.builtins.filesystem_tool import FilesystemTool
from ant_ai.tools.builtins.memory_tool import MemoryTool


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def fs_tool(tmp_path: Path) -> FilesystemTool:
    return FilesystemTool(workspace_root=str(tmp_path))


class StubMemory(MemoryTool):
    """`MemoryTool` backend stand-in that records calls for assertions."""

    retrieve_calls: list[dict] = Field(default_factory=list)
    update_calls: list[dict] = Field(default_factory=list)
    _fixed_entries: list[Message] = PrivateAttr(default_factory=list)

    def set_entries(self, entries: list[Message]) -> None:
        self._fixed_entries = entries

    async def retrieve(
        self, query: str, *, top_k: int = 5, **kwargs: Any
    ) -> list[Message]:
        self.retrieve_calls.append({"query": query, "top_k": top_k, **kwargs})
        return list(self._fixed_entries)

    async def update(self, messages: list[Message], **kwargs: Any) -> None:
        self.update_calls.append({"messages": list(messages), **kwargs})


@pytest.fixture
def stub_memory() -> StubMemory:
    return StubMemory()
