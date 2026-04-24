from pathlib import Path

import pytest

from ant_ai.tools.builtins.filesystem_tool import FilesystemTool


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def fs_tool(tmp_path: Path) -> FilesystemTool:
    return FilesystemTool(workspace_root=str(tmp_path))
