from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def ant_home(tmp_path_factory, monkeypatch):
    """Keep saved conversations out of the real home and out of the workspace."""
    home = tmp_path_factory.mktemp("ant-home")
    monkeypatch.setenv("ANT_HOME", str(home))
    return home
