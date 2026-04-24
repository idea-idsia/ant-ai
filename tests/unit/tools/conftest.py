import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).parent


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mcp_server():
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    env = {**os.environ, "MCP_TEST_PORT": str(port)}

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(_TEST_DIR)!r});"
            "from test_tool_mcp import mcp; mcp.run(transport='streamable-http')",
        ],
        env=env,
    )

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("MCP server did not start in time")

    yield url

    proc.kill()
    proc.wait(timeout=5)
