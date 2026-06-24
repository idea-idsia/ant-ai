"""Local stdio → WebSocket proxy for a remote ACP server.

VSCode (or any ACP client) spawns this script as a subprocess and talks to it
over stdin/stdout. The proxy forwards every JSON-RPC message over WebSocket to
a remote :class:`~ant_ai.acp.server.ACPServer` and streams responses back.

Usage
-----
Point your ACP client at this script:

.. code-block:: json

    {
        "acp.agents": {
            "ant-ai (remote)": {
                "command": "python",
                "args": [
                    "/path/to/examples/acp_proxy.py",
                    "ws://your-remote-host:9001/acp/ws"
                ]
            }
        }
    }

The default URL is ``ws://127.0.0.1:9001/acp/ws`` (useful for local testing
when you want to keep the agent process separate from the editor process).
"""

from __future__ import annotations

import asyncio
import sys

DEFAULT_URL = "ws://127.0.0.1:9001/acp/ws"


async def _run(url: str) -> None:
    try:
        import websockets
    except ImportError:
        sys.stderr.write("websockets is required: pip install websockets\n")
        sys.exit(1)

    stdin = asyncio.StreamReader()
    proto = asyncio.StreamReaderProtocol(stdin)
    await asyncio.get_event_loop().connect_read_pipe(lambda: proto, sys.stdin)

    stdout = sys.stdout.buffer

    async with websockets.connect(url) as ws:

        async def _stdin_to_ws() -> None:
            while True:
                line = await stdin.readline()
                if not line:
                    break
                await ws.send(line.decode().rstrip("\n"))

        async def _ws_to_stdout() -> None:
            async for message in ws:
                stdout.write((message + "\n").encode())
                stdout.flush()

        await asyncio.gather(
            _stdin_to_ws(),
            _ws_to_stdout(),
            return_exceptions=True,
        )


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    asyncio.run(_run(url))


if __name__ == "__main__":
    main()
