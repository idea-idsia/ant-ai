from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from starlette.applications import Starlette


def run_uvicorn(app: Starlette, host: str, port: int) -> None:
    """Serve `app` with uvicorn until interrupted."""
    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    except ImportError as e:
        raise ImportError(
            "Uvicorn is not installed. Please install it with 'uv add uvicorn'."
        ) from e
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise RuntimeError(f"Failed to start the server: {e}") from e
