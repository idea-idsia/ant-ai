from __future__ import annotations

import os
import sys
from typing import Any

from loguru import logger


def configure_logging(*, level: str | None = None, json: bool | None = None) -> None:
    """
    Configure loguru sinks once at process startup.

    Uses the following environment variables:

    - LOG_LEVEL: Log level (default: INFO)
    - LOG_JSON: Set to "1" for JSON output

    Args:
        level: _description_. Defaults to None.
        json: _description_. Defaults to None.
    """
    _level: str = level or os.getenv("LOG_LEVEL", "INFO")
    _json: bool = json if json is not None else (os.getenv("LOG_JSON", "0") == "1")

    logger.remove()

    if _json:
        logger.add(
            sys.stdout,
            level=_level,
            serialize=True,  # emits {"text": ..., "record": {...}} per line
            backtrace=False,  # keep traces out of prod JSON logs
            diagnose=False,
            enqueue=True,
        )
        return

    # Human-readable format for local development.
    logger.add(
        sys.stdout,
        level=_level,
        backtrace=True,
        diagnose=False,
        enqueue=True,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "session={extra[session_id]} agent={extra[agent]} "
            "step={extra[step]} event={extra[event]} | "
            "<level>{message}</level>"
        ),
    )


def bind_logger(
    *,
    session_id: str = "-",
    task_id: str = "-",
    context_id: str = "-",
    agent: str = "-",
    node: str = "-",
    step: int | str = "-",
    event: str = "-",
    **extra: Any,
):
    """
    Return a loguru logger bound with all standard correlation fields.

    All fields default to "-" so the format string never raises KeyError.

    Args:
        session_id: Request/session identifier (primary Loki label). Defaults to "-".
        task_id: Workflow task identifier. Defaults to "-".
        context_id: Sub-task or thread context. Defaults to "-".
        agent: Agent name. Defaults to "-".
        node: Workflow node name. Defaults to "-".
        step: Step name (e.g. "llm", "tool"). Defaults to "-".
        event: Dot-namespaced action (e.g. "step.start", "step.error"). Defaults to "-".

    Returns:
        The loguru logger instance.
    """
    return logger.bind(
        session_id=session_id,
        task_id=task_id,
        context_id=context_id,
        agent=agent,
        node=node,
        step=step,
        event=event,
        **extra,
    )
