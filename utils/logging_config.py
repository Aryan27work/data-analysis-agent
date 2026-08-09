"""Structured logging setup shared across the application.

Deliberately does NOT log full dataframes, API keys, or raw uploaded content.
Only stage names, timings, warnings, and error messages are logged.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from utils.config import LOG_LEVEL


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""

    global _CONFIGURED

    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(
                logging,
                LOG_LEVEL.upper(),
                logging.INFO,
            ),
            format=(
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(name)s | "
                "%(message)s"
            ),
        )

        _CONFIGURED = True

    return logging.getLogger(name)


@contextmanager
def log_stage(
    logger: logging.Logger,
    stage_name: str,
):
    """Log stage start, completion time, and failures.

    Exceptions are never swallowed.
    """

    start = time.perf_counter()

    logger.info(
        "stage_start stage=%s",
        stage_name,
    )

    try:
        yield

    except Exception as exc:
        elapsed = time.perf_counter() - start

        logger.error(
            "stage_failed stage=%s elapsed_s=%.2f error=%s",
            stage_name,
            elapsed,
            exc.__class__.__name__,
        )

        raise

    else:
        elapsed = time.perf_counter() - start

        logger.info(
            "stage_complete stage=%s elapsed_s=%.2f",
            stage_name,
            elapsed,
        )


def timed_node(
    node_fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Decorator for LangGraph node functions.

    Logs entry, completion time, and newly-added errors.
    """

    logger = get_logger(
        f"agent.nodes.{node_fn.__name__}"
    )

    @wraps(node_fn)
    def wrapper(
        state: dict[str, Any],
    ) -> dict[str, Any]:

        start = time.perf_counter()

        logger.info("node_start")

        try:
            result = node_fn(state)

        except Exception:
            logger.exception("node_failed")
            raise

        elapsed = time.perf_counter() - start

        previous_errors = state.get(
            "errors",
            [],
        )

        result_errors = result.get(
            "errors",
            [],
        )

        try:
            n_new_errors = max(
                len(result_errors) - len(previous_errors),
                0,
            )
        except TypeError:
            n_new_errors = 0

        logger.info(
            "node_complete elapsed_s=%.2f new_warnings=%d",
            elapsed,
            n_new_errors,
        )

        return result

    return wrapper