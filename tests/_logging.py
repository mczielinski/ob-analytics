"""Shared helper for tests that check what ob-analytics logs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def warnings_logged() -> Iterator[list[str]]:
    """Collect the messages logged at WARNING or above inside the block."""
    messages: list[str] = []
    # The package disables its own logger on import, as a library should.
    logger.enable("ob_analytics")
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)
        logger.disable("ob_analytics")
