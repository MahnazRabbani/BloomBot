"""Structured JSON logging for BloomBot.

Configures Python's logging module with a JSON formatter so every log record is
emitted as a single machine-parsable line. This makes per-request observability
data (timings, token counts, retrieved ids) easy to ship to and query in a log
aggregator.

Call :func:`configure_logging` once at startup, then obtain the shared logger
with :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger.json import JsonFormatter

LOGGER_NAME = "bloombot"

# Standard LogRecord attributes are renamed/promoted into the JSON output so the
# emitted object reads naturally (``timestamp`` and ``level`` rather than the
# raw ``asctime``/``levelname``).
_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the shared ``bloombot`` logger.

    Idempotent: repeated calls do not stack duplicate handlers, so importing or
    re-initializing the app never causes doubled log lines.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    # Records are handled here; don't also bubble up to the root logger's
    # (typically plain-text) handlers, which would double-log every line.
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            JsonFormatter(
                _FORMAT,
                rename_fields={"asctime": "timestamp", "levelname": "level"},
            )
        )
        logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the shared ``bloombot`` logger, configuring it if needed."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return configure_logging()
    return logger
