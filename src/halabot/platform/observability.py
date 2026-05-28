"""Logging + causal-chain context (INV-4, INV-5).

Binds the active event's ``correlation_id`` onto every log record emitted while
handling it, so a decision's whole causal chain is greppable in the logs (and
the dashboard's decision stream has a key to query). ``setup_logging`` installs
a formatter that surfaces it. INV-4 (log the exception *type*, never a bare
empty ``str(e)``) is a call-site convention the codebase already follows; this
module makes the correlation context automatic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationFilter(logging.Filter):
    """Attaches ``correlation_id`` to every record (``-`` when unset)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get() or "-"
        return True


@contextmanager
def correlation_scope(correlation_id: str | None) -> Iterator[None]:
    """Bind ``correlation_id`` for the duration of the block (ContextVar, so it
    flows across ``await`` within the same task)."""
    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


def current_correlation_id() -> str | None:
    return _correlation_id.get()


def setup_logging(level: int | str = "INFO") -> None:
    """Install a root handler whose format carries the correlation id."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s")
    )
    handler.addFilter(CorrelationFilter())
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)
