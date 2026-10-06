"""Correlation-id plumbing for structured logging.

Every cycle, monitor exit, and HTTP request gets an id that flows through
ContextVars and into JSON log records via `ObservabilityFilter`. Operators
can grep `cycle_id=cycle-...` to follow a single iteration end-to-end.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

cycle_id_var: ContextVar[str] = ContextVar("cycle_id", default="")
monitor_id_var: ContextVar[str] = ContextVar("monitor_id", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Which process owns these log records. Set once at bot startup
# (set_service) so records can be filtered by service wherever logs from
# several processes end up side by side.
# Deliberately a plain module global, NOT a ContextVar: it's process-wide
# (one bot per process) and must be visible from APScheduler job contexts
# and worker threads, which a ContextVar set in run() does not propagate to.
_service_name: str = ""


def set_service(name: str) -> None:
    """Tag every subsequent log record from this process with the owning bot."""
    global _service_name
    _service_name = name


def new_id(prefix: str) -> str:
    """Return ``prefix-XXXXXXXX`` with 4 hex bytes of randomness."""
    return f"{prefix}-{secrets.token_hex(4)}"


@contextmanager
def cycle_context(cycle_id: str | None = None) -> Iterator[str]:
    """Set ``cycle_id_var`` for the duration of the block."""
    cid = cycle_id or new_id("cycle")
    token = cycle_id_var.set(cid)
    try:
        yield cid
    finally:
        cycle_id_var.reset(token)


class ObservabilityFilter(logging.Filter):
    """Attach the active correlation ids to every LogRecord.

    The JSON formatter only emits a key when its value is non-empty, so
    records outside any cycle/monitor/request scope simply omit the field.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        cid = cycle_id_var.get()
        mid = monitor_id_var.get()
        rid = request_id_var.get()
        if cid:
            record.cycle_id = cid
        if mid:
            record.monitor_id = mid
        if rid:
            record.request_id = rid
        if _service_name:
            record.service = _service_name
        return True
