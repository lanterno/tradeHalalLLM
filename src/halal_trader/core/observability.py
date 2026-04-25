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


@contextmanager
def monitor_context(monitor_id: str | None = None) -> Iterator[str]:
    """Set ``monitor_id_var`` for the duration of the block (per-trade exits)."""
    mid = monitor_id or new_id("mon")
    token = monitor_id_var.set(mid)
    try:
        yield mid
    finally:
        monitor_id_var.reset(token)


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Set ``request_id_var`` for the duration of an HTTP request."""
    rid = request_id or new_id("req")
    token = request_id_var.set(rid)
    try:
        yield rid
    finally:
        request_id_var.reset(token)


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
        return True
