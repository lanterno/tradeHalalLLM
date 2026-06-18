"""Tests for correlation-id plumbing and structured event logging."""

from __future__ import annotations

import logging

import pytest

from halal_trader.core import events
from halal_trader.core.cycle import BaseCycleService
from halal_trader.core.observability import (
    ObservabilityFilter,
    cycle_context,
    cycle_id_var,
    monitor_context,
    monitor_id_var,
    new_id,
    request_context,
    request_id_var,
)


def test_new_id_has_prefix_and_hex_suffix():
    cid = new_id("cycle")
    assert cid.startswith("cycle-")
    suffix = cid.removeprefix("cycle-")
    assert len(suffix) == 8
    int(suffix, 16)


def test_cycle_context_sets_and_resets():
    assert cycle_id_var.get() == ""
    with cycle_context() as cid:
        assert cycle_id_var.get() == cid
        assert cid.startswith("cycle-")
    assert cycle_id_var.get() == ""


def test_cycle_context_accepts_explicit_id():
    with cycle_context("cycle-deadbeef") as cid:
        assert cid == "cycle-deadbeef"
        assert cycle_id_var.get() == "cycle-deadbeef"


def test_monitor_and_request_contexts_independent():
    with cycle_context("cycle-aaaaaaaa"):
        with monitor_context("mon-bbbbbbbb"):
            with request_context("req-cccccccc"):
                assert cycle_id_var.get() == "cycle-aaaaaaaa"
                assert monitor_id_var.get() == "mon-bbbbbbbb"
                assert request_id_var.get() == "req-cccccccc"
            assert request_id_var.get() == ""
        assert monitor_id_var.get() == ""
    assert cycle_id_var.get() == ""


def test_observability_filter_attaches_only_set_ids():
    filt = ObservabilityFilter()

    rec_outside = logging.LogRecord(
        name="halal_trader.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="hi",
        args=None,
        exc_info=None,
    )
    filt.filter(rec_outside)
    assert not hasattr(rec_outside, "cycle_id")
    assert not hasattr(rec_outside, "monitor_id")
    assert not hasattr(rec_outside, "request_id")

    with cycle_context("cycle-x"):
        rec_in = logging.LogRecord(
            name="halal_trader.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg="hi",
            args=None,
            exc_info=None,
        )
        filt.filter(rec_in)
        assert rec_in.cycle_id == "cycle-x"
        assert not hasattr(rec_in, "monitor_id")


def test_observability_filter_attaches_service_when_set():
    """set_service() tags every record so the shared stock/crypto log file
    can be filtered by which bot emitted each line."""
    from halal_trader.core.observability import service_var

    filt = ObservabilityFilter()

    def _rec():
        return logging.LogRecord(
            name="halal_trader.test", level=logging.INFO, pathname=__file__,
            lineno=0, msg="hi", args=None, exc_info=None,
        )

    rec_default = _rec()
    filt.filter(rec_default)
    assert not hasattr(rec_default, "service")  # unset → field omitted

    token = service_var.set("stock")
    try:
        rec_tagged = _rec()
        filt.filter(rec_tagged)
        assert rec_tagged.service == "stock"
    finally:
        service_var.reset(token)


class _StubCycle(BaseCycleService):
    def __init__(self) -> None:
        super().__init__()
        self.observed_cycle_ids: list[str] = []
        self._halt = False

    async def _pre_cycle_checks(self) -> bool:
        self.observed_cycle_ids.append(cycle_id_var.get())
        return True

    async def _should_halt(self) -> bool:
        return self._halt

    async def _run_cycle_impl(self) -> None:
        self.observed_cycle_ids.append(cycle_id_var.get())


@pytest.mark.asyncio
async def test_run_cycle_sets_unique_cycle_id_each_run():
    c = _StubCycle()
    await c.run_cycle()
    await c.run_cycle()
    assert len(c.observed_cycle_ids) == 4
    first_run = c.observed_cycle_ids[0]
    second_run = c.observed_cycle_ids[2]
    assert first_run.startswith("cycle-")
    assert first_run != second_run
    # The same id is seen by every step inside one run.
    assert c.observed_cycle_ids[0] == c.observed_cycle_ids[1]


@pytest.mark.asyncio
async def test_run_cycle_emits_complete_event(caplog):
    caplog.set_level(logging.INFO, logger="halal_trader.core.cycle")
    await _StubCycle().run_cycle()
    event_names = [r.__dict__.get("event") for r in caplog.records]
    assert events.CYCLE_START in event_names
    assert events.CYCLE_COMPLETE in event_names


@pytest.mark.asyncio
async def test_run_cycle_emits_halted_event(caplog):
    caplog.set_level(logging.INFO, logger="halal_trader.core.cycle")
    cycle = _StubCycle()
    cycle._halt = True
    await cycle.run_cycle()
    event_names = [r.__dict__.get("event") for r in caplog.records]
    assert events.CYCLE_HALTED in event_names
    assert events.CYCLE_COMPLETE not in event_names


class _ExplodingCycle(BaseCycleService):
    async def _pre_cycle_checks(self) -> bool:
        return True

    async def _should_halt(self) -> bool:
        return False

    async def _run_cycle_impl(self) -> None:
        raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_run_cycle_exception_triggers_alert():
    from unittest.mock import AsyncMock

    alerts = AsyncMock()
    cycle = _ExplodingCycle(alerts=alerts)
    await cycle.run_cycle()
    alerts.notify.assert_awaited_once()
    error_type, details = alerts.notify.await_args.args
    assert error_type == events.CYCLE_FAILED
    assert "RuntimeError" in details
    assert "kaboom" in details
