"""LLM daily spend cap tests — accumulation, day rollover, and halt trip."""

from decimal import Decimal

import pytest

from halal_trader.core.halt import clear_halt, get_status, set_halt
from halal_trader.core.llm import budget as budget_mod
from halal_trader.core.llm.budget import LLMBudget


async def test_records_spend_without_tripping_when_under_cap(engine):
    b = LLMBudget(engine, cap_usd=10.0)
    await b.record(Decimal("3.50"))
    await b.record(Decimal("2.00"))
    assert b.spent_today_usd == Decimal("5.50")
    assert (await get_status(engine)).enabled is False


async def test_zero_cap_disables_enforcement(engine):
    b = LLMBudget(engine, cap_usd=0)
    await b.record(Decimal("999.00"))
    assert (await get_status(engine)).enabled is False


async def test_trip_engages_kill_switch_and_logs_reason(engine):
    b = LLMBudget(engine, cap_usd=5.0)
    await b.record(Decimal("3.00"))
    await b.record(Decimal("3.00"))  # cumulative 6.00 > 5.00
    status = await get_status(engine)
    assert status.enabled is True
    assert status.set_by == "llm-budget"
    assert "5.00" in (status.reason or "")
    assert "6.00" in (status.reason or "")


async def test_trip_does_not_clobber_pre_existing_halt(engine):
    await set_halt(engine, reason="operator stop", set_by="ahmed")
    b = LLMBudget(engine, cap_usd=1.0)
    await b.record(Decimal("2.00"))  # would have tripped
    status = await get_status(engine)
    assert status.enabled is True
    # The original halt reason wins — we never overwrite an existing halt.
    assert status.reason == "operator stop"
    assert status.set_by == "ahmed"


async def test_day_rollover_resets_spent(engine, monkeypatch):
    b = LLMBudget(engine, cap_usd=100.0)
    await b.record(Decimal("50.00"))
    assert b.spent_today_usd == Decimal("50.00")

    monkeypatch.setattr(budget_mod, "_today", lambda: "2099-01-01")
    await b.record(Decimal("1.00"))
    assert b.spent_today_usd == Decimal("1.00")
    await clear_halt(engine)


async def test_record_only_trips_once(engine):
    b = LLMBudget(engine, cap_usd=1.0)
    await b.record(Decimal("2.00"))  # trips
    # Subsequent recordings should not re-engage halt or duplicate logs.
    await b.record(Decimal("1.00"))
    await b.record(Decimal("1.00"))
    status = await get_status(engine)
    assert status.enabled is True


async def test_negative_or_zero_cost_is_noop(engine):
    b = LLMBudget(engine, cap_usd=1.0)
    await b.record(Decimal("0"))
    await b.record(Decimal("-5"))
    assert b.spent_today_usd == Decimal("0")


@pytest.fixture(autouse=True)
def _isolate_halt_state():
    yield  # individual tests dispose their engine, so state is per-DB
