"""Tests for the idle-cash treasury log in :class:`RecordCycleAnalyticsStage`.

The stage logs an idle-cash treasury plan when the bot is fully flat.
It has clear branching (``in_order`` short-circuit, low-cash
short-circuit, happy path with planning + log, exception swallow,
no-op plan no-log, de-dup of identical plans across cycles).

The stage exposes ``_log_treasury_plan(account)`` as a private helper
the test suite calls directly — no full ``run_cycle`` integration
needed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from halal_trader.core.cycle_stages import RecordCycleAnalyticsStage
from halal_trader.domain.models import CryptoAccount


def _stage() -> RecordCycleAnalyticsStage:
    """Construct an analytics stage with stubbed deps — the treasury
    helper only reads ``self._last_treasury_key`` (initialised to
    ``None``) and the passed-in account."""
    return RecordCycleAnalyticsStage(
        hub=MagicMock(),
        shadow_runner=None,
        replay_store=None,
    )


def _account(*, total: float, available: float, in_order: float = 0.0) -> CryptoAccount:
    return CryptoAccount(
        total_balance_usdt=total,
        available_balance_usdt=available,
        in_order_usdt=in_order,
        usdt_free=available,
    )


# ── Short-circuit branches ─────────────────────────────────


def test_log_treasury_plan_skips_when_funds_in_order(caplog):
    """``in_order_usdt > 0`` means the bot has open orders — not flat,
    don't bother computing a treasury plan."""
    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=1000, available=900, in_order=100))
    assert not any("treasury" in r.message for r in caplog.records)


def test_log_treasury_plan_skips_when_low_available_balance(caplog):
    """Below $50 available → don't deploy (would be sub-floor anyway).
    Saves a ``plan_idle_cash`` call on the dust-only path."""
    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=49, available=49))
    assert not any("treasury" in r.message for r in caplog.records)


def test_log_treasury_plan_just_at_threshold_of_50_runs(caplog):
    """Boundary: exactly $50 → enters the plan branch (the check is
    ``< 50``, not ``<= 50``)."""
    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=50, available=50))
    # The lack of an exception is the load-bearing assertion.


# ── Happy path ─────────────────────────────────────────────


def test_log_treasury_plan_logs_deploy_action_when_cash_far_above_floor(caplog):
    """A large available balance should produce a non-noop deploy plan
    and emit the ``treasury: ...`` log line."""
    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=100_000, available=100_000))
    treasury_lines = [r.message for r in caplog.records if "treasury" in r.message]
    assert len(treasury_lines) >= 1
    msg = treasury_lines[0]
    assert "$" in msg
    assert "into" in msg


# ── Exception swallow ─────────────────────────────────────


def test_log_treasury_plan_swallows_exception(caplog, monkeypatch):
    """If ``plan_idle_cash`` raises (e.g. config bug), the cycle must
    keep running — the stage logs at debug and returns silently."""
    import halal_trader.core.treasury as treasury_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated treasury failure")

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", boom)

    s = _stage()
    with caplog.at_level(logging.DEBUG):
        s._log_treasury_plan(_account(total=10_000, available=10_000))
    info_lines = [
        r for r in caplog.records if r.levelno == logging.INFO and "treasury" in r.message
    ]
    assert info_lines == []
    debug_lines = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "treasury plan computation failed" in r.message
    ]
    assert len(debug_lines) == 1


# ── No-op plan ─────────────────────────────────────────────


def test_log_treasury_plan_skips_log_when_plan_is_noop(caplog, monkeypatch):
    """When ``plan_idle_cash`` returns a no-op plan, no log line fires."""
    import halal_trader.core.treasury as treasury_mod

    class _Noop:
        is_noop = True
        action = "hold"
        amount_usd = 0.0
        instrument = ""
        reason = "balanced"

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", lambda **kw: _Noop())

    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=10_000, available=10_000))
    info_treasury = [
        r for r in caplog.records if r.levelno == logging.INFO and "treasury" in r.message
    ]
    assert info_treasury == []


def test_log_treasury_plan_emits_log_when_plan_is_not_noop(caplog, monkeypatch):
    """A non-noop plan emits the formatted line."""
    import halal_trader.core.treasury as treasury_mod

    class _Plan:
        is_noop = False
        action = "deploy"
        amount_usd = 5000.0
        instrument = "TBIL"
        reason = "cash above floor"

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", lambda **kw: _Plan())

    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=10_000, available=10_000))
    treasury_lines = [r for r in caplog.records if "treasury" in r.message]
    assert len(treasury_lines) == 1
    msg = treasury_lines[0].message
    assert "deploy" in msg
    assert "$5000.00" in msg
    assert "TBIL" in msg
    assert "cash above floor" in msg


def test_log_treasury_plan_passes_correct_args_to_planner(monkeypatch):
    """The helper splits ``total - available`` as ``positions_value`` and
    threads ``available`` as ``cash_balance``. Pin so a future refactor
    doesn't accidentally swap the two."""
    import halal_trader.core.treasury as treasury_mod

    captured: dict = {}

    class _Plan:
        is_noop = True
        action = "hold"
        amount_usd = 0.0
        instrument = ""
        reason = ""

    def fake_plan(**kw):
        captured.update(kw)
        return _Plan()

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", fake_plan)

    s = _stage()
    s._log_treasury_plan(_account(total=10_000, available=7_000))
    assert captured["cash_balance"] == 7_000
    assert captured["positions_value"] == 3_000
    assert captured["current_treasury_value"] == 0.0


def test_log_treasury_plan_zero_available_skips_via_low_cash_branch(caplog):
    """Edge: $0 available → < 50, short-circuit before planning."""
    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=0, available=0))
    assert not any("treasury" in r.message for r in caplog.records)


def test_log_treasury_plan_dedupes_identical_plans_across_calls(caplog, monkeypatch):
    """Same action+instrument+rounded-amount on consecutive calls →
    only the first call emits a log line (avoids the same idle-cash
    suggestion firing every cycle while flat)."""
    import halal_trader.core.treasury as treasury_mod

    class _Plan:
        is_noop = False
        action = "deploy"
        amount_usd = 5000.0
        instrument = "TBIL"
        reason = "cash above floor"

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", lambda **kw: _Plan())

    s = _stage()
    with caplog.at_level(logging.INFO):
        s._log_treasury_plan(_account(total=10_000, available=10_000))
        s._log_treasury_plan(_account(total=10_000, available=10_000))
    treasury_lines = [r for r in caplog.records if "treasury" in r.message]
    assert len(treasury_lines) == 1
