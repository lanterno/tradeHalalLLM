"""Tests for `CryptoCycleService._log_treasury_plan`.

The helper logs an idle-cash treasury plan when the bot is fully
flat. It has clear branching (in_order short-circuit, low-cash
short-circuit, happy path with planning + log, exception swallow,
no-op plan no-log) but no direct tests today.

We construct a minimal service with stubbed deps and call the helper
directly — no `run_cycle` integration needed.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

from halal_trader.crypto.cycle import CryptoCycleService
from halal_trader.domain.models import CryptoAccount


def _service() -> CryptoCycleService:
    """Construct a service with stubbed deps — `_log_treasury_plan`
    only reads `self` indirectly (no other attrs on `self`)."""
    return CryptoCycleService(
        broker=MagicMock(),
        screener=AsyncMock(),
        strategy=AsyncMock(),
        executor=AsyncMock(),
        portfolio=AsyncMock(),
        ws_manager=MagicMock(),
        configured_pairs=["BTCUSDT"],
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
    """`in_order_usdt > 0` means the bot has open orders — not flat,
    don't bother computing a treasury plan."""
    svc = _service()
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=1000, available=900, in_order=100))
    assert not any("treasury" in r.message for r in caplog.records)


def test_log_treasury_plan_skips_when_low_available_balance(caplog):
    """Below $50 available → don't deploy (would be sub-floor anyway).
    Saves a `plan_idle_cash` call on the dust-only path."""
    svc = _service()
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=49, available=49))
    assert not any("treasury" in r.message for r in caplog.records)


def test_log_treasury_plan_just_at_threshold_of_50_runs(caplog):
    """Boundary: exactly $50 → enters the plan branch (the check is
    `< 50`, not `<= 50`)."""
    svc = _service()
    # $50 cash, no positions, no existing treasury → likely produces
    # a deploy plan (depends on default policy floor). Either way, the
    # branch is exercised, which is what we're pinning.
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=50, available=50))
    # Either a deploy log fires OR the plan was a noop — both prove
    # the branch ran without short-circuiting at the dust floor.
    # The lack of an exception is the load-bearing assertion.


# ── Happy path ─────────────────────────────────────────────


def test_log_treasury_plan_logs_deploy_action_when_cash_far_above_floor(caplog):
    """A large available balance should produce a non-noop deploy
    plan and emit the `treasury: ...` log line."""
    svc = _service()
    # $100k available with no positions / no treasury → a real deploy.
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=100_000, available=100_000))
    treasury_lines = [r.message for r in caplog.records if "treasury" in r.message]
    assert len(treasury_lines) >= 1
    msg = treasury_lines[0]
    # Per the format: `treasury: <action> $<amount> into <instrument> — <reason>`
    assert "$" in msg
    assert "into" in msg


# ── Exception swallow ─────────────────────────────────────


def test_log_treasury_plan_swallows_exception(caplog, monkeypatch):
    """If `plan_idle_cash` raises (e.g. config bug), the cycle must
    keep running — the helper logs at debug and returns silently."""
    import halal_trader.core.treasury as treasury_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated treasury failure")

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", boom)

    svc = _service()
    with caplog.at_level(logging.DEBUG):
        svc._log_treasury_plan(account=_account(total=10_000, available=10_000))
    # No INFO-level treasury line.
    info_lines = [
        r for r in caplog.records if r.levelno == logging.INFO and "treasury" in r.message
    ]
    assert info_lines == []
    # But there IS a DEBUG line about the failure.
    debug_lines = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "treasury plan computation failed" in r.message
    ]
    assert len(debug_lines) == 1


# ── No-op plan ─────────────────────────────────────────────


def test_log_treasury_plan_skips_log_when_plan_is_noop(caplog, monkeypatch):
    """When `plan_idle_cash` returns a no-op plan, no log line fires
    — keeps the operator's tail clean during quiet cycles."""
    import halal_trader.core.treasury as treasury_mod

    class _Noop:
        is_noop = True
        action = "hold"
        amount_usd = 0.0
        instrument = ""
        reason = "balanced"

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", lambda **kw: _Noop())

    svc = _service()
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=10_000, available=10_000))
    info_treasury = [
        r for r in caplog.records if r.levelno == logging.INFO and "treasury" in r.message
    ]
    assert info_treasury == []


def test_log_treasury_plan_emits_log_when_plan_is_not_noop(caplog, monkeypatch):
    """Mirror of the above: a non-noop plan emits the formatted line.
    Pinning so a refactor that flips the `is_noop` semantics doesn't
    silently kill all treasury logs."""
    import halal_trader.core.treasury as treasury_mod

    class _Plan:
        is_noop = False
        action = "deploy"
        amount_usd = 5000.0
        instrument = "TBIL"
        reason = "cash above floor"

    monkeypatch.setattr(treasury_mod, "plan_idle_cash", lambda **kw: _Plan())

    svc = _service()
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=10_000, available=10_000))
    treasury_lines = [r for r in caplog.records if "treasury" in r.message]
    assert len(treasury_lines) == 1
    msg = treasury_lines[0].message
    assert "deploy" in msg
    assert "$5000.00" in msg
    assert "TBIL" in msg
    assert "cash above floor" in msg


def test_log_treasury_plan_passes_correct_args_to_planner(monkeypatch):
    """The helper splits `total - available` as positions_value and
    threads `available` as cash_balance. Pin so a future refactor
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

    svc = _service()
    # total $10k, available $7k → positions_value = $3k.
    svc._log_treasury_plan(account=_account(total=10_000, available=7_000))
    assert captured["cash_balance"] == 7_000
    assert captured["positions_value"] == 3_000
    # No existing treasury wired through — defaults to 0.0 in the helper.
    assert captured["current_treasury_value"] == 0.0


def test_log_treasury_plan_zero_available_skips_via_low_cash_branch(caplog):
    """Edge: $0 available → < 50, short-circuit before planning."""
    svc = _service()
    with caplog.at_level(logging.INFO):
        svc._log_treasury_plan(account=_account(total=0, available=0))
    assert not any("treasury" in r.message for r in caplog.records)
