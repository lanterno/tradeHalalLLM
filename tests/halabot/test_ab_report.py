"""ab_report — shadow proposals vs live trades over a window (PG; :5433)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from halabot.analysis.ab_report import ab_report
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import PgEventLog
from halabot.platform.events import EventType, new_event

NOW = datetime(2026, 5, 28, 18, 0, tzinfo=UTC)


async def _seed_shadow(engine, proposals: list[tuple[str, str]], *, at: datetime):
    log = PgEventLog(engine)
    clock = FakeClock(at)
    for asset, side in proposals:
        await log.append(
            new_event(
                clock, EventType.POLICY_TRADE_PROPOSED, source="policy.shadow", asset=asset,
                payload={"side": side, "shadow": True},
            )
        )


async def _seed_live_trades(engine, trades: list[tuple[str, str]], *, at: datetime):
    async with engine.begin() as conn:
        for symbol, side in trades:
            await conn.execute(
                sa.text(
                    "INSERT INTO trades (symbol, side, quantity, status, timestamp) "
                    "VALUES (:s, :side, 1, 'filled', :ts)"
                ),
                {"s": symbol, "side": side, "ts": at},
            )


@pytest.mark.asyncio
async def test_counts_shadow_and_live(halabot_engine):
    await _seed_shadow(halabot_engine, [("NVDA", "buy"), ("NOW", "buy")], at=NOW)
    await _seed_live_trades(
        halabot_engine,
        [("NVDA", "buy"), ("NVDA", "sell"), ("MSFT", "buy"), ("SHOP", "buy"), ("SHOP", "sell")],
        at=NOW,
    )
    rep = await ab_report(
        halabot_engine, since=NOW - timedelta(hours=1), until=NOW + timedelta(hours=1)
    )
    assert rep.shadow_total == 2
    assert rep.live_total == 5
    assert rep.shadow_by_symbol == {"NVDA": 1, "NOW": 1}
    assert rep.live_by_symbol == {"NVDA": 2, "MSFT": 1, "SHOP": 2}


@pytest.mark.asyncio
async def test_churn_reduction_and_live_only_symbols(halabot_engine):
    await _seed_shadow(halabot_engine, [("NVDA", "buy")], at=NOW)
    await _seed_live_trades(
        halabot_engine, [("NVDA", "buy"), ("MSFT", "buy"), ("SHOP", "buy"), ("CSCO", "buy")], at=NOW
    )
    rep = await ab_report(
        halabot_engine, since=NOW - timedelta(hours=1), until=NOW + timedelta(hours=1)
    )
    # shadow proposed 1 vs live 4 → 75% fewer trades
    assert rep.churn_reduction_pct == pytest.approx(0.75)
    assert rep.symbols_only_live == {"MSFT", "SHOP", "CSCO"}


@pytest.mark.asyncio
async def test_window_excludes_out_of_range(halabot_engine):
    await _seed_shadow(halabot_engine, [("NVDA", "buy")], at=NOW - timedelta(days=3))  # old
    await _seed_live_trades(halabot_engine, [("MSFT", "buy")], at=NOW)
    rep = await ab_report(
        halabot_engine, since=NOW - timedelta(hours=1), until=NOW + timedelta(hours=1)
    )
    assert rep.shadow_total == 0  # the 3-day-old proposal is outside the window
    assert rep.live_total == 1


async def _seed_outcomes(engine, returns: list[tuple[str, float]], *, at: datetime):
    from halabot.platform.db import outcome as o

    async with engine.begin() as conn:
        for asset, ret in returns:
            await conn.execute(
                sa.insert(o).values(
                    asset=asset, entry_ts=at, exit_ts=at, entry_price=100.0,
                    exit_price=100.0 * (1 + ret), closed_weight=0.1, return_pct=ret,
                    hold_seconds=3600, belief_version=1, entry_belief=None,
                    label=1 if ret > 0.002 else 0, reason="test", created_at=at,
                )
            )


@pytest.mark.asyncio
async def test_report_includes_shadow_hypothetical_pnl(halabot_engine):
    await _seed_outcomes(halabot_engine, [("NVDA", 0.10), ("NOW", 0.05), ("SHOP", -0.04)], at=NOW)
    rep = await ab_report(
        halabot_engine, since=NOW - timedelta(hours=1), until=NOW + timedelta(hours=1)
    )
    assert rep.shadow_closed == 3
    assert rep.shadow_avg_return_pct == pytest.approx((0.10 + 0.05 - 0.04) / 3)
    assert rep.shadow_win_rate == pytest.approx(2 / 3)  # 2 of 3 above threshold
    assert rep.shadow_weighted_return == pytest.approx((0.10 + 0.05 - 0.04) * 0.1)


@pytest.mark.asyncio
async def test_no_live_trades_gives_none_churn(halabot_engine):
    await _seed_shadow(halabot_engine, [("NVDA", "buy")], at=NOW)
    rep = await ab_report(
        halabot_engine, since=NOW - timedelta(hours=1), until=NOW + timedelta(hours=1)
    )
    assert rep.live_total == 0
    assert rep.churn_reduction_pct is None  # undefined with no live baseline


@pytest.mark.asyncio
async def test_promotion_gate_holds_on_thin_data(halabot_engine):
    """With few outcomes + no live P&L, the Phase-3 gate stays HOLD (data-gated)."""
    await _seed_outcomes(halabot_engine, [("NVDA", 0.10), ("NOW", 0.05)], at=NOW)
    rep = await ab_report(
        halabot_engine, since=NOW - timedelta(hours=1), until=NOW + timedelta(hours=1)
    )
    assert rep.promotion is not None
    assert rep.promotion.promote is False  # not enough samples to promote
    assert rep.shadow_return_std is not None  # variance reported
