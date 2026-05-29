"""Outcome attribution — per-regime / per-source win-rate from closed outcomes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa

from halabot.analysis.attribution import attribution
from halabot.platform.db import open_position as _open_position
from halabot.platform.db import outcome as _outcome

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


async def _insert(engine, *, return_pct, label, regime, sources, i):
    async with engine.begin() as conn:
        await conn.execute(
            sa.insert(_outcome).values(
                asset="NVDA", entry_ts=T0, exit_ts=T0 + timedelta(minutes=i),
                entry_price=100.0, exit_price=100.0 * (1 + return_pct), closed_weight=0.1,
                return_pct=return_pct, hold_seconds=60, belief_version=1,
                entry_belief={"regime": regime, "sources": sources},
                label=label, reason="test", created_at=T0,
            )
        )


async def _insert_open(engine, *, asset, unrealized, regime, sources):
    async with engine.begin() as conn:
        await conn.execute(
            sa.insert(_open_position).values(
                asset=asset, entry_ts=T0, entry_vwap=100.0, weight=0.1,
                last_price=100.0 * (1 + unrealized), unrealized_return_pct=unrealized,
                belief_version=1, entry_belief={"regime": regime, "sources": sources},
                updated_at=T0,
            )
        )


@pytest.mark.asyncio
async def test_attribution_buckets_by_regime_and_source(halabot_engine):
    # news-driven winners, momentum-only losers.
    await _insert(halabot_engine, return_pct=0.03, label=1, regime="trending_up",
                  sources=["news", "indicator.momentum"], i=0)
    await _insert(halabot_engine, return_pct=0.02, label=1, regime="trending_up",
                  sources=["news", "indicator.momentum"], i=1)
    await _insert(halabot_engine, return_pct=-0.01, label=0, regime="ranging",
                  sources=["indicator.momentum"], i=2)

    attr = await attribution(halabot_engine)
    assert attr.total == 3
    regimes = {b.key: b for b in attr.by_regime}
    assert regimes["trending_up"].win_rate == 1.0
    assert regimes["ranging"].win_rate == 0.0

    sources = {b.key: b for b in attr.by_source}
    # news appears only on the two winners → 100% win; momentum on all 3 → 2/3.
    assert sources["news"].n == 2 and sources["news"].win_rate == 1.0
    assert sources["indicator.momentum"].n == 3
    assert sources["indicator.momentum"].win_rate == pytest.approx(2 / 3)
    # Sorted by avg return descending — news (winners only) ranks first.
    assert attr.by_source[0].key == "news"


@pytest.mark.asyncio
async def test_attribution_min_n_filters(halabot_engine):
    await _insert(halabot_engine, return_pct=0.01, label=1, regime="trending_up",
                  sources=["news"], i=0)
    attr = await attribution(halabot_engine, min_n=2)
    assert attr.by_source == []  # the single news trade is below min_n


@pytest.mark.asyncio
async def test_attribution_includes_open_positions_bias_corrected(halabot_engine):
    # Closed-only would show a 0% win rate in trending_up (the one realized trade
    # lost); the held OPEN winner corrects that survivorship bias.
    await _insert(halabot_engine, return_pct=-0.01, label=0, regime="trending_up",
                  sources=["indicator.momentum"], i=0)
    await _insert_open(halabot_engine, asset="AMD", unrealized=0.05,
                       regime="trending_up", sources=["forecaster"])
    attr = await attribution(halabot_engine, win_threshold_pct=0.002)
    assert attr.total == 1 and attr.open_count == 1
    regimes = {b.key: b for b in attr.by_regime}
    assert regimes["trending_up"].n == 2  # 1 closed loss + 1 open win
    assert regimes["trending_up"].win_rate == 0.5  # bias-corrected (was 0% closed-only)
    sources = {b.key: b for b in attr.by_source}
    assert sources["forecaster"].n == 1 and sources["forecaster"].win_rate == 1.0


@pytest.mark.asyncio
async def test_attribution_can_exclude_open(halabot_engine):
    await _insert_open(halabot_engine, asset="AMD", unrealized=0.05,
                       regime="trending_up", sources=["forecaster"])
    attr = await attribution(halabot_engine, include_open=False)
    assert attr.open_count == 0
    assert all(b.key != "forecaster" for b in attr.by_source)  # open excluded
