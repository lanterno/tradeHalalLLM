"""Wave G wiring tests — live-order feature extraction and the retrainer.

The slippage model itself is covered by ``test_slippage_model.py``.
These tests cover ``features_from_live_order`` and the retrainer that
fits + persists a model from recent filled trades.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from halal_trader.ml.slippage import features_from_live_order

# ── features_from_live_order ────────────────────────────────────


def test_features_from_live_order_derives_spread_from_orderbook() -> None:
    """When ``spread_bps`` is missing from indicators, derive it from
    the orderbook's top of book."""
    feats = features_from_live_order(
        size_usd=1000.0,
        indicators={"atr_14": 100.0, "rsi_14": 60.0},
        price=50_000.0,
        orderbook={"bids": [[49_995.0, 1.0]], "asks": [[50_005.0, 1.0]]},
    )
    assert feats["spread_bps"] == pytest.approx(2.0, rel=0.05)  # ~2 bps
    assert feats["atr_pct"] == pytest.approx(100.0 / 50_000.0)
    assert feats["rsi_14"] == 60.0


def test_features_from_live_order_no_orderbook_zero_spread() -> None:
    """Without an orderbook the spread falls back to whatever the
    indicators provide (here: nothing → 0)."""
    feats = features_from_live_order(
        size_usd=500.0,
        indicators={"atr_14": 50.0},
        price=10_000.0,
    )
    assert feats["spread_bps"] == 0.0


# ── RetrainingScheduler slippage refit ───────────────────────────


@pytest.mark.asyncio
async def test_retrainer_slippage_refit_skipped_without_trade_repo(tmp_path) -> None:
    """No crypto_trade_repo → slippage refit returns False without
    touching disk; the existing anomaly/classifier path is untouched."""
    from halal_trader.ml.retrainer import RetrainingScheduler

    snap_repo = AsyncMock()
    rs = RetrainingScheduler(snap_repo, models_dir=tmp_path)
    assert await rs._retrain_slippage() is False


@pytest.mark.asyncio
async def test_retrainer_slippage_refit_persists_when_enough_samples(tmp_path) -> None:
    """With ≥30 valid samples the retrainer fits a model and writes it
    to ``models/<namespace>/slippage_v1.json``."""
    from halal_trader.ml.retrainer import RetrainingScheduler

    snap_repo = AsyncMock()
    snap_repo.get_labeled_snapshots = AsyncMock(
        return_value=[
            {
                "trade_id": i,
                "rsi_14": 50.0,
                "atr_14": 100.0,
                "volume_ratio": 1.0,
            }
            for i in range(40)
        ]
    )
    trade_repo = AsyncMock()
    trade_repo.get_filled_trades = AsyncMock(
        return_value=[
            {
                "id": i,
                "price": 100.0,
                "filled_price": 100.0 * (1 + 0.0005 + i * 1e-5),  # adverse trend
                "filled_quantity": 1.0,
                "quantity": 1.0,
                "timestamp": "2026-05-18T12:00:00+00:00",
            }
            for i in range(40)
        ]
    )

    rs = RetrainingScheduler(
        snap_repo,
        models_dir=tmp_path,
        crypto_trade_repo=trade_repo,
    )
    ok = await rs._retrain_slippage()
    assert ok is True
    persisted = tmp_path / "crypto" / "slippage_v1.json"
    assert persisted.exists()


@pytest.mark.asyncio
async def test_retrainer_slippage_refit_handles_repo_failure(tmp_path) -> None:
    """A DB hiccup on the trade fetch returns False; cycle continues."""
    from halal_trader.ml.retrainer import RetrainingScheduler

    snap_repo = AsyncMock()
    trade_repo = AsyncMock()
    trade_repo.get_filled_trades = AsyncMock(side_effect=RuntimeError("DB lost"))
    rs = RetrainingScheduler(
        snap_repo,
        models_dir=tmp_path,
        crypto_trade_repo=trade_repo,
    )
    assert await rs._retrain_slippage() is False
