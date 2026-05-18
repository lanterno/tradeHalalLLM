"""Wave G wiring tests — executor, backtester, prompt stage, retrainer.

The slippage model itself is covered by ``test_slippage_model.py``.
These tests cover the *consumers* of the model: the executor writes
``predicted_slippage_pct`` on the trade row; the simulated backtest
executor reads the model's prediction instead of the constant
baseline; the new ``BuildSlippageTextStage`` formats a one-line-per-pair
block for the prompt; the retrainer fits + persists a model from
recent filled trades.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from halal_trader.ml.slippage import SlippageModel, features_from_live_order


def _model_with_intercept(pct: float) -> SlippageModel:
    """A model that always predicts ``pct`` (intercept only, no learned coefs)."""
    return SlippageModel(
        coefs=dict.fromkeys(SlippageModel.identity().coefs.keys(), 0.0),
        intercept=pct,
        n_samples=10,
        feature_means=dict.fromkeys(SlippageModel.identity().coefs.keys(), 0.0),
    )


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


# ── BuildSlippageTextStage ──────────────────────────────────────


@pytest.mark.asyncio
async def test_slippage_text_stage_skips_when_no_model() -> None:
    """No model wired → no slippage block in the prompt."""
    from halal_trader.core.cycle_pipeline import CycleState
    from halal_trader.core.cycle_stages import BuildSlippageTextStage

    state = CycleState(halal_pairs=["BTCUSDT"])
    out = await BuildSlippageTextStage(slippage_model=None, max_position_pct=0.25).run(state)
    assert out.slippage_text == ""


@pytest.mark.asyncio
async def test_slippage_text_stage_formats_per_pair_predictions() -> None:
    """When the model is wired, the stage emits one line per pair
    with indicators + klines, in basis points."""
    from halal_trader.core.cycle_pipeline import CycleState
    from halal_trader.core.cycle_stages import BuildSlippageTextStage
    from halal_trader.domain.models import CryptoAccount, Kline

    model = _model_with_intercept(0.0008)  # 8 bps
    kline = Kline(
        open_time=1,
        open=50_000.0,
        high=50_100.0,
        low=49_900.0,
        close=50_000.0,
        volume=1.0,
        close_time=2,
    )
    state = CycleState(
        account=CryptoAccount(
            total_balance_usdt=10_000.0,
            available_balance_usdt=10_000.0,
            in_order_usdt=0.0,
            usdt_free=10_000.0,
        ),
        halal_pairs=["BTCUSDT"],
        indicators_cache={"BTCUSDT": {"rsi_14": 60.0, "atr_14": 100.0}},
        klines_by_symbol={"BTCUSDT": [kline]},
    )
    out = await BuildSlippageTextStage(slippage_model=model, max_position_pct=0.25).run(state)
    assert "BTCUSDT" in out.slippage_text
    assert "bps" in out.slippage_text
    # Stage name is part of the prom-histogram label.
    assert BuildSlippageTextStage(None, max_position_pct=0.25).name == "build_slippage_text"


@pytest.mark.asyncio
async def test_slippage_text_stage_skips_pairs_without_indicators() -> None:
    """A pair with no indicators in the cache → silently skipped, no
    crash; the prompt only mentions the pairs that have full inputs."""
    from halal_trader.core.cycle_pipeline import CycleState
    from halal_trader.core.cycle_stages import BuildSlippageTextStage
    from halal_trader.domain.models import CryptoAccount, Kline

    kline = Kline(
        open_time=1,
        open=50_000.0,
        high=50_100.0,
        low=49_900.0,
        close=50_000.0,
        volume=1.0,
        close_time=2,
    )
    state = CycleState(
        account=CryptoAccount(
            total_balance_usdt=10_000.0,
            available_balance_usdt=10_000.0,
            in_order_usdt=0.0,
            usdt_free=10_000.0,
        ),
        halal_pairs=["BTCUSDT", "ETHUSDT"],
        indicators_cache={"BTCUSDT": {"rsi_14": 60.0, "atr_14": 100.0}},  # only one pair
        klines_by_symbol={"BTCUSDT": [kline], "ETHUSDT": [kline]},
    )
    model = _model_with_intercept(0.0005)
    out = await BuildSlippageTextStage(slippage_model=model, max_position_pct=0.25).run(state)
    assert "BTCUSDT" in out.slippage_text
    assert "ETHUSDT" not in out.slippage_text


# ── Backtest SimulatedExecutor wiring ───────────────────────────


def test_simulated_executor_uses_model_baseline_when_provided() -> None:
    """The backtester should read the model's prediction instead of
    the constant ``slippage_pct`` baseline."""
    from halal_trader.crypto.backtest import SimulatedExecutor

    model = _model_with_intercept(0.0010)  # 10 bps
    exec_with_model = SimulatedExecutor(slippage_pct=0.0001, slippage_model=model)
    exec_constant = SimulatedExecutor(slippage_pct=0.0001)

    # Same inputs; different baseline produces different fill price.
    fp_model = exec_with_model._fill_price(
        side="buy", price=10_000.0, notional_usd=1_000.0, atr_pct=0.02
    )
    fp_constant = exec_constant._fill_price(
        side="buy", price=10_000.0, notional_usd=1_000.0, atr_pct=0.02
    )
    assert fp_model > fp_constant  # 10 bps > 1 bps adverse on a buy
    # And it's roughly 10 bps above the intent.
    assert (fp_model - 10_000.0) / 10_000.0 == pytest.approx(0.001, rel=0.2)


def test_simulated_executor_model_failure_falls_back_to_constant() -> None:
    """If predict() raises, the executor falls back to the constant —
    a model bug must never block a backtest."""
    from halal_trader.crypto.backtest import SimulatedExecutor

    broken = MagicMock()
    broken.predict.side_effect = RuntimeError("boom")
    se = SimulatedExecutor(slippage_pct=0.0005, slippage_model=broken)
    baseline = se._baseline_slippage_for(notional_usd=100.0, atr_pct=0.01, price=10_000.0)
    assert baseline == 0.0005  # fell back to constant


# ── CryptoExecutor._predict_slippage helper ─────────────────────


def test_executor_predict_slippage_returns_none_without_model() -> None:
    """No model → no prediction recorded on the trade row."""
    from halal_trader.crypto.executor import CryptoExecutor

    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
    )
    out = ex._predict_slippage(
        symbol="BTCUSDT",
        price=50_000.0,
        quantity=0.01,
        indicators_cache={"BTCUSDT": {"rsi_14": 50.0}},
        orderbooks={},
    )
    assert out is None


def test_executor_predict_slippage_uses_model_when_wired() -> None:
    """Wired model + non-empty indicators → numeric prediction."""
    from halal_trader.crypto.executor import CryptoExecutor

    model = _model_with_intercept(0.0007)
    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
        slippage_model=model,
    )
    out = ex._predict_slippage(
        symbol="BTCUSDT",
        price=50_000.0,
        quantity=0.01,
        indicators_cache={"BTCUSDT": {"rsi_14": 60.0, "atr_14": 100.0}},
        orderbooks={"BTCUSDT": {"bids": [[49_999.0, 1.0]], "asks": [[50_001.0, 1.0]]}},
    )
    assert out is not None
    assert 0.0001 < out < 0.01


def test_executor_predict_slippage_returns_none_without_indicators() -> None:
    """A symbol with no indicators → skip prediction (don't make
    something up from feature_means alone)."""
    from halal_trader.crypto.executor import CryptoExecutor

    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
        slippage_model=_model_with_intercept(0.0005),
    )
    assert (
        ex._predict_slippage(
            symbol="BTCUSDT",
            price=50_000.0,
            quantity=0.01,
            indicators_cache={},
            orderbooks={},
        )
        is None
    )


def test_executor_predict_slippage_swallows_model_failure() -> None:
    """A broken model returning garbage → None, never blocks the fill."""
    from halal_trader.crypto.executor import CryptoExecutor

    broken = MagicMock()
    broken.predict.side_effect = RuntimeError("model corrupt")
    ex = CryptoExecutor(
        broker=MagicMock(),
        repo=MagicMock(),
        max_position_pct=0.25,
        max_simultaneous_positions=5,
        slippage_model=broken,
    )
    out = ex._predict_slippage(
        symbol="BTCUSDT",
        price=50_000.0,
        quantity=0.01,
        indicators_cache={"BTCUSDT": {"rsi_14": 50.0}},
        orderbooks={},
    )
    assert out is None


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
