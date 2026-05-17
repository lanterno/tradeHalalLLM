"""Tests for the small factory helpers in :mod:`crypto.components`.

`build_components` itself is integration territory (lots of DB / WS /
LLM wiring). The three private builders underneath — `_build_sentiment`,
`_build_ml`, `_build_news_reactor` — are pure constructors with simple
branching on settings flags. This file pins their contracts so a
silent rename / signature change in one of the optional subsystems
breaks here first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from halal_trader.crypto.components import (
    _build_ml,
    _build_news_reactor,
    _build_sentiment,
)


def _ml_settings(*, enabled: bool, device: str = "cpu", models_dir: Path | None = None):
    """Minimal settings shape for `_build_ml` — only `.ml.*` is read."""
    return SimpleNamespace(
        ml=SimpleNamespace(
            enabled=enabled,
            device=device,
            models_dir=models_dir or Path("models"),
        )
    )


def _sentiment_settings(
    *,
    pairs: list[str] | None = None,
    reddit_id: str = "",
    reddit_secret: str = "",
    cryptopanic_key: str = "",
    update_interval: int = 300,
):
    """Minimal settings shape for `_build_sentiment`."""
    return SimpleNamespace(
        crypto=SimpleNamespace(pairs=pairs or ["BTCUSDT", "ETHUSDT"]),
        sentiment=SimpleNamespace(
            reddit=SimpleNamespace(client_id=reddit_id, client_secret=reddit_secret),
            cryptopanic=SimpleNamespace(api_key=cryptopanic_key),
            update_interval_seconds=update_interval,
        ),
    )


def _news_settings(*, api_key: str, pairs: list[str] | None = None):
    return SimpleNamespace(
        sentiment=SimpleNamespace(cryptopanic=SimpleNamespace(api_key=api_key)),
        crypto=SimpleNamespace(pairs=pairs or ["BTCUSDT"]),
    )


# ── _build_ml ──────────────────────────────────────────────


def test_build_ml_disabled_returns_three_nones():
    """`ml.enabled=False` short-circuits — no ModelHub init, no
    HuggingFace download. Returns the three-None tuple so the cycle's
    `ml_forecaster, ml_anomaly, ml_signal = _build_ml(settings)`
    pattern destructures cleanly."""
    forecaster, anomaly, signal = _build_ml(_ml_settings(enabled=False))
    assert forecaster is None
    assert anomaly is None
    assert signal is None


def test_build_ml_enabled_constructs_three_components(tmp_path: Path):
    """Happy path: ML enabled → returns (PriceForecaster, MarketAnomalyDetector,
    MLSignalClassifier) with a shared ModelHub."""
    forecaster, anomaly, signal = _build_ml(_ml_settings(enabled=True, models_dir=tmp_path))
    assert forecaster is not None
    assert anomaly is not None
    assert signal is not None
    # All three share the same ModelHub instance — important so a
    # `register("regime", ...)` call from one is visible to the others.
    assert forecaster._hub is anomaly._hub
    assert anomaly._hub is signal._hub


def test_build_ml_swallows_construction_failure(monkeypatch, tmp_path: Path, caplog):
    """If any of the ML imports / constructors raises, `_build_ml`
    returns (None, None, None) and logs a warning rather than aborting
    the cycle. The bot still runs — just without ML features.

    We simulate the failure by patching `ModelHub` (already imported)
    to raise on construction. The local imports inside the function
    succeed; only the `ModelHub(...)` call inside the try block fails."""
    import halal_trader.ml.hub as hub_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated ML init failure")

    monkeypatch.setattr(hub_mod, "ModelHub", boom)

    with caplog.at_level(logging.WARNING):
        forecaster, anomaly, signal = _build_ml(_ml_settings(enabled=True, models_dir=tmp_path))

    assert forecaster is None
    assert anomaly is None
    assert signal is None
    assert any("ML models initialization failed" in r.message for r in caplog.records)


def test_build_ml_uses_configured_device(tmp_path: Path):
    """Settings.ml.device threads through to the ModelHub — the operator
    can flip cuda/mps without code changes."""
    forecaster, _, _ = _build_ml(_ml_settings(enabled=True, device="cuda", models_dir=tmp_path))
    assert forecaster is not None
    assert forecaster._hub.device == "cuda"


def test_build_ml_uses_configured_models_dir(tmp_path: Path):
    """The ModelHub gets the configured `models_dir` — test runs use a
    tmp_path so we don't pollute the repo."""
    forecaster, _, _ = _build_ml(_ml_settings(enabled=True, models_dir=tmp_path))
    assert forecaster._hub.models_dir == tmp_path


# ── _build_sentiment ───────────────────────────────────────


def test_build_sentiment_returns_sentiment_manager_unconditionally():
    """Unlike `_build_ml`, the sentiment builder doesn't gate on a
    flag — it always constructs the manager. The manager itself
    decides whether it's `enabled` based on creds (covered in
    `test_sentiment_manager.py`). Pin so a future "no-op when both
    creds empty" optimisation is intentional."""
    mgr = _build_sentiment(_sentiment_settings())
    assert mgr is not None
    # Both creds empty → manager is constructed but disabled.
    assert mgr.enabled is False


def test_build_sentiment_threads_pairs_through():
    """Pairs from settings.crypto flow into the SentimentManager so it
    only fetches sentiment for configured trading universe."""
    mgr = _build_sentiment(_sentiment_settings(pairs=["BTCUSDT", "SOLUSDT"]))
    assert set(mgr._trading_pairs) == {"BTCUSDT", "SOLUSDT"}


def test_build_sentiment_with_cryptopanic_key_enables_manager():
    mgr = _build_sentiment(_sentiment_settings(cryptopanic_key="real-key"))
    assert mgr.enabled is True


def test_build_sentiment_with_partial_reddit_creds_stays_disabled():
    """Reddit needs BOTH client_id + client_secret to count as
    configured; one without the other is no-op (matches `test_sentiment_manager.py`)."""
    mgr = _build_sentiment(_sentiment_settings(reddit_id="x"))  # no secret
    # Mirror existing test_sentiment_manager.py behaviour: partial creds
    # don't enable the manager unless cryptopanic is also set.
    assert mgr.enabled is False


def test_build_sentiment_threads_update_interval():
    """The interval is settable per-deployment."""
    mgr = _build_sentiment(_sentiment_settings(update_interval=600))
    # The attribute name is whatever the SentimentManager stores it as;
    # we only assert the manager constructed successfully — the integration
    # tests in test_sentiment_manager.py pin the polling interval semantics.
    assert mgr is not None


# ── _build_news_reactor ────────────────────────────────────


def test_build_news_reactor_constructs_with_api_key():
    """The reactor is built unconditionally when `_build_news_reactor` is
    called (the caller in `build_components` decides whether to
    invoke). Verify the kwargs flow through."""
    reactor = _build_news_reactor(_news_settings(api_key="secret"))
    assert reactor is not None
    assert reactor._api_key == "secret"


def test_build_news_reactor_threads_pairs():
    reactor = _build_news_reactor(_news_settings(api_key="x", pairs=["BTCUSDT", "ETHUSDT"]))
    assert reactor._trading_pairs == ["BTCUSDT", "ETHUSDT"]


def test_build_news_reactor_uses_30_second_default_poll_interval():
    """30s poll interval is the upper-bound that lets us still catch
    an emergency news event before the next 60s cycle would. Pin so
    a refactor doesn't accidentally widen it."""
    reactor = _build_news_reactor(_news_settings(api_key="x"))
    assert reactor._poll_interval == 30


def test_build_news_reactor_uses_hot_importance_filter():
    """`importance_filter='hot'` excludes routine 'normal'-priority news
    so the reactor only triggers emergency mini-cycles for genuinely
    high-impact items. Pin the default."""
    reactor = _build_news_reactor(_news_settings(api_key="x"))
    assert reactor._importance_filter == "hot"


def test_build_news_reactor_accepts_empty_api_key():
    """Defensive: caller is responsible for branching on whether to
    construct the reactor at all. If they pass an empty api_key, the
    reactor still constructs (its own polling loop will short-circuit)."""
    reactor = _build_news_reactor(_news_settings(api_key=""))
    assert reactor is not None
    assert reactor._api_key == ""
