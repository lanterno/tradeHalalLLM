"""Wave B stage classes — each owns one prompt-context block.

A ``CycleStage`` takes a :class:`CycleState`, mutates one or two
fields, returns the state. The cycle service holds an ordered list of
stages and runs them. New prompt-context sources land as one new file
under this module + one new line on the cycle's stage list.

Both the crypto and stock cycles now drive a single stage list via
:func:`core.cycle_pipeline.run_stages` end-to-end. The 18 classes
defined here cover every prompt-context block the LLMs see — twelve
``BuildXStage`` builders, four ``Augment*Stage`` augmenters,
:class:`ApplyRegimeGateStage` (post-analyze BUY veto in confirmed
downtrends), and :class:`BuildForecastsStage` (Chronos forecaster).
Stage exceptions are swallowed by the ``run_stages`` driver so a
regional failure can't abort the cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from halal_trader.core.cycle_pipeline import CycleState

logger = logging.getLogger(__name__)


class CycleStage(Protocol):
    """Protocol every Wave B stage class satisfies."""

    name: str

    async def run(self, state: CycleState) -> CycleState: ...


# ── Build-regime stage ───────────────────────────────────────────


class BuildRegimeStage:
    """Run the regime detector and stamp ``state.regime_text``.

    No-op when no detector is wired; failure-handling lives in the
    underlying :func:`crypto.regime.build_regime_text` helper so the
    stage stays a one-liner.
    """

    name = "build_regime_text"

    def __init__(self, detector: Any | None) -> None:
        self._detector = detector

    async def run(self, state: CycleState) -> CycleState:
        from halal_trader.crypto.regime import build_regime_text

        state.regime_text = build_regime_text(self._detector, state.indicators_cache)
        return state


# ── Build-ML-signals stage ───────────────────────────────────────


class BuildForecastsStage:
    """Run the price forecaster (Chronos) and stamp ``state.forecasts_text``.

    Crypto-only — daily stock bars are too sparse for Chronos's 96-step
    minimum, so the stocks cycle just doesn't include this stage. No-op
    when no forecaster is wired or no klines are available.
    """

    name = "build_forecasts_text"

    def __init__(self, forecaster: Any | None = None) -> None:
        self._forecaster = forecaster

    async def run(self, state: CycleState) -> CycleState:
        if self._forecaster is None or not state.klines_by_symbol:
            return state
        try:
            from halal_trader.ml.forecaster import format_forecasts_for_prompt

            forecasts: dict[str, Any] = {}
            for pair, klines in state.klines_by_symbol.items():
                if len(klines) < 20:
                    continue
                closes = [k.close for k in klines]
                fc = self._forecaster.forecast(pair, closes)
                if fc:
                    forecasts[pair] = fc
            state.forecasts_text = format_forecasts_for_prompt(forecasts)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Forecaster unavailable: %s", exc)
        return state


class BuildMlSignalsStage:
    """Run anomaly + signal classifier inference and stamp ``state.ml_signals_text``.

    Reads any pre-computed ``state.forecasts_text`` (populated by
    :class:`BuildForecastsStage`) and threads it through the shared
    formatter so a single block carries both the anomaly/confidence
    output and the Chronos forecast block.
    """

    name = "build_ml_signals_text"

    def __init__(
        self,
        anomaly_detector: Any | None = None,
        signal_classifier: Any | None = None,
    ) -> None:
        self._anomaly = anomaly_detector
        self._signal = signal_classifier

    async def run(self, state: CycleState) -> CycleState:
        from halal_trader.ml.anomaly import build_ml_signals_text

        state.ml_signals_text = build_ml_signals_text(
            indicators_by_symbol=state.indicators_cache,
            anomaly_detector=self._anomaly,
            signal_classifier=self._signal,
            forecasts_text=state.forecasts_text,
        )
        return state


# ── Build-timeframe stage ────────────────────────────────────────


class BuildTimeframeStage:
    """Run the multi-timeframe analyzer and stamp ``state.timeframe_text``."""

    name = "build_timeframe_text"

    def __init__(self, analyzer: Any | None) -> None:
        self._analyzer = analyzer

    async def run(self, state: CycleState) -> CycleState:
        from halal_trader.crypto.timeframes import build_timeframe_text

        state.timeframe_text = await build_timeframe_text(
            self._analyzer, state.halal_pairs
        )
        return state


# ── Build-performance stage ──────────────────────────────────────


class BuildPerformanceStage:
    """Run the rolling performance summary and stamp ``state.performance_text``.

    Reads from any analytics impl that exposes
    ``compute_stats(lookback_days=...)`` and ``format_for_prompt(stats)`` —
    both ``crypto.analytics.PerformanceAnalytics`` and
    ``core.analytics.CrossAssetAnalytics`` satisfy the shape.
    """

    name = "build_performance_text"

    def __init__(self, analytics: Any | None, *, lookback_days: int = 7) -> None:
        self._analytics = analytics
        self._lookback_days = lookback_days

    async def run(self, state: CycleState) -> CycleState:
        if self._analytics is None:
            state.performance_text = ""
            return state
        try:
            stats = await self._analytics.compute_stats(lookback_days=self._lookback_days)
            state.performance_text = self._analytics.format_for_prompt(stats)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Performance stats unavailable: %s", exc)
            state.performance_text = ""
        return state


# ── Build-active-adjustments stage ───────────────────────────────


class BuildActiveAdjustmentsStage:
    """Surface any persisted self-improvement knob overrides to the prompt.

    The crypto cycle's ``TradeSelfReview`` exposes
    ``format_adjustments_for_prompt()``; the stage just calls that and
    stamps ``state.active_adjustments``. No-op when no reviewer is wired.
    """

    name = "build_active_adjustments"

    def __init__(self, self_review: Any | None) -> None:
        self._self_review = self_review

    async def run(self, state: CycleState) -> CycleState:
        if self._self_review is None:
            state.active_adjustments = ""
            return state
        try:
            state.active_adjustments = self._self_review.format_adjustments_for_prompt()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Active adjustments unavailable: %s", exc)
            state.active_adjustments = ""
        return state


# ── Build-exchange-rules stage ───────────────────────────────────


class BuildExchangeRulesStage:
    """Surface broker-side trading constraints to the prompt.

    The crypto bot's ``BinanceClient`` exposes `format_filters_for_prompt`
    which renders the latest min-notional, lot-size, and tick-size
    constraints from Binance's exchangeInfo. No-op for brokers that
    don't expose the helper.
    """

    name = "build_exchange_rules_text"

    def __init__(self, broker: Any | None) -> None:
        self._broker = broker

    async def run(self, state: CycleState) -> CycleState:
        if self._broker is None or not hasattr(self._broker, "format_filters_for_prompt"):
            state.exchange_rules_text = ""
            return state
        try:
            state.exchange_rules_text = self._broker.format_filters_for_prompt()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Exchange rules unavailable: %s", exc)
            state.exchange_rules_text = ""
        return state


# ── Build-catalysts stage (stocks) ───────────────────────────────


class BuildCatalystsStage:
    """Stocks-side catalyst feed → ``state.catalysts_text``.

    Pulls news / earnings / insider events from the configured
    ``StockCatalystFeed`` (or any object exposing ``fetch_all``) and
    formats them via :func:`trading.catalysts.format_catalysts_for_prompt`.
    No-op when no feed is wired.
    """

    name = "build_catalysts_text"

    def __init__(self, feed: Any | None) -> None:
        self._feed = feed

    async def run(self, state: CycleState) -> CycleState:
        if self._feed is None or not state.halal_pairs:
            state.catalysts_text = ""
            return state
        try:
            from halal_trader.trading.catalysts import format_catalysts_for_prompt

            cats = await self._feed.fetch_all(state.halal_pairs)
            state.catalysts_text = format_catalysts_for_prompt(cats, symbols=state.halal_pairs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Catalyst feed unavailable: %s", exc)
            state.catalysts_text = ""
        return state


# ── Build-microstructure stage (crypto) ──────────────────────────


class BuildMicrostructureStage:
    """Format orderbook depth-imbalance / spread features per pair.

    Reads from ``state.orderbooks`` (populated by an upstream fetch
    stage) and stamps ``state.microstructure_text``. The optional
    basis + whale-flow augmentations live outside the stage — those
    are crypto-only and will land as separate stages once their
    fetchers are also CycleStage-shaped.
    """

    name = "build_microstructure_text"

    async def run(self, state: CycleState) -> CycleState:
        if not state.orderbooks:
            state.microstructure_text = ""
            return state
        try:
            from halal_trader.crypto.microstructure import (
                format_microstructure_for_prompt,
                orderbook_features,
            )

            lines: list[str] = []
            for pair, book in sorted(state.orderbooks.items()):
                feats = orderbook_features(book)
                line = format_microstructure_for_prompt(pair=pair, book=feats)
                if line:
                    lines.append(line)
            state.microstructure_text = "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Microstructure unavailable: %s", exc)
            state.microstructure_text = ""
        return state


# ── Build-sentiment stage (crypto) ───────────────────────────────


class BuildSentimentStage:
    """Composite sentiment block: CryptoPanic + Reddit + mention velocity.

    Multi-source so the stage takes more deps than the others:

    * ``sentiment_manager`` exposes ``enabled`` and ``latest_signals``.
      Each signal carries a ``buzz`` score; values ≥3.0 trigger a
      Telegram notification via the optional ``notifier``.
    * ``reddit_fetcher`` exposes ``fetch_for_symbols(bases)`` which
      returns ``Mention`` objects. Velocity + novelty are computed via
      :func:`sentiment.velocity.compute_velocity`; the resulting dict
      is stashed on ``hub.velocity`` (the dashboard reads it from
      there) and a "Mention surges" block is appended to the
      sentiment text.

    All optional — when nothing is wired, ``state.sentiment_text``
    stays empty.
    """

    name = "build_sentiment_text"

    def __init__(
        self,
        sentiment_manager: Any | None = None,
        reddit_fetcher: Any | None = None,
        hub: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        self._sentiment = sentiment_manager
        self._reddit_fetcher = reddit_fetcher
        self._hub = hub
        self._notifier = notifier

    async def run(self, state: CycleState) -> CycleState:
        sentiment_text = ""

        # ── Composite news + Reddit signals ────────────────────
        if self._sentiment is not None and getattr(self._sentiment, "enabled", False):
            try:
                from halal_trader.sentiment.scoring import format_sentiment_for_prompt

                signals = getattr(self._sentiment, "latest_signals", None) or {}
                if signals:
                    sentiment_text = format_sentiment_for_prompt(signals)
                    await self._maybe_notify_buzz(signals)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sentiment data unavailable: %s", exc)

        # ── Reddit mention velocity surge block ────────────────
        if self._reddit_fetcher is not None and state.halal_pairs:
            try:
                from halal_trader.sentiment.velocity import (
                    compute_velocity,
                    format_velocity_for_prompt,
                )

                bases = sorted(
                    {
                        p.upper().removesuffix("USDT").removesuffix("BUSD")
                        for p in state.halal_pairs
                    }
                )
                mentions = await self._reddit_fetcher.fetch_for_symbols(bases)
                if mentions:
                    velocity = compute_velocity(mentions)
                    if self._hub is not None:
                        self._hub.velocity = velocity
                    velocity_block = format_velocity_for_prompt(velocity)
                    if velocity_block:
                        sentiment_text = (
                            sentiment_text + "\n\n" + velocity_block
                            if sentiment_text
                            else velocity_block
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Reddit velocity fetch failed: %s", exc)

        state.sentiment_text = sentiment_text
        return state

    async def _maybe_notify_buzz(self, signals: dict[str, Any]) -> None:
        """Fire Telegram alerts for any pair whose buzz score crosses 3.0."""
        if self._notifier is None:
            return
        for pair, sig in signals.items():
            buzz = getattr(sig, "buzz", 0.0)
            if buzz < 3.0:
                continue
            try:
                score = getattr(sig, "score", 0.0)
                await self._notifier.notify_buzz(pair, buzz, score, market="crypto")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to send buzz alert: %s", exc)


# ── Build-stock-risk stage ───────────────────────────────────────


class BuildStockRiskStage:
    """Stocks-side: run the shared portfolio-risk engine over Alpaca bars.

    Populates four fields:

    * ``state.risk_text`` — the prompt block.
    * ``state.indicators_cache`` — the per-symbol indicator dict that
      downstream stages (regime, ML signals) read.
    * ``state.risk_state`` — the structured ``PortfolioRiskState`` so
      the dashboard's risk panel can render heat / drawdown / correlation.
    * ``state.halt`` — mirrors ``state.risk_state.is_halted`` so the
      stocks cycle can short-circuit on a heat/drawdown breach (parity
      with :class:`BuildCryptoRiskStage`).

    Returns empty risk text on failure so the cycle never aborts on
    a transient bars-fetch hiccup.
    """

    name = "evaluate_stock_risk"

    def __init__(self, settings: Any | None = None) -> None:
        self._settings = settings

    async def run(self, state: CycleState) -> CycleState:
        if not state.bars:
            state.risk_text = ""
            state.halt = False
            return state
        try:
            from halal_trader.config import get_settings
            from halal_trader.trading.risk import evaluate_stock_risk

            equity = (
                getattr(state.account, "effective_equity", None)
                or getattr(state.account, "equity", None)
                or 0
            )
            output = evaluate_stock_risk(
                settings=self._settings or get_settings(),
                bars_by_symbol=state.bars,
                positions=state.open_positions,
                total_equity=float(equity),
            )
            state.risk_text = output.risk_text
            state.indicators_cache = output.indicators_by_symbol
            state.risk_state = output.state
            state.halt = bool(getattr(output.state, "is_halted", False))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stock risk engine evaluation failed: %s", exc)
            state.risk_text = ""
            state.halt = False
        return state


# ── Build-crypto-risk stage ──────────────────────────────────────


class BuildCryptoRiskStage:
    """Crypto-side: run the portfolio-risk engine over open trades + klines.

    Mirrors :class:`BuildStockRiskStage` but for the crypto cycle.
    Populates ``state.risk_text`` and ``state.halt`` (so the cycle can
    short-circuit on a heat / drawdown breach). Crypto-specific because
    it pulls position values from open ``CryptoTrade`` rows (which carry
    entry_price + quantity) and reads ``account.total_balance_usdt``.
    """

    name = "evaluate_portfolio_risk"

    def __init__(
        self,
        risk_engine: Any | None = None,
        broker: Any | None = None,
        open_trades: list[Any] | None = None,
    ) -> None:
        self._engine = risk_engine
        self._broker = broker
        # Stage gets ``open_trades`` separately because the crypto cycle
        # already pulled them into a typed list (rather than a generic
        # state field). Could move to ``state.open_positions`` once
        # the rest of the cycle uses CycleState.
        self._open_trades = open_trades or []

    async def run(self, state: CycleState) -> CycleState:
        if self._engine is None:
            state.risk_text = ""
            state.halt = False
            return state
        try:
            open_pos_value: dict[str, float] = {}
            unrealized_pnl: dict[str, float] = {}
            for t in self._open_trades:
                price = state.current_prices.get(t.pair)
                if price is None and self._broker is not None:
                    price = self._broker.get_cached_price(t.pair)
                if price and getattr(t, "entry_price", None):
                    open_pos_value[t.pair] = t.quantity * price
                    unrealized_pnl[t.pair] = (price - t.entry_price) * t.quantity

            equity = (
                getattr(state.account, "total_balance_usdt", None)
                or getattr(state.account, "effective_equity", None)
                or 0.0
            )
            rs = self._engine.evaluate(
                klines_by_symbol=state.klines_by_symbol,
                indicators_cache=state.indicators_cache,
                open_positions_value=open_pos_value,
                unrealized_pnl=unrealized_pnl,
                total_equity=float(equity),
            )
            state.risk_state = rs
            state.risk_text = self._engine.format_for_prompt(rs)
            state.halt = bool(getattr(rs, "is_halted", False))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Crypto risk engine evaluation failed: %s", exc)
            state.risk_text = ""
            state.halt = False
        return state


# ── Apply-regime-gate stage (crypto) ─────────────────────────────


class ApplyRegimeGateStage:
    """Strip BUY decisions for symbols in a confirmed downtrend.

    Runs after the LLM ``analyze`` call — reads ``state.plan``, queries
    the regime detector per symbol via ``state.indicators_cache``, and
    removes any buy whose symbol is classified as ``TRENDING_DOWN`` with
    confidence ≥ 0.6. No-op when no detector is wired or the plan has
    no buys. Symbol-source-agnostic: works for crypto pairs (keyed off
    ``klines_by_symbol``) and stock tickers (keyed off ``bars``).
    """

    name = "apply_regime_gate"

    def __init__(self, detector: Any | None) -> None:
        self._detector = detector

    async def run(self, state: CycleState) -> CycleState:
        from halal_trader.crypto.regime import MarketRegime

        plan = state.plan
        if self._detector is None or plan is None or not getattr(plan, "buys", None):
            return state
        downtrend_pairs: set[str] = set()
        for pair, indicators in state.indicators_cache.items():
            if not indicators or "error" in indicators:
                continue
            try:
                regime, confidence, _ = self._detector.detect(indicators)
            except Exception:  # noqa: BLE001
                continue
            if regime == MarketRegime.TRENDING_DOWN and confidence >= 0.6:
                downtrend_pairs.add(pair)
        if not downtrend_pairs:
            return state
        blocked = [d for d in plan.buys if d.symbol in downtrend_pairs]
        if not blocked:
            return state
        for d in blocked:
            plan.decisions.remove(d)
        logger.warning(
            "Regime gate blocked %d BUY(s) in downtrend: %s",
            len(blocked),
            ", ".join(d.symbol for d in blocked),
        )
        return state


# ── Augment-microstructure-with-whale-flows stage (crypto) ───────


class AugmentMicrostructureWithWhaleFlowsStage:
    """Append on-chain whale-flow signals to ``state.microstructure_text``.

    The watched ERC-20s (USDT / USDC / DAI / WETH) are universe-wide —
    their flows apply to every pair regardless of which symbols the
    bot is trading. Mutates ``hub.whale_flows`` as a side effect so
    the dashboard can render the latest signal table.
    """

    name = "augment_microstructure_with_whale_flows"

    def __init__(
        self,
        whale_flow_source: Any | None,
        hub: Any | None = None,
    ) -> None:
        self._source = whale_flow_source
        self._hub = hub

    async def run(self, state: CycleState) -> CycleState:
        if self._source is None:
            return state
        try:
            from halal_trader.crypto.onchain import (
                TOKENS,
                format_whale_flows_for_prompt,
            )

            prices: dict[str, float] = {}
            eth_klines = state.klines_by_symbol.get("ETHUSDT") or []
            if eth_klines:
                prices["WETH"] = float(eth_klines[-1].close)

            signals = await self._source.fetch(list(TOKENS.keys()), prices=prices)
            if self._hub is not None:
                self._hub.whale_flows = signals
            block = format_whale_flows_for_prompt(signals)
            if block:
                state.microstructure_text = (
                    state.microstructure_text + "\n\n" + block
                    if state.microstructure_text
                    else block
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("whale-flow augmentation failed: %s", exc)
        return state


# ── Augment-microstructure-with-basis stage (crypto) ─────────────


class AugmentMicrostructureWithBasisStage:
    """Append spot-perp basis features to ``state.microstructure_text``.

    Reads funding signals from the broker (``get_funding_signal``) and
    feeds spot+perp+funding into the hub's :class:`BasisTracker.observe`
    helper, which produces a per-pair feature dict the formatter
    renders. No-op when the broker doesn't expose the helper.
    """

    name = "augment_microstructure_with_basis"

    def __init__(self, broker: Any | None, basis_tracker: Any | None) -> None:
        self._broker = broker
        self._basis = basis_tracker

    async def run(self, state: CycleState) -> CycleState:
        if (
            self._broker is None
            or self._basis is None
            or not state.halal_pairs
            or not hasattr(self._broker, "get_funding_signal")
        ):
            return state
        try:
            from halal_trader.crypto.basis import format_basis_for_prompt
        except Exception:  # noqa: BLE001
            return state

        features: dict[str, Any] = {}
        for pair in state.halal_pairs:
            try:
                sig = await self._broker.get_funding_signal(pair)
            except Exception:  # noqa: BLE001
                continue
            if not sig:
                continue
            try:
                spot_klines = state.klines_by_symbol.get(pair) or []
                spot_price = (
                    spot_klines[-1].close
                    if spot_klines
                    else self._broker.get_cached_price(pair)
                )
                if not spot_price:
                    continue
                features[pair] = self._basis.observe(
                    pair=pair,
                    spot_price=float(spot_price),
                    perp_price=float(sig.get("mark_price", spot_price)),
                    funding_rate_pct=float(sig.get("funding_rate", 0.0)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("basis observe failed for %s: %s", pair, exc)

        if features:
            basis_text = format_basis_for_prompt(features)
            if basis_text:
                state.microstructure_text = (
                    state.microstructure_text + "\n\n" + basis_text
                    if state.microstructure_text
                    else basis_text
                )
        return state


# ── Augment-regime-with-rag stage (crypto) ───────────────────────


class AugmentRegimeWithRagStage:
    """Append top-K analogous past-trade rationales to ``state.regime_text``.

    The query text is built from the current indicator dict + sentiment +
    regime text via :func:`core.llm.rag.build_rag_query` and fed into
    :class:`DBRationaleStore.query` for cosine search over the
    pgvector-backed RAG index. No-op when no store is wired or it's
    empty.
    """

    name = "augment_regime_with_rag"

    def __init__(self, rag_store: Any | None) -> None:
        self._rag_store = rag_store

    async def run(self, state: CycleState) -> CycleState:
        if self._rag_store is None:
            return state
        try:
            from halal_trader.core.llm.rag import (
                build_rag_query,
                format_rag_for_prompt,
            )

            if await self._rag_store.size() <= 0:
                return state
            query = build_rag_query(
                indicators_cache=state.indicators_cache,
                sentiment_text=state.sentiment_text,
                regime_text=state.regime_text,
            )
            hits = await self._rag_store.query(query, k=5, min_similarity=0.0)
            rag_text = format_rag_for_prompt(hits)
            if rag_text:
                state.regime_text = (
                    state.regime_text + "\n\n" + rag_text
                    if state.regime_text
                    else rag_text
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("RAG query failed: %s", exc)
        return state


# ── Augment-regime-with-memory stage (crypto) ────────────────────


class AugmentRegimeWithMemoryStage:
    """Append the top-K analogous past regimes to ``state.regime_text``.

    Reads the indicator cache + ``today_pnl`` + ``account.total_balance_usdt``
    from the state, builds a daily ``RegimeFeatures`` via
    :func:`ml.regime_memory.build_regime_features`, queries the memory
    for analogues, and appends a "Past analogous regimes" block to the
    existing regime text. No-op when no memory is wired or features
    can't be built.
    """

    name = "augment_regime_with_memory"

    def __init__(self, regime_memory: Any | None) -> None:
        self._regime_memory = regime_memory

    async def run(self, state: CycleState) -> CycleState:
        if self._regime_memory is None:
            return state
        try:
            from halal_trader.ml.regime_memory import (
                build_regime_features,
                format_for_prompt,
            )

            equity = (
                getattr(state.account, "total_balance_usdt", None)
                or getattr(state.account, "effective_equity", None)
                or 0.0
            )
            features = build_regime_features(
                indicators_cache=state.indicators_cache,
                today_pnl=state.today_pnl,
                equity=float(equity),
            )
            if features is None:
                return state
            if await self._regime_memory.size() <= 0:
                return state
            hits = await self._regime_memory.query(features, k=3)
            analog_text = format_for_prompt(features, hits)
            if analog_text and "No analogous" not in analog_text:
                state.regime_text = (
                    state.regime_text + "\n\n" + analog_text
                    if state.regime_text
                    else analog_text
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("regime memory query failed: %s", exc)
        return state


# ── Build-news stage (crypto) ────────────────────────────────────


class BuildNewsStage:
    """Pull a bounded snapshot from a ``RecentNewsFeed`` → ``state.news_text``."""

    name = "build_news_text"

    def __init__(self, news_feed: Any | None) -> None:
        self._news_feed = news_feed

    async def run(self, state: CycleState) -> CycleState:
        if self._news_feed is None:
            state.news_text = ""
            return state
        try:
            from halal_trader.sentiment.feed import format_news_for_prompt

            events = self._news_feed.snapshot()
            state.news_text = format_news_for_prompt(events, pair_filter=state.halal_pairs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("News feed unavailable: %s", exc)
            state.news_text = ""
        return state
