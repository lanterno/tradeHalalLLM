"""Crypto component registry — builds the full live-trading wiring once.

The scheduler used to do this inline in a 215-line ``_create_components``
method with deeply nested conditionals for the ML / sentiment / news
branches. This module centralises the wiring; the scheduler just calls
:func:`build_components` and fans the result out into its own attributes.

Optional subsystems (sentiment, ML, news reactor) live behind small
``_build_*`` helpers; each returns ``None`` when not configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import Settings
from halal_trader.core.llm import create_llm
from halal_trader.core.llm.base import BaseLLM
from halal_trader.core.safeguards import LiveModeChecker, check_live_mode_token
from halal_trader.crypto.analytics import PerformanceAnalytics
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.crypto.executor import CryptoExecutor
from halal_trader.crypto.monitor import PositionMonitor
from halal_trader.crypto.portfolio import CryptoPortfolioTracker
from halal_trader.crypto.regime import RegimeDetector
from halal_trader.crypto.risk import PortfolioRiskEngine
from halal_trader.crypto.screener import CryptoHalalScreener
from halal_trader.crypto.self_improve import TradeSelfReview
from halal_trader.crypto.strategy import CryptoTradingStrategy
from halal_trader.crypto.timeframes import TimeframeAnalyzer
from halal_trader.crypto.websocket import BinanceWSManager
from halal_trader.db.repository import Repository
from halal_trader.ml.retrainer import RetrainingScheduler
from halal_trader.notifications.telegram import AlertSink, TelegramNotifier
from halal_trader.sentiment.events import NewsEvent, NewsEventReactor
from halal_trader.sentiment.feed import RecentNewsFeed

logger = logging.getLogger(__name__)


@dataclass
class CryptoComponents:
    """Every long-lived object the crypto cycle needs, in one bag."""

    # Live-mode safety
    live_mode_checker: LiveModeChecker

    # Brokers + market data
    binance: BinanceClient
    ws: BinanceWSManager

    # LLM
    llm: BaseLLM
    strategy: CryptoTradingStrategy

    # Order flow
    executor: CryptoExecutor
    monitor: PositionMonitor

    # Portfolio + analytics
    screener: CryptoHalalScreener
    portfolio: CryptoPortfolioTracker
    analytics: PerformanceAnalytics
    risk_engine: PortfolioRiskEngine
    self_review: TradeSelfReview
    retrainer: RetrainingScheduler

    # Notifications + ops
    notifier: TelegramNotifier
    alerts: AlertSink

    # Optional / conditionally-enabled
    sentiment_manager: Any = None
    timeframe_analyzer: Any = None
    regime_detector: RegimeDetector | None = None
    ml_forecaster: Any = None
    ml_anomaly: Any = None
    ml_signal: Any = None
    news_reactor: NewsEventReactor | None = None
    news_feed: RecentNewsFeed | None = None
    shadow_runner: Any = None
    whale_flow_source: Any = None
    reddit_fetcher: Any = None


# ── Optional subsystem builders ───────────────────────────────


def _build_sentiment(settings: Settings) -> Any:
    from halal_trader.sentiment.manager import SentimentManager

    return SentimentManager(
        trading_pairs=settings.crypto.pairs,
        reddit_client_id=settings.sentiment.reddit.client_id,
        reddit_client_secret=settings.sentiment.reddit.client_secret,
        cryptopanic_api_key=settings.sentiment.cryptopanic.api_key,
        use_finbert=settings.sentiment.use_finbert,
        update_interval_seconds=settings.sentiment.update_interval_seconds,
    )


def _build_ml(settings: Settings) -> tuple[Any, Any, Any]:
    """Return (forecaster, anomaly_detector, signal_classifier) or all-None."""
    if not settings.ml.enabled:
        return None, None, None
    try:
        from halal_trader.ml.anomaly import MarketAnomalyDetector, MLSignalClassifier
        from halal_trader.ml.forecaster import PriceForecaster
        from halal_trader.ml.hub import ModelHub

        hub = ModelHub(device=settings.ml.device, models_dir=settings.ml.models_dir)
        logger.info("ML models enabled (device: %s)", settings.ml.device)
        return PriceForecaster(hub), MarketAnomalyDetector(hub), MLSignalClassifier(hub)
    except Exception as e:
        logger.warning("ML models initialization failed: %s", e)
        return None, None, None


def _build_news_reactor(settings: Settings) -> NewsEventReactor:
    return NewsEventReactor(
        api_key=settings.sentiment.cryptopanic.api_key,
        trading_pairs=settings.crypto.pairs,
        poll_interval_seconds=30,
        importance_filter="hot",
    )


# ── Main builder ──────────────────────────────────────────────


async def build_components(
    *,
    settings: Settings,
    repo: Repository,
    engine: AsyncEngine | None,
    binance: BinanceClient,
    exiting_pairs: set[str],
) -> CryptoComponents:
    """Wire every long-lived crypto-bot component once.

    The scheduler still owns lifecycle (start/stop of background tasks);
    this function just constructs the objects in dependency order.
    """
    check_live_mode_token(settings, market="crypto")
    live_mode_checker = LiveModeChecker(settings=settings, market="crypto")

    await binance.connect()

    ws = BinanceWSManager(binance.client, symbols=settings.crypto.pairs)
    await ws.start()

    llm = create_llm(settings)

    screener = CryptoHalalScreener(
        repo,
        coingecko_api_key=settings.coingecko.api_key,
        min_market_cap=settings.crypto.min_market_cap,
    )

    # Optional adversarial co-bot — same provider stack as primary by
    # default. Constructed only when the operator has flipped the flag,
    # so the cost is opt-in.
    attacker_llm: BaseLLM | None = None
    if getattr(settings.llm, "adversarial_enabled", False):
        try:
            attacker_llm = create_llm(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("adversarial LLM init failed: %s — disabling", exc)
            attacker_llm = None

    # Optional ensemble — N additional providers from the same factory.
    # Empty list = disabled.
    ensemble_llms: list[BaseLLM] = []
    for _ in range(int(getattr(settings.llm, "ensemble_size", 0) or 0)):
        try:
            ensemble_llms.append(create_llm(settings))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensemble LLM init failed: %s — variant skipped", exc)

    # Build the post-close analytics recorder bundle. The hub is
    # process-wide; sidecar paths anchor under the data dir.
    from halal_trader.core.insights_hub import hub as insights_hub
    from halal_trader.core.llm.rag_db import DBRationaleStore
    from halal_trader.core.post_close import CloseRecorders
    from halal_trader.core.regret_db import DBRegretRecorder
    from halal_trader.core.thesis_db import DBThesisTagStore
    from halal_trader.halal.round_trip_purification import (
        RoundTripLedger,
    )

    if engine is None:
        raise RuntimeError("CryptoComponents requires a live database engine")

    data_dir = settings.resolve_data_dir() / "analytics"
    data_dir.mkdir(parents=True, exist_ok=True)
    rag_store: Any = DBRationaleStore(engine=engine)
    thesis_store: Any = DBThesisTagStore(engine=engine)
    regret_store: Any = DBRegretRecorder(engine=engine)
    insights_hub.rag = rag_store

    # Optional Etherscan whale-flow source. The cycle records its
    # signal into insights_hub.whale_flows and the prompt builder
    # surfaces them in the microstructure block.
    whale_flow_source: Any = None
    if getattr(settings, "etherscan", None) and settings.etherscan.api_key:
        from halal_trader.crypto.onchain import EtherscanWhaleFlow

        whale_flow_source = EtherscanWhaleFlow(api_key=settings.etherscan.api_key)

    # Reddit mention-velocity source — uses public JSON, no OAuth.
    # Free; the Reddit API ToS just wants a unique User-Agent. We
    # always wire this since there's no cost or key to manage.
    from halal_trader.sentiment.reddit_public import (
        DEFAULT_CRYPTO_SUBS,
        RedditPublicFetcher,
    )

    reddit_fetcher = RedditPublicFetcher(
        user_agent="halal-trader/0.1 (crypto-velocity)",
        subreddits=DEFAULT_CRYPTO_SUBS,
    )

    close_recorders = CloseRecorders(
        hub=insights_hub,
        thesis_store=thesis_store,
        regret_recorder=regret_store,
        purification_ledger=RoundTripLedger(path=data_dir / "round_trip_purification.json"),
        purification_rules={},  # Operator wires per-symbol rules later
        rag_store=rag_store,
    )

    strategy = CryptoTradingStrategy(
        llm,
        repo,
        llm_provider_name=settings.llm.provider.value,
        max_position_pct=settings.crypto.max_position_pct,
        daily_loss_limit=settings.crypto.daily_loss_limit,
        daily_return_target=settings.crypto.daily_return_target,
        max_simultaneous_positions=settings.crypto.max_simultaneous_positions,
        llm_failure_threshold=settings.crypto.llm_failure_threshold,
        llm_cooldown_seconds=settings.crypto.llm_cooldown_seconds,
        attacker_llm=attacker_llm,
        ensemble_llms=ensemble_llms,
    )

    executor = CryptoExecutor(
        binance,
        repo,
        max_position_pct=settings.crypto.max_position_pct,
        max_simultaneous_positions=settings.crypto.max_simultaneous_positions,
        configured_pairs=settings.crypto.pairs,
        circuit_breaker_threshold=settings.crypto.circuit_breaker_threshold,
        circuit_breaker_window=settings.crypto.circuit_breaker_window,
        circuit_breaker_cooldown=settings.crypto.circuit_breaker_cooldown,
        exiting_pairs=exiting_pairs,
    )

    portfolio = CryptoPortfolioTracker(
        binance, repo, daily_loss_limit=settings.crypto.daily_loss_limit
    )
    analytics = PerformanceAnalytics(repo)

    notifier = TelegramNotifier(
        bot_token=settings.telegram.bot_token, chat_id=settings.telegram.chat_id
    )
    alerts = AlertSink(notifier)
    if notifier.enabled:
        logger.info("Telegram notifications enabled")

    sentiment_manager = _build_sentiment(settings)
    if sentiment_manager is not None:
        await sentiment_manager.start()

    timeframe_analyzer = TimeframeAnalyzer(binance)
    regime_detector = RegimeDetector(models_dir=settings.ml.models_dir)
    ml_forecaster, ml_anomaly, ml_signal = _build_ml(settings)

    self_review = TradeSelfReview(llm, repo, strategy=strategy)
    await self_review.load_from_db()

    risk_engine = PortfolioRiskEngine(
        base_max_position_pct=settings.crypto.max_position_pct,
        max_portfolio_heat_pct=settings.crypto.max_portfolio_heat_pct,
        max_drawdown_pct=settings.crypto.max_drawdown_pct,
        high_correlation_threshold=settings.crypto.high_correlation_threshold,
        correlation_reduction_factor=settings.crypto.correlation_reduction_factor,
        atr_baseline=settings.crypto.atr_baseline,
    )

    retrainer = RetrainingScheduler(repo, models_dir=settings.ml.models_dir)

    monitor = PositionMonitor(
        broker=binance,
        repo=repo,
        ws_manager=ws,
        check_interval=settings.crypto.monitor_interval,
        trailing_stop_activation_pct=settings.crypto.trailing_stop_activation_pct,
        trailing_stop_distance_pct=settings.crypto.trailing_stop_distance_pct,
        notifier=notifier if notifier.enabled else None,
        retrainer=retrainer,
        exiting_pairs=exiting_pairs,
        close_recorders=close_recorders,
    )

    # Optional shadow strategy — frozen-prompt variant that runs
    # alongside the live strategy and feeds the divergence ledger.
    shadow_runner = None
    if getattr(settings.llm, "shadow_enabled", False):
        try:
            from halal_trader.core.shadow_runner import (
                FrozenPromptStrategy,
                ShadowRunner,
            )
            from halal_trader.crypto.prompts import (
                PROMPT_VERSION as _CRYPTO_PV,
            )

            shadow_strategy = CryptoTradingStrategy(
                create_llm(settings),
                repo,
                llm_provider_name=settings.llm.provider.value,
                max_position_pct=settings.crypto.max_position_pct,
                daily_loss_limit=settings.crypto.daily_loss_limit,
                daily_return_target=settings.crypto.daily_return_target,
                max_simultaneous_positions=settings.crypto.max_simultaneous_positions,
            )
            frozen = FrozenPromptStrategy(
                inner=shadow_strategy,
                frozen_prompt_version=_CRYPTO_PV.short,
            )
            shadow_runner = ShadowRunner(
                shadow_strategy=frozen,
                ledger=insights_hub.shadow,
                starting_cash=settings.llm.shadow_starting_cash,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shadow runner init failed: %s — disabling", exc)
            shadow_runner = None

    news_reactor = _build_news_reactor(settings)

    # The cycle reads from this bounded buffer; the reactor's job is to
    # push every fired event in. We keep both as separate components so
    # the reactor can also trigger emergency mini-cycles independently
    # of the LLM-prompt feed.
    news_feed = RecentNewsFeed(capacity=10, max_age_seconds=1800)

    async def _push_news_to_feed(event: NewsEvent) -> None:
        news_feed.push(event)

    news_reactor.on_event(_push_news_to_feed)

    _ = engine  # kept in the signature so future components can read it.

    return CryptoComponents(
        live_mode_checker=live_mode_checker,
        binance=binance,
        ws=ws,
        llm=llm,
        strategy=strategy,
        executor=executor,
        monitor=monitor,
        screener=screener,
        portfolio=portfolio,
        analytics=analytics,
        risk_engine=risk_engine,
        self_review=self_review,
        retrainer=retrainer,
        notifier=notifier,
        alerts=alerts,
        sentiment_manager=sentiment_manager,
        timeframe_analyzer=timeframe_analyzer,
        regime_detector=regime_detector,
        ml_forecaster=ml_forecaster,
        ml_anomaly=ml_anomaly,
        ml_signal=ml_signal,
        news_reactor=news_reactor,
        news_feed=news_feed,
        shadow_runner=shadow_runner,
        whale_flow_source=whale_flow_source,
        reddit_fetcher=reddit_fetcher,
    )
