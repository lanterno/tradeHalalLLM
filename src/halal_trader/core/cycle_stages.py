"""Cycle stage classes — each owns one prompt-context block.

A ``CycleStage`` takes a :class:`CycleState`, mutates one or two
fields, returns the state. The cycle service holds an ordered list of
stages and runs them. New prompt-context sources land as one new file
under this module + one new line on the cycle's stage list.

The stock cycle drives these through
:func:`core.cycle_pipeline.run_stages`; each stage stamps one
prompt-context block the strategy LLM sees. Stage exceptions are
swallowed by the ``run_stages`` driver so a regional failure can't
abort the cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from halal_trader.core.cycle_pipeline import CycleState

logger = logging.getLogger(__name__)


class CycleStage(Protocol):
    """Protocol every stage class satisfies."""

    name: str

    async def run(self, state: CycleState) -> CycleState: ...


# ── Build-timeframe stage ────────────────────────────────────────


class BuildTimeframeStage:
    """Run the multi-timeframe analyzer and stamp ``state.timeframe_text``."""

    name = "build_timeframe_text"

    def __init__(self, analyzer: Any | None) -> None:
        self._analyzer = analyzer

    async def run(self, state: CycleState) -> CycleState:
        from halal_trader.signals.timeframes import build_timeframe_text

        state.timeframe_text = await build_timeframe_text(self._analyzer, state.halal_pairs)
        return state


# ── Build-performance stage ──────────────────────────────────────


class BuildPerformanceStage:
    """Run the rolling performance summary and stamp ``state.performance_text``.

    Reads from any analytics impl that exposes
    ``compute_stats(lookback_days=...)`` and ``format_for_prompt(stats)`` —
    :class:`portfolio.analytics.PerformanceAnalytics` satisfies the shape.
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

    The stock self-review (``StockTradeSelfReview``) exposes
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


# ── Build-stock-risk stage ───────────────────────────────────────


class BuildStockRiskStage:
    """Stocks-side: run the shared portfolio-risk engine over Alpaca bars.

    Populates four fields:

    * ``state.risk_text`` — the prompt block.
    * ``state.indicators_cache`` — the per-symbol indicator dict.
    * ``state.risk_state`` — the structured ``PortfolioRiskState`` so
      the dashboard's risk panel can render heat / drawdown / correlation.
    * ``state.halt`` — mirrors ``state.risk_state.is_halted`` so the
      stocks cycle can short-circuit on a heat/drawdown breach.

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


# ── Fetch-stock-news stage ──────────────────────────────────────


class FetchStockNewsStage:
    """Fetch equities news per cycle → ``state.news_text``.

    Pulls on demand from the configured collector (Finnhub, or Yahoo
    Finance as the fallback); the 15-min cadence is slow enough that
    per-cycle pulls are cheap. When ``news_collector`` is None the stage
    is a no-op so the cycle still runs without network access.
    """

    name = "fetch_stock_news_text"

    def __init__(self, news_collector: Any | None) -> None:
        self._news_collector = news_collector

    async def run(self, state: CycleState) -> CycleState:
        if self._news_collector is None or not state.halal_pairs:
            state.news_text = ""
            return state
        try:
            from halal_trader.sentiment.feed import format_news_for_prompt

            events = await self._news_collector.fetch_for_symbols(list(state.halal_pairs))
            state.news_text = format_news_for_prompt(events, symbol_filter=state.halal_pairs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stock news fetch failed: %s", exc)
            state.news_text = ""
        return state
