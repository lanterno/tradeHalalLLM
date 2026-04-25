"""Crypto trading strategy — LLM prompt engineering for 1-minute scalping."""

from __future__ import annotations

import logging
import time
from typing import Any

from halal_trader.core.strategy import BaseStrategy
from halal_trader.crypto.indicators import compute_all, format_indicators_for_prompt
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoTradingPlan,
    Kline,
)
from halal_trader.domain.ports import LLMBackend, TradeRepository

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert crypto scalping AI. Your job is to analyze technical indicators \
and real-time market data for cryptocurrency pairs, making precise buy/sell decisions \
on a 1-minute timeframe to achieve at least {daily_return_target:.0%} daily return.

RULES:
1. You ONLY trade pairs from the provided halal-compliant list.
2. You make short-term momentum/scalping trades — hold times range from 1 to 60 minutes.
3. Each trade must have a clear reasoning based on the technical indicators provided.
4. CRITICAL SIZING RULE: each trade's (quantity × current_price) MUST be STRICTLY LESS \
than the "Max Position Size" dollar value shown in the portfolio status. Use at most 90% of \
that limit to leave room for price movement. Check the "Available" balance too — you cannot \
spend more USDT than what is available.
4b. CRITICAL QUANTITY RULE: your quantity MUST comply with the EXCHANGE TRADING RULES \
section. The quantity must be a multiple of the step size and >= min_qty. The order's \
notional value (quantity × price) must be >= min_notional.
5. Current daily loss limit is {daily_loss_limit:.0%} — if losses approach this, be conservative.
6. Target daily return: {daily_return_target:.0%}.
7. Maximum simultaneous open positions: {max_positions}.
8. Trading fees are ~0.1% per trade (0.2% round trip) — factor this into your decisions.

STRATEGY GUIDELINES:
- Use RSI for overbought/oversold signals (buy below 40, sell above 65).
- Use MACD crossovers for momentum confirmation — act early on emerging crossovers.
- Bollinger Band squeezes signal potential breakouts — enter before the breakout confirms.
- EMA crossovers (9/21) indicate short-term trend changes — act on the crossover, not after.
- Volume ratios >1.2x average are sufficient to confirm moves.
- VWAP acts as intraday support/resistance.
- Order book imbalance indicates short-term pressure direction.
- Use stop-losses of {stop_loss_pct:.1%} below entry for longs.
- Take profits at {take_profit_pct:.1%} above entry (accounting for 0.2% fees).
- You SHOULD be making trades most cycles — look for opportunities, not reasons to hold.
- If 2+ indicators align even moderately, take the trade with appropriate sizing.
- MINIMUM POSITION SIZING: each trade's notional value (quantity × price) should be at \
least 5% of the Max Position Size shown. Never trade amounts below $50. \
For high-confidence setups (confidence >= 0.7), use 50-90% of Max Position Size. \
For moderate setups, use 20-50%. Calculate: quantity = (dollar_amount / current_price).
- Scale into positions: start with a partial position and add on confirmation.
- Review your recent performance stats: avoid pairs with consistently negative P&L, \
and adjust aggression based on your current win rate and streak.

SENTIMENT & ALTERNATIVE DATA (our competitive edge):
- High buzz + positive sentiment = act fast. If Reddit is suddenly talking about a coin, \
there's often a 15-60 minute window to profit before price fully adjusts.
- A buzz score >= 3.0 is a strong signal — combine with technical confirmation \
for high-confidence trades.
- Negative sentiment diverging from bullish technicals = caution, reduce position size.
- News headlines from CryptoPanic provide context — weight them alongside indicators.

MULTI-TIMEFRAME CONTEXT:
- Only take 1m signals that align with the 15m+ trend direction.
- The trend alignment score ranges from -1 (all bearish) to +1 (all bullish).
- Avoid counter-trend trades when alignment is below +0.3 (for buys) or above -0.3 (for sells).
- Higher-timeframe support/resistance levels are key — respect them.

ML MODEL SIGNALS:
- When ML price forecasts are available, use them as confirmation, not primary signals.
- ML confidence scores reflect patterns learned from our own trade history.
- Anomaly alerts mean unusual market microstructure — proceed with caution \
or exploit the opportunity.

PORTFOLIO RISK:
- When correlation is HIGH (>0.7), reduce position sizes — all positions will move together.
- Respect the risk-adjusted position limits shown in the PORTFOLIO RISK section.
- If portfolio heat is negative and significant, be conservative with new entries.
- If drawdown is approaching the limit, only take high-confidence trades or close losing positions.

MARKET REGIME AWARENESS:
- In TRENDING markets: trade with the trend, wider TP, tighter SL on counter-trend side.
- In RANGING markets: mean-reversion strategy, buy at BB lower, sell at BB upper.
- In HIGH VOLATILITY: reduce position sizes by 50%, widen SL, require strong confirmation.
- In DOWNTREND: only sell/hold, no new buys unless strong reversal signals.

{active_adjustments}

You MUST respond with valid JSON matching this exact schema:
{{
  "decisions": [
    {{
      "action": "buy" | "sell" | "hold",
      "symbol": "PAIR",
      "quantity": <float>,
      "confidence": <float 0-1>,
      "reasoning": "<brief explanation referencing specific indicators>",
      "entry_price": <float or null>,
      "target_price": <float or null>,
      "stop_loss": <float or null>
    }}
  ],
  "market_outlook": "<1-2 sentence crypto market assessment>",
  "risk_notes": "<any risk concerns>"
}}

Only return an empty decisions list if ALL indicators across ALL pairs are genuinely flat \
with no directional signal at all. Otherwise, find the best available setup and trade it.
"""

USER_PROMPT_TEMPLATE = """\
=== PORTFOLIO STATUS ===
Total Balance: ${total_balance:,.2f} USDT
Available: ${available_balance:,.2f} USDT
In Orders: ${in_order:,.2f} USDT
Max Position Size: ${max_position_value:,.2f} USDT ({max_position_pct:.0%} of portfolio)
Today's P&L: ${today_pnl:+,.2f} ({today_pnl_pct:+.2%})
Open Positions: {open_position_count}/{max_positions}
{position_limit_warning}
=== CURRENT POSITIONS ===
{positions_text}

=== HALAL-COMPLIANT PAIRS ===
{halal_pairs}

=== TECHNICAL INDICATORS (1-minute candles) ===
{indicators_text}

=== EXCHANGE TRADING RULES ===
{exchange_rules_text}

=== ORDER BOOK SUMMARY ===
{orderbook_text}

=== SOCIAL SENTIMENT ===
{sentiment_text}

=== MULTI-TIMEFRAME ANALYSIS ===
{timeframe_text}

=== ML MODEL SIGNALS ===
{ml_signals_text}

=== MARKET REGIME ===
{regime_text}

=== PORTFOLIO RISK ===
{risk_text}

=== YOUR RECENT PERFORMANCE (last 7 days) ===
{performance_text}

Based on these indicators, sentiment, ML signals, and your track record, what trades should I \
make right now? Remember: optimize for {daily_return_target:.0%}+ daily return — being in cash \
earns nothing. Account for 0.2% round-trip fees but don't let fees prevent you from acting on \
good setups. Bias toward action: find the best opportunity available and size it appropriately. \
Use sentiment and ML signals as your edge — big players don't have this data.
"""


class CryptoTradingStrategy(BaseStrategy):
    """Crypto scalping strategy with LLM circuit breaker."""

    def __init__(
        self,
        llm: LLMBackend,
        repo: TradeRepository,
        *,
        llm_provider_name: str,
        max_position_pct: float,
        daily_loss_limit: float,
        daily_return_target: float,
        max_simultaneous_positions: int,
        llm_failure_threshold: int = 5,
        llm_cooldown_seconds: int = 600,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
    ) -> None:
        super().__init__(
            llm,
            repo,
            llm_provider_name=llm_provider_name,
            max_position_pct=max_position_pct,
            daily_loss_limit=daily_loss_limit,
            daily_return_target=daily_return_target,
            max_simultaneous_positions=max_simultaneous_positions,
        )
        self._consecutive_llm_failures = 0
        self._llm_cooldown_until: float = 0
        self._llm_failure_threshold = llm_failure_threshold
        self._llm_cooldown_seconds = llm_cooldown_seconds
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct

    def _on_llm_success(self) -> None:
        self._consecutive_llm_failures = 0

    def _on_llm_failure(self, error: Exception, elapsed_ms: int, prefix: str) -> None:
        self._consecutive_llm_failures += 1
        logger.error(
            "Crypto LLM analysis failed after %dms (%d consecutive): %s",
            elapsed_ms,
            self._consecutive_llm_failures,
            error,
        )
        if self._consecutive_llm_failures >= self._llm_failure_threshold:
            self._llm_cooldown_until = time.monotonic() + self._llm_cooldown_seconds
            logger.warning(
                "LLM failed %d times consecutively — entering %ds cooldown",
                self._consecutive_llm_failures,
                self._llm_cooldown_seconds,
            )

    async def analyze(
        self,
        account: CryptoAccount,
        positions_text: str,
        halal_pairs: list[str],
        klines_by_symbol: dict[str, list[Kline]],
        orderbooks: dict[str, dict[str, Any]],
        today_pnl: float = 0.0,
        performance_text: str = "",
        sentiment_text: str = "",
        timeframe_text: str = "",
        ml_signals_text: str = "",
        regime_text: str = "",
        active_adjustments: str = "",
        exchange_rules_text: str = "",
        indicators_cache: dict[str, dict] | None = None,
        open_position_count: int = 0,
        risk_text: str = "",
    ) -> CryptoTradingPlan:
        now = time.monotonic()
        if now < self._llm_cooldown_until:
            remaining = int(self._llm_cooldown_until - now)
            logger.warning("LLM in cooldown (%ds remaining) — holding positions", remaining)
            return CryptoTradingPlan(
                market_outlook="LLM cooldown active — holding",
                risk_notes=(
                    f"Cooldown for {remaining}s after "
                    f"{self._consecutive_llm_failures} consecutive failures"
                ),
            )

        portfolio_value = account.total_balance_usdt
        if not portfolio_value:
            logger.warning("Portfolio value is zero/None — falling back to $1000 for sizing")
            portfolio_value = 1000
        today_pnl_pct = today_pnl / portfolio_value if portfolio_value else 0

        indicators_text = self._build_indicators_text(klines_by_symbol, indicators_cache)
        orderbook_text = self._build_orderbook_text(orderbooks)

        adjustments_block = ""
        if active_adjustments:
            adjustments_block = (
                "ACTIVE STRATEGY ADJUSTMENTS (from your own performance review):\n"
                + active_adjustments
            )

        at_max = open_position_count >= self._max_simultaneous_positions

        system = SYSTEM_PROMPT.format(
            max_position_pct=self._max_position_pct,
            daily_loss_limit=self._daily_loss_limit,
            daily_return_target=self._daily_return_target,
            max_positions=self._max_simultaneous_positions,
            active_adjustments=adjustments_block,
            stop_loss_pct=self._stop_loss_pct,
            take_profit_pct=self._take_profit_pct,
        )

        if at_max:
            system += (
                "\n\n*** SELL-ONLY MODE ***\n"
                "You are currently at the MAXIMUM number of open positions. "
                "You CANNOT buy anything. Any buy decisions will be rejected.\n"
                "Focus ONLY on:\n"
                "1. Selling your weakest position(s) to free up capital and slots.\n"
                "2. Holding strong positions.\n"
                "Do NOT include any buy decisions in your response."
            )

        pct_limit = portfolio_value * self._max_position_pct
        spendable = account.usdt_free if account.usdt_free > 0 else account.available_balance_usdt
        max_position_value = min(pct_limit, spendable)
        position_limit_warning = ""
        if at_max:
            position_limit_warning = (
                "⚠ POSITION LIMIT REACHED — you MUST fully close (sell ALL quantity of) "
                "an existing position before any new buys will be accepted. A partial "
                "sell does NOT free the slot. Sell the ENTIRE holding of your weakest "
                "position to open a slot for a better opportunity."
            )
        elif open_position_count >= self._max_simultaneous_positions - 1:
            position_limit_warning = "⚠ Only 1 position slot remaining — be selective with buys."

        optional_sections: list[tuple[str, str, str]] = [
            ("=== SOCIAL SENTIMENT ===", sentiment_text, "No sentiment data available."),
            (
                "=== MULTI-TIMEFRAME ANALYSIS ===",
                timeframe_text,
                "No multi-timeframe data available.",
            ),
            ("=== ML MODEL SIGNALS ===", ml_signals_text, "No ML model data available."),
            ("=== MARKET REGIME ===", regime_text, "No regime data available."),
            ("=== PORTFOLIO RISK ===", risk_text, "No portfolio risk data available."),
        ]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            total_balance=account.total_balance_usdt,
            available_balance=account.available_balance_usdt,
            in_order=account.in_order_usdt,
            max_position_value=max_position_value,
            max_position_pct=self._max_position_pct,
            today_pnl=today_pnl,
            today_pnl_pct=today_pnl_pct,
            open_position_count=open_position_count,
            max_positions=self._max_simultaneous_positions,
            position_limit_warning=position_limit_warning,
            positions_text=positions_text or "No open positions.",
            halal_pairs=", ".join(halal_pairs),
            indicators_text=indicators_text,
            exchange_rules_text=exchange_rules_text or "No exchange trading rules available.",
            orderbook_text=orderbook_text,
            sentiment_text=sentiment_text or "No sentiment data available.",
            timeframe_text=timeframe_text or "No multi-timeframe data available.",
            ml_signals_text=ml_signals_text or "No ML model data available.",
            regime_text=regime_text or "No regime data available.",
            risk_text=risk_text or "No portfolio risk data available.",
            performance_text=performance_text or "No completed trades yet.",
            daily_return_target=self._daily_return_target,
        )

        for header, value, placeholder in optional_sections:
            if not value:
                user_prompt = user_prompt.replace(f"{header}\n{placeholder}\n\n", "")

        return await self._run_llm_analysis(
            system,
            user_prompt,
            prompt_summary=(
                f"Crypto: analyzed {len(halal_pairs)} halal pairs, "
                f"balance=${account.total_balance_usdt:.2f}"
            ),
            validate=lambda raw: CryptoTradingPlan.model_validate(raw),
            make_empty=lambda msg: CryptoTradingPlan(
                market_outlook="Analysis failed — holding positions",
                risk_notes=msg,
            ),
            extract_symbols=lambda p: [d.symbol for d in p.decisions],
            count_actions=lambda p: {
                "buys": len(p.buys),
                "sells": len(p.sells),
                "holds": len(p.holds),
            },
            log_prefix="Crypto",
        )

    def _build_indicators_text(
        self,
        klines_by_symbol: dict[str, list[Kline]],
        indicators_cache: dict[str, dict] | None = None,
    ) -> str:
        if not klines_by_symbol:
            return "No indicator data available."
        lines = []
        for symbol, klines in klines_by_symbol.items():
            if indicators_cache and symbol in indicators_cache:
                indicators = indicators_cache[symbol]
            else:
                indicators = compute_all(klines)
            lines.append(format_indicators_for_prompt(symbol, indicators))
        return "\n".join(lines)

    def _build_orderbook_text(self, orderbooks: dict[str, dict[str, Any]]) -> str:
        if not orderbooks:
            return "No order book data available."
        lines = []
        for symbol, book in orderbooks.items():
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            bid_vol = sum(q for _, q in bids[:5]) if bids else 0
            ask_vol = sum(q for _, q in asks[:5]) if asks else 0
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = (bid_vol - ask_vol) / total
                direction = (
                    "BUY pressure"
                    if imbalance > 0.1
                    else ("SELL pressure" if imbalance < -0.1 else "NEUTRAL")
                )
            else:
                imbalance = 0
                direction = "N/A"
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if best_ask and best_bid else 0
            spread_pct = (spread / best_bid * 100) if best_bid else 0
            lines.append(
                f"  {symbol}: Bid={best_bid:.2f}, Ask={best_ask:.2f}, "
                f"Spread={spread_pct:.4f}%, "
                f"Imbalance={imbalance:+.2f} ({direction})"
            )
        return "\n".join(lines)
