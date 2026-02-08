"""Crypto trading strategy — LLM prompt engineering for 1-minute scalping."""

import json
import logging
import time
from typing import Any

from halal_trader.crypto.indicators import compute_all, format_indicators_for_prompt
from halal_trader.domain.models import (
    CryptoAccount,
    CryptoTradingPlan,
    Kline,
)
from halal_trader.domain.ports import LLMProvider, TradeRepository

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert crypto scalping AI. Your job is to analyze technical indicators \
and real-time market data for cryptocurrency pairs, making precise buy/sell decisions \
on a 1-minute timeframe to achieve at least {daily_return_target:.0%} daily return.

RULES:
1. You ONLY trade pairs from the provided halal-compliant list.
2. You make short-term momentum/scalping trades — hold times range from 1 to 60 minutes.
3. Each trade must have a clear reasoning based on the technical indicators provided.
4. Risk management: no single position should exceed {max_position_pct:.0%} of the portfolio.
5. Current daily loss limit is {daily_loss_limit:.0%} — if losses approach this, be conservative.
6. Target daily return: {daily_return_target:.0%}.
7. Maximum simultaneous open positions: {max_positions}.
8. Trading fees are ~0.1% per trade (0.2% round trip) — factor this into your decisions.

STRATEGY GUIDELINES:
- Use RSI for overbought/oversold signals (buy below 30, sell above 70).
- Use MACD crossovers for momentum confirmation.
- Bollinger Band squeezes signal potential breakouts.
- EMA crossovers (9/21) indicate short-term trend changes.
- High volume ratios (>1.5x average) confirm moves.
- VWAP acts as intraday support/resistance.
- Order book imbalance indicates short-term pressure direction.
- Set tight stop-losses (0.3-0.5% below entry for longs).
- Take profits at 0.5-1.0% above entry (accounting for 0.2% fees).
- If indicators are mixed or unclear, HOLD and wait for a clearer setup.
- Crypto markets are 24/7 — there is no rush, wait for high-probability setups.

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

If there are no good setups, return an empty decisions list with your market outlook.
"""

USER_PROMPT_TEMPLATE = """\
=== PORTFOLIO STATUS ===
Total Balance: ${total_balance:,.2f} USDT
Available: ${available_balance:,.2f} USDT
In Orders: ${in_order:,.2f} USDT
Today's P&L: ${today_pnl:+,.2f} ({today_pnl_pct:+.2%})

=== CURRENT POSITIONS ===
{positions_text}

=== HALAL-COMPLIANT PAIRS ===
{halal_pairs}

=== TECHNICAL INDICATORS (1-minute candles) ===
{indicators_text}

=== ORDER BOOK SUMMARY ===
{orderbook_text}

Based on these indicators, what trades should I make right now? \
Remember: optimize for {daily_return_target:.0%}+ daily return with tight risk management. \
Account for 0.2% round-trip fees in your calculations.
"""


class CryptoTradingStrategy:
    """Orchestrates the LLM to produce a CryptoTradingPlan from market data."""

    def __init__(
        self,
        llm: LLMProvider,
        repo: TradeRepository,
        *,
        llm_provider_name: str,
        max_position_pct: float,
        daily_loss_limit: float,
        daily_return_target: float,
        max_simultaneous_positions: int,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._llm_provider_name = llm_provider_name
        self._max_position_pct = max_position_pct
        self._daily_loss_limit = daily_loss_limit
        self._daily_return_target = daily_return_target
        self._max_simultaneous_positions = max_simultaneous_positions

    async def analyze(
        self,
        account: CryptoAccount,
        positions_text: str,
        halal_pairs: list[str],
        klines_by_symbol: dict[str, list[Kline]],
        orderbooks: dict[str, dict[str, Any]],
        today_pnl: float = 0.0,
    ) -> CryptoTradingPlan:
        """Run the LLM analysis and return a structured CryptoTradingPlan."""
        portfolio_value = account.total_balance_usdt or 1000
        today_pnl_pct = today_pnl / portfolio_value if portfolio_value else 0

        # Pre-compute all technical indicators
        indicators_text = self._build_indicators_text(klines_by_symbol)
        orderbook_text = self._build_orderbook_text(orderbooks)

        system = SYSTEM_PROMPT.format(
            max_position_pct=self._max_position_pct,
            daily_loss_limit=self._daily_loss_limit,
            daily_return_target=self._daily_return_target,
            max_positions=self._max_simultaneous_positions,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            total_balance=account.total_balance_usdt,
            available_balance=account.available_balance_usdt,
            in_order=account.in_order_usdt,
            today_pnl=today_pnl,
            today_pnl_pct=today_pnl_pct,
            positions_text=positions_text or "No open positions.",
            halal_pairs=", ".join(halal_pairs),
            indicators_text=indicators_text,
            orderbook_text=orderbook_text,
            daily_return_target=self._daily_return_target,
        )

        t0 = time.monotonic()
        try:
            raw = await self._llm.generate_json(user_prompt, system=system)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            plan = CryptoTradingPlan.model_validate(raw)

            # Audit trail
            await self._repo.record_decision(
                provider=self._llm_provider_name,
                model=self._llm.model,
                prompt_summary=(
                    f"Crypto: analyzed {len(halal_pairs)} halal pairs, "
                    f"balance=${account.total_balance_usdt:.2f}"
                ),
                raw_response=json.dumps(raw),
                parsed_action={
                    "buys": len(plan.buys),
                    "sells": len(plan.sells),
                    "holds": len(plan.holds),
                },
                symbols=[d.symbol for d in plan.decisions],
                execution_ms=elapsed_ms,
            )

            logger.info(
                "Crypto LLM analysis complete in %dms: %d buys, %d sells, %d holds",
                elapsed_ms,
                len(plan.buys),
                len(plan.sells),
                len(plan.holds),
            )
            return plan

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.error("Crypto LLM analysis failed after %dms: %s", elapsed_ms, e)
            await self._repo.record_decision(
                provider=self._llm_provider_name,
                model=self._llm.model,
                prompt_summary="FAILED crypto analysis",
                raw_response=str(e),
                execution_ms=elapsed_ms,
            )
            return CryptoTradingPlan(
                market_outlook="Analysis failed — holding positions",
                risk_notes=str(e),
            )

    # ── Private helpers ────────────────────────────────────────

    def _build_indicators_text(self, klines_by_symbol: dict[str, list[Kline]]) -> str:
        """Compute technical indicators for each symbol and format for the prompt."""
        if not klines_by_symbol:
            return "No indicator data available."

        lines = []
        for symbol, klines in klines_by_symbol.items():
            indicators = compute_all(klines)
            lines.append(format_indicators_for_prompt(symbol, indicators))
        return "\n".join(lines)

    def _build_orderbook_text(self, orderbooks: dict[str, dict[str, Any]]) -> str:
        """Format order book data for the prompt."""
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
