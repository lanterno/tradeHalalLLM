"""Day-trading strategy with prompt engineering for LLM decision-making."""

import json
import logging
import time
from typing import Any

from halal_trader.config import get_settings
from halal_trader.domain.models import TradingPlan
from halal_trader.domain.ports import LLMProvider, TradeRepository

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert intraday stock trader AI. Your job is to analyze market data \
and make precise buy/sell decisions to achieve at least 1% daily portfolio return.

RULES:
1. You ONLY trade stocks from the provided halal-compliant list.
2. You make ONLY intraday trades — all positions must be closeable by market close.
3. You optimize for high-probability short-term momentum trades.
4. Each trade must have a clear reasoning based on the data provided.
5. You manage risk: no single position should exceed {max_position_pct:.0%} of the portfolio.
6. Current daily loss limit is {daily_loss_limit:.0%} — if losses approach this, be conservative.
7. Target daily return: {daily_return_target:.0%}.
8. Maximum simultaneous open positions: {max_positions}.

STRATEGY GUIDELINES:
- Look for stocks with strong pre-market/intraday momentum.
- Consider volume spikes as entry signals.
- Use support/resistance from recent price bars.
- Prefer liquid, large-cap stocks for easier fills.
- Set mental stop-losses for every trade.
- If the market outlook is uncertain, it is OK to HOLD and not trade.

You MUST respond with valid JSON matching this exact schema:
{{
  "decisions": [
    {{
      "action": "buy" | "sell" | "hold",
      "symbol": "TICKER",
      "quantity": <integer>,
      "confidence": <float 0-1>,
      "reasoning": "<brief explanation>",
      "target_price": <float or null>,
      "stop_loss": <float or null>
    }}
  ],
  "market_outlook": "<1-2 sentence market assessment>",
  "risk_notes": "<any risk concerns>"
}}

If there are no good trades, return an empty decisions list with your market outlook.
"""

USER_PROMPT_TEMPLATE = """\
=== PORTFOLIO STATUS ===
Buying Power: ${buying_power:,.2f}
Portfolio Value: ${portfolio_value:,.2f}
Cash: ${cash:,.2f}
Today's P&L: ${today_pnl:+,.2f} ({today_pnl_pct:+.2%})

=== CURRENT POSITIONS ===
{positions_text}

=== HALAL-COMPLIANT STOCK UNIVERSE ===
{halal_symbols}

=== MARKET DATA (Snapshots) ===
{snapshots_text}

=== RECENT PRICE BARS (5-day daily) ===
{bars_text}

=== SENTIMENT ANALYSIS ===
{sentiment_text}

Based on this data, what trades should I make right now? \
Remember: optimize for 1%+ daily return with proper risk management.
"""


def _format_positions(positions: list[dict[str, Any]]) -> str:
    if not positions:
        return "No open positions."
    lines = []
    for p in positions:
        lines.append(
            f"  {p.get('symbol', '?')}: {p.get('qty', 0)} shares @ "
            f"${p.get('avg_entry_price', 0):.2f} | "
            f"Current: ${p.get('current_price', 0):.2f} | "
            f"P&L: ${p.get('unrealized_pl', 0):+.2f} ({p.get('unrealized_plpc', 0):+.2%})"
        )
    return "\n".join(lines)


def _format_snapshots(snapshots: dict[str, Any]) -> str:
    if not snapshots:
        return "No snapshot data available."
    lines = []
    if isinstance(snapshots, dict):
        for sym, data in snapshots.items():
            if isinstance(data, dict):
                price = data.get("latest_trade", {}).get("price", "N/A")
                bid = data.get("latest_quote", {}).get("bid_price", "N/A")
                ask = data.get("latest_quote", {}).get("ask_price", "N/A")
                vol = data.get("daily_bar", {}).get("volume", "N/A")
                lines.append(f"  {sym}: Price=${price} Bid=${bid} Ask=${ask} Vol={vol}")
            else:
                lines.append(f"  {sym}: {data}")
    else:
        lines.append(str(snapshots))
    return "\n".join(lines) if lines else str(snapshots)


def _format_bars(bars: dict[str, Any]) -> str:
    if not bars:
        return "No bar data available."
    lines = []
    if isinstance(bars, dict):
        for sym, data in bars.items():
            lines.append(f"  {sym}:")
            if isinstance(data, list):
                for bar in data[-5:]:  # Last 5 bars
                    ts = bar.get("timestamp", "")
                    lines.append(
                        f"    {ts}: O={bar.get('open', 0):.2f} H={bar.get('high', 0):.2f} "
                        f"L={bar.get('low', 0):.2f} "
                        f"C={bar.get('close', 0):.2f} V={bar.get('volume', 0)}"
                    )
            else:
                lines.append(f"    {data}")
    else:
        lines.append(str(bars))
    return "\n".join(lines) if lines else str(bars)


class TradingStrategy:
    """Orchestrates the LLM to produce a TradingPlan from market data."""

    def __init__(self, llm: LLMProvider, repo: TradeRepository) -> None:
        self._llm = llm
        self._repo = repo

    async def analyze(
        self,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
        halal_symbols: list[str],
        snapshots: dict[str, Any],
        bars: dict[str, Any],
        today_pnl: float = 0.0,
        sentiment_text: str = "Sentiment data: not available",
    ) -> TradingPlan:
        """Run the LLM analysis and return a structured TradingPlan.

        Args:
            sentiment_text: Pre-formatted sentiment analysis from FinGPT (optional).
        """
        settings = get_settings()

        portfolio_value = float(
            account.get("portfolio_value", 0) or account.get("equity", 0) or 100000
        )
        today_pnl_pct = today_pnl / portfolio_value if portfolio_value else 0

        system = SYSTEM_PROMPT.format(
            max_position_pct=settings.max_position_pct,
            daily_loss_limit=settings.daily_loss_limit,
            daily_return_target=settings.daily_return_target,
            max_positions=settings.max_simultaneous_positions,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            buying_power=float(account.get("buying_power", 0)),
            portfolio_value=portfolio_value,
            cash=float(account.get("cash", 0)),
            today_pnl=today_pnl,
            today_pnl_pct=today_pnl_pct,
            positions_text=_format_positions(positions),
            halal_symbols=", ".join(halal_symbols),
            snapshots_text=_format_snapshots(snapshots),
            bars_text=_format_bars(bars),
            sentiment_text=sentiment_text,
        )

        t0 = time.monotonic()
        try:
            raw = await self._llm.generate_json(user_prompt, system=system)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            plan = TradingPlan.model_validate(raw)

            # Audit trail
            await self._repo.record_decision(
                provider=settings.llm_provider.value,
                model=self._llm.model,
                prompt_summary=f"Analyzed {len(halal_symbols)} halal symbols, "
                f"{len(positions)} positions, buying_power=${account.get('buying_power', 0)}",
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
                "LLM analysis complete in %dms: %d buys, %d sells, %d holds",
                elapsed_ms,
                len(plan.buys),
                len(plan.sells),
                len(plan.holds),
            )
            return plan

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.error("LLM analysis failed after %dms: %s", elapsed_ms, e)
            await self._repo.record_decision(
                provider=settings.llm_provider.value,
                model=self._llm.model,
                prompt_summary="FAILED analysis",
                raw_response=str(e),
                execution_ms=elapsed_ms,
            )
            # Return an empty plan on failure (safe default)
            return TradingPlan(
                market_outlook="Analysis failed — holding positions",
                risk_notes=str(e),
            )
