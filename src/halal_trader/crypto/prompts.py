"""Prompt assembly for the crypto strategy.

Both live (``CryptoTradingStrategy.analyze``) and backtest
(``LLMBacktestEngine.run``) call :func:`build_prompts` so prompt
engineering iterations transfer between them. The cache key in the
backtest hashes the assembled prompt strings, so any template change
busts the cache automatically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from halal_trader.core.llm.prompts import register as _register_prompt
from halal_trader.crypto.indicators import format_indicators_for_prompt
from halal_trader.domain.models import CryptoAccount, Kline

# ── Templates ──────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert crypto scalping AI. Analyze technical indicators and real-time \
market data for the provided pairs and emit precise buy/sell/hold decisions on a \
1-minute timeframe to achieve at least {daily_return_target:.0%} daily return.

RULES:
1. Trade ONLY pairs from the provided halal-compliant list.
2. Scalping horizon: 1–60 minute holds.
3. Hard sizing: (quantity × current_price) MUST be STRICTLY less than the \
"Max Position Size" dollar value shown in the portfolio status; use at most 90% \
of that. Never exceed Available USDT.
4. Quantity must comply with EXCHANGE TRADING RULES: step size multiple, ≥ min_qty, \
and notional (quantity × price) ≥ min_notional.
5. Stop trading for the day if cumulative losses hit {daily_loss_limit:.0%} of portfolio.
6. Output ONLY a single JSON object — no prose, no markdown.

STRATEGY:
- Vol ratios >1.2× confirm moves; VWAP is intraday S/R; order-book imbalance signals \
short-term pressure.
- Default exits are {stop_loss_pct:.1%} SL / {take_profit_pct:.1%} TP (executor applies \
these unless you override per-decision).
- If 2+ indicators align, take the trade. Confidence ≥0.7 → 50–90% of Max Position Size; \
moderate → 20–50%. Never trade < $50 notional.
- Trending: trade with the trend; Ranging: BB mean-reversion; High-vol: smaller / wider stops.
- Reduce sizes when correlation > 0.7. Be conservative on negative heat / approaching drawdown.

Maximum positions: {max_positions}. {active_adjustments}

OUTPUT JSON SCHEMA (keep it minimal — every extra field costs tokens):
{{
  "decisions": [
    {{"action": "buy"|"sell"|"hold", "symbol": "BTCUSDT", "quantity": 0.05, "confidence": 0.85}}
  ],
  "reasoning": "One concise sentence — the rationale for the *plan as a whole*",
  "market_outlook": "Brief market view"
}}

Per-decision SL/TP overrides are OPTIONAL — only include them when your read differs \
from the {stop_loss_pct:.1%}/{take_profit_pct:.1%} defaults. When you do include them, \
use exactly the field names "stop_loss" and "target_price" (numeric, USDT prices).
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

=== HALAL TRADING PAIRS ===
{halal_pairs}

=== EXCHANGE TRADING RULES ===
{exchange_rules_text}

=== TECHNICAL INDICATORS ===
{indicators_text}

=== ORDER BOOK ===
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

=== MICROSTRUCTURE (read-only signal — execute SPOT only) ===
{microstructure_text}

=== RECENT BREAKING NEWS ===
{news_text}

=== YOUR RECENT PERFORMANCE (last 7 days) ===
{performance_text}

=== TASK ===
Analyze the data and decide on trades. Target {daily_return_target:.0%} daily return.
Output the JSON specified in the system prompt.
"""


# ── Dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyParams:
    """Knobs the system prompt + sizing logic depend on."""

    max_position_pct: float
    daily_loss_limit: float
    daily_return_target: float
    max_positions: int
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class PromptContext:
    """All inputs the prompt template needs.

    Strings default to empty so callers (especially the backtest) can
    leave optional sections off; ``build_prompts`` collapses any empty
    sections out of the rendered user prompt.
    """

    account: CryptoAccount
    positions_text: str = ""
    halal_pairs: list[str] = field(default_factory=list)
    klines_by_symbol: dict[str, list[Kline]] = field(default_factory=dict)
    orderbooks: dict[str, dict[str, Any]] = field(default_factory=dict)
    today_pnl: float = 0.0
    performance_text: str = ""
    sentiment_text: str = ""
    timeframe_text: str = ""
    ml_signals_text: str = ""
    regime_text: str = ""
    active_adjustments: str = ""
    exchange_rules_text: str = ""
    indicators_cache: dict[str, dict[str, Any]] | None = None
    open_position_count: int = 0
    risk_text: str = ""
    microstructure_text: str = ""
    news_text: str = ""


# ── Builders ───────────────────────────────────────────────────


def build_indicators_text(
    klines_by_symbol: dict[str, list[Kline]],
    indicators_cache: dict[str, dict[str, Any]] | None,
) -> str:
    parts: list[str] = []
    for pair, klines in klines_by_symbol.items():
        if indicators_cache is not None and pair in indicators_cache:
            inds = indicators_cache[pair]
        else:
            from halal_trader.crypto.indicators import compute_all

            inds = compute_all(klines)
        parts.append(format_indicators_for_prompt(pair, inds))
    return "\n".join(parts) if parts else "No indicator data available."


def build_orderbook_text(orderbooks: dict[str, dict[str, Any]]) -> str:
    if not orderbooks:
        return "No order book data available."
    lines: list[str] = []
    for pair, book in orderbooks.items():
        bids = book.get("bids", []) or []
        asks = book.get("asks", []) or []
        if not bids or not asks:
            continue
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        bid_vol = sum(float(q) for _, q in bids[:10])
        ask_vol = sum(float(q) for _, q in asks[:10])
        imb = "buy-side" if bid_vol > ask_vol else "sell-side"
        lines.append(
            f"{pair}: bid ${best_bid:,.2f} / ask ${best_ask:,.2f} "
            f"(spread ${spread:.2f}) — {imb} imbalance"
        )
    return "\n".join(lines) if lines else "No order book data available."


def build_prompts(ctx: PromptContext, params: StrategyParams) -> tuple[str, str]:
    """Render the (system, user) prompt pair from a ``PromptContext``.

    Pure function; no IO, no LLM calls.  Both live ``analyze`` and the
    backtest engine use this so prompt edits transfer between them.
    """
    portfolio_value = ctx.account.total_balance_usdt or 1000
    today_pnl_pct = ctx.today_pnl / portfolio_value if portfolio_value else 0

    indicators_text = build_indicators_text(ctx.klines_by_symbol, ctx.indicators_cache)
    orderbook_text = build_orderbook_text(ctx.orderbooks)

    adjustments_block = ""
    if ctx.active_adjustments:
        adjustments_block = (
            "ACTIVE STRATEGY ADJUSTMENTS (from your own performance review):\n"
            + ctx.active_adjustments
        )

    at_max = ctx.open_position_count >= params.max_positions

    system = SYSTEM_PROMPT.format(
        max_position_pct=params.max_position_pct,
        daily_loss_limit=params.daily_loss_limit,
        daily_return_target=params.daily_return_target,
        max_positions=params.max_positions,
        active_adjustments=adjustments_block,
        stop_loss_pct=params.stop_loss_pct,
        take_profit_pct=params.take_profit_pct,
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

    pct_limit = portfolio_value * params.max_position_pct
    spendable = (
        ctx.account.usdt_free if ctx.account.usdt_free > 0 else ctx.account.available_balance_usdt
    )
    max_position_value = min(pct_limit, spendable) if spendable > 0 else pct_limit

    if at_max:
        position_limit_warning = (
            "⚠ POSITION LIMIT REACHED — you MUST fully close (sell ALL quantity of) "
            "an existing position before any new buys will be accepted. A partial "
            "sell does NOT free the slot. Sell the ENTIRE holding of your weakest "
            "position to open a slot for a better opportunity."
        )
    elif ctx.open_position_count >= params.max_positions - 1:
        position_limit_warning = "⚠ Only 1 position slot remaining — be selective with buys."
    else:
        position_limit_warning = ""

    user_prompt = USER_PROMPT_TEMPLATE.format(
        total_balance=ctx.account.total_balance_usdt,
        available_balance=ctx.account.available_balance_usdt,
        in_order=ctx.account.in_order_usdt,
        max_position_value=max_position_value,
        max_position_pct=params.max_position_pct,
        today_pnl=ctx.today_pnl,
        today_pnl_pct=today_pnl_pct,
        open_position_count=ctx.open_position_count,
        max_positions=params.max_positions,
        position_limit_warning=position_limit_warning,
        positions_text=ctx.positions_text or "No open positions.",
        halal_pairs=", ".join(ctx.halal_pairs),
        indicators_text=indicators_text,
        exchange_rules_text=ctx.exchange_rules_text or "No exchange trading rules available.",
        orderbook_text=orderbook_text,
        sentiment_text=ctx.sentiment_text or "No sentiment data available.",
        timeframe_text=ctx.timeframe_text or "No multi-timeframe data available.",
        ml_signals_text=ctx.ml_signals_text or "No ML model data available.",
        regime_text=ctx.regime_text or "No regime data available.",
        risk_text=ctx.risk_text or "No portfolio risk data available.",
        microstructure_text=ctx.microstructure_text or "No microstructure data available.",
        news_text=ctx.news_text or "No recent breaking news.",
        performance_text=ctx.performance_text or "No completed trades yet.",
        daily_return_target=params.daily_return_target,
    )

    optional_sections: list[tuple[str, str, str]] = [
        ("=== SOCIAL SENTIMENT ===", ctx.sentiment_text, "No sentiment data available."),
        (
            "=== MULTI-TIMEFRAME ANALYSIS ===",
            ctx.timeframe_text,
            "No multi-timeframe data available.",
        ),
        ("=== ML MODEL SIGNALS ===", ctx.ml_signals_text, "No ML model data available."),
        ("=== MARKET REGIME ===", ctx.regime_text, "No regime data available."),
        ("=== PORTFOLIO RISK ===", ctx.risk_text, "No portfolio risk data available."),
        (
            "=== MICROSTRUCTURE (read-only signal — execute SPOT only) ===",
            ctx.microstructure_text,
            "No microstructure data available.",
        ),
        ("=== RECENT BREAKING NEWS ===", ctx.news_text, "No recent breaking news."),
    ]
    for header, value, placeholder in optional_sections:
        if not value:
            user_prompt = user_prompt.replace(f"{header}\n{placeholder}\n\n", "")

    return system, user_prompt


def prompt_cache_key(system: str, user: str) -> str:
    """Stable hash of the (system, user) prompt pair for backtest caching.

    Any change to the templates or the inputs produces a new key, so the
    LLMBacktestEngine cache is invalidated automatically on prompt
    changes — that's the whole point of unifying these two paths.
    """
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x1f")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


# Register the static template (not the per-cycle data) so every LlmDecision
# row can record exactly which template version produced it. Editing
# SYSTEM_PROMPT bumps this hash automatically.
PROMPT_VERSION = _register_prompt("crypto.strategy.system", SYSTEM_PROMPT)
