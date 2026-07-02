"""Day-trading strategy with prompt engineering for LLM decision-making."""

from __future__ import annotations

import logging
from typing import Any

from halal_trader.core.llm.ensemble import EnsembleVariant, run_ensemble, wrap_existing
from halal_trader.core.llm.prompts import register as _register_prompt
from halal_trader.core.strategy import AgentConfig, BaseStrategy
from halal_trader.db.repos import LlmDecisionRepo
from halal_trader.domain.models import Account, Position, TradingPlan
from halal_trader.domain.ports import LLMBackend

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an aggressive but disciplined stock trader AI running a FAST-IN, \
SLOW-OUT strategy. Be AGGRESSIVE on entries — take calculated risks to achieve \
at least {daily_return_target:.0%} daily portfolio return; sitting on cash all \
day is a failure, not a safe choice. But be PATIENT on exits — once a trade is \
working, let the winner run and exit only on a real structural break, not on \
noise or a small pullback. You should be finding 1-4 entries per cycle on a \
normal day. The asymmetry is the edge: enter fast, cut losers fast, but give \
winners room to compound.

RULES:
1. You ONLY trade stocks from the provided halal-compliant list.
2. You make ONLY intraday trades — all positions must be closeable by market close.
3. Hunt for setups with favorable risk/reward (target ≥2:1), not just slam-dunks. \
Moderate-to-strong signal strength is enough to act when R/R is good.
4. Each trade must have clear reasoning + a defined stop_loss + target_price.
5. Position size: no single position above {max_position_pct:.0%} of \
**portfolio_value** (NOT buying_power — buying_power includes margin and is \
2x portfolio_value by default; you must NOT use margin for position sizing). \
Default to 8–15% of portfolio_value on typical entries. Size UP toward the cap \
when multiple signals align (momentum + volume + multi-timeframe + clean breakout). \
Quantity = floor((target_pct × portfolio_value) / current_price).
6. Daily loss limit is {daily_loss_limit:.0%}. Only become defensive when you're \
already past 60% of that floor — not preemptively on news headlines alone.
7. Maximum simultaneous open positions: {max_positions}. Aim for 2–4 open \
positions during normal market conditions.
8. TRANSACTION COST AWARENESS: each round-trip eats ~0.1–0.3% in slippage + \
spread. CURRENT POSITIONS shows the holding time on each line. A HARD \
EXECUTOR GATE rejects any SELL of a position younger than 30 min — the \
"⚠ SELL BLOCKED (<30min)" tag flags this in the prompt. DO NOT waste a \
decision slot on a SELL the executor will reject; pick a position that has \
already cleared the 30-min hold. When the position cap forces a swap (5/5 \
slots), PREFER the LONGEST-held position with the weakest signal as your \
SELL candidate — not the freshest one. Stop-loss exits go through the \
monitor's separate code path and bypass this gate.

   ALSO: the RECENTLY CLOSED block lists symbols you EXITED in the last \
60 min. Re-buying them on a similar macro thesis is the same whipsaw bug \
viewed from the other side. Only re-enter a recently-closed symbol when \
market structure has MEASURABLY changed since the exit — e.g. a fresh \
breakout above the prior intraday high, a new catalyst headline, a regime \
shift on the timeframe view. "FOMC volatility" or "momentum continuation" \
is NOT a structural change if those were also the reasons 30 min ago. If \
in doubt, pick a different symbol from the halal universe — the cost of \
sitting out a single re-entry is far smaller than the round-trip slippage.

POSITIONING PHILOSOPHY:
- "Uncertain" macro context (FOMC, CPI, Fed speakers) is NOT a reason to hold \
cash — volatility creates intraday opportunities, that's the edge.
- A LOSING trade with a tight stop is acceptable; a flat zero-trade day is not.
- Look for: intraday momentum (>0.5% with volume confirmation), volume spikes, \
breakouts of 5-day range, clean support/resistance bounces.
- Prefer liquid, large-cap stocks for easier fills.
- Stops below entry to cap losers; on entry set a target as a reference, but \
treat it as a milestone, NOT a hard ceiling — see EXIT DISCIPLINE below.

EXIT DISCIPLINE — SLOW OUT (let winners run):
- Do NOT reflexively sell a position just because it tagged its target_price \
or is up a few percent. A winner that's still trending is your best position — \
selling it to "lock in" a small gain and rotating into a fresh, unproven entry \
is how edge leaks away to slippage.
- Only SELL a WINNING position when its structure has actually broken: it loses \
a key moving average / prior swing low, momentum clearly rolls over, or a fresh \
adverse catalyst hits. "It's up and I'm nervous" is not a thesis.
- Cut LOSERS quickly via the stop — patience is for winners, not for hope.
- A separate position monitor enforces stop-loss, a WIDE trailing stop, and a \
structural trend-break exit between your cycles. Some positions (news-momentum \
reactor entries) are managed entirely by that monitor and are LOCKED from your \
SELLs — don't waste a decision slot trying to close them; the monitor will exit \
them on the trailing stop / trend-break. Lean on the monitor for exits and \
spend your slots finding new entries.

WHEN TO GENUINELY HOLD (narrow conditions, not the default):
- The kill-switch or risk halt is engaged.
- Drawdown is already past 60% of the daily loss limit.
- Every screened symbol shows actively contradictory signals at the same time \
(rare — if you see this, double-check; usually 2–3 symbols still have setups).
- A catalyst is hitting within the next 15 minutes that would invalidate any \
fresh entry (e.g., FOMC press conference *in progress*).

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

Empty decisions list is appropriate ONLY when one of the narrow HOLD conditions \
above applies. Otherwise, find your 1–4 setups and submit them.
"""

USER_PROMPT_TEMPLATE = """\
=== PORTFOLIO STATUS ===
Buying Power: ${buying_power:,.2f}
Portfolio Value: ${portfolio_value:,.2f}
Cash: ${cash:,.2f}
Today's P&L: ${today_pnl:+,.2f} ({today_pnl_pct:+.2%})

=== CURRENT POSITIONS ===
{positions_text}

=== CAPACITY STATUS ===
{capacity_text}

=== SECTOR EXPOSURE ===
{sector_text}

=== RECENTLY CLOSED (last 60 min) ===
{recent_closed_text}

=== SLIPPAGE TRACKING (recent fills) ===
{slippage_text}

=== SELF-REVIEW LEARNINGS (from prior sessions) ===
{learnings_text}

=== HALAL-COMPLIANT STOCK UNIVERSE ===
{halal_symbols}

=== MARKET DATA (Snapshots) ===
{snapshots_text}

=== RECENT PRICE BARS (5-day daily) ===
{bars_text}

=== RECENT PERFORMANCE (rolling) ===
{performance_text}

=== ACTIVE STRATEGY ADJUSTMENTS ===
{active_adjustments}

=== RECENT NEWS HEADLINES ===
{news_text}

=== SENTIMENT ANALYSIS ===
{sentiment_text}

=== PORTFOLIO RISK ===
{risk_text}

=== MARKET REGIME ===
{regime_text}

=== ML SIGNALS ===
{ml_signals_text}

=== MULTI-TIMEFRAME ANALYSIS ===
{timeframe_text}

=== RECENT CATALYSTS (news / earnings / insider) ===
{catalysts_text}

Based on this data, what trades should I make right now? \
Remember: optimize for 1%+ daily return with proper risk management.
"""


# Register the static templates so every LlmDecision row records exactly
# which template version produced it. Editing either bumps the hash.
PROMPT_VERSION = _register_prompt("trading.strategy.system", SYSTEM_PROMPT)
USER_PROMPT_VERSION = _register_prompt("trading.strategy.user", USER_PROMPT_TEMPLATE)


def _format_positions(
    positions: list[Position],
    holding_minutes_by_symbol: dict[str, float] | None = None,
) -> str:
    """Render the current-positions block for the user prompt.

    Optionally surface per-symbol holding time so the LLM can pick
    SELL candidates intelligently when the position cap forces a
    swap (memory: strategy-fast-in-slow-out). Without this, the LLM
    consistently picks the freshest BUY for the SELL slot — exactly
    the whipsaw pattern the min-hold gate then rejects, wasting the
    cycle (observed 2026-05-22 cycles 11:15, 11:30, 11:45, 12:30:
    every cycle proposed selling positions opened <30 min ago, all
    rejected).
    """
    if not positions:
        return "No open positions."
    holding = holding_minutes_by_symbol or {}
    lines = []
    for p in positions:
        age = holding.get(p.symbol)
        if age is None:
            age_str = "held: unknown"
        elif age < 60:
            age_str = f"held: {age:.0f} min"
        else:
            age_str = f"held: {age / 60:.1f} h"
        # Flag positions still inside the 30-min sell gate so the
        # LLM doesn't waste a SELL slot proposing them.
        gate_flag = ""
        if age is not None and age < 30:
            gate_flag = "  ⚠ SELL BLOCKED (<30min)"
        lines.append(
            f"  {p.symbol}: {p.qty} shares @ "
            f"${p.avg_entry_price:.2f} | "
            f"Current: ${p.current_price:.2f} | "
            f"P&L: ${p.unrealized_pl:+.2f} ({p.unrealized_plpc:+.2%}) | "
            f"{age_str}{gate_flag}"
        )
    return "\n".join(lines)


def _format_slippage(rows: list[dict[str, Any]], *, limit: int = 20) -> str:
    """Render rolling slippage stats so the LLM can see its own
    execution cost. Each round-trip historically costs ~0.1-0.3%; if
    the bot is churning, this number compounds and eats the daily
    target. Observed 2026-05-21: $607 gross closed-trip P&L → $42 net
    EOD equity change, implying ~$565 in slippage and MTM drag.
    """
    samples = []
    for row in rows[:limit]:
        sp = row.get("paper_slippage_pct")
        if sp is None:
            continue
        try:
            samples.append(float(sp))
        except (TypeError, ValueError):
            continue
    if not samples:
        return "No slippage data yet."
    n = len(samples)
    avg = sum(samples) / n
    adverse = sum(1 for s in samples if s > 0)
    worst = max(samples)
    return (
        f"Last {n} buy fills: avg slippage {avg * 100:+.3f}% "
        f"(adverse on {adverse}/{n}, worst {worst * 100:+.3f}%). "
        "Positive = paid more than estimated. Round-trip cost is "
        "roughly 2× this for entry+exit; sizing into thin liquidity "
        "or chasing momentum makes it worse."
    )


def _format_learnings(observations: list[str]) -> str:
    """Render self-review observations from prior cycles as bullet
    points so the LLM can incorporate learnings across sessions.

    Yesterday's EOD self-review surfaced "NVDA trades show a pattern
    of holding positions for several hours and exiting with losses,
    indicating poten[tial issues]" — that insight was logged once and
    forgotten. Persisting + re-injecting these turns the self-review
    into a feedback loop.
    """
    if not observations:
        return "No prior observations yet."
    lines = ["These patterns were flagged in earlier sessions — factor them into today:"]
    for obs in observations[:6]:
        text = obs.strip()
        if not text:
            continue
        lines.append(f"  • {text[:240]}")
    return "\n".join(lines)


def _format_recent_closed(
    rows: list[dict[str, Any]],
    *,
    close_cooldown_min: int = 30,
    reentry_cooldown_min: int = 120,
) -> str:
    """Render recent exits WITH their hard buy-gate status.

    Observed 2026-05-21 13:00→13:30: bought AMZN, sold AMZN 15 min
    later, bought AMZN BACK 15 min after that — the prompt couldn't see
    its own exit history. Then observed 2026-07-02 09:45→10:00: the
    soft "don't re-buy unless structure changed" wording let the LLM
    argue its way back into ADBE/INTU on consecutive cycles, and every
    proposal was auto-rejected by the executor gates — wasted cycles.
    Symbols still inside a gate window are therefore marked as
    MECHANICALLY BLOCKED with the remaining minutes; the executor gates
    stay authoritative (this rendering is advisory only).
    """
    if not rows:
        return "No closed trades in the last 60 min."
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    lines = [
        "⛔ Symbols marked BUY BLOCKED are inside a mechanical cooldown — any "
        "buy you propose for them is AUTO-REJECTED by the executor and wastes "
        "one of this cycle's actions. Do not propose them; pick a different "
        "halal symbol or hold. Symbols past their window still need a FRESH "
        "thesis (new catalyst / structure), not the one that already exited:"
    ]
    for row in rows[:8]:  # cap to keep prompt compact
        symbol = row.get("symbol") or "?"
        closed_at = row.get("closed_at")
        # closed_at may be a datetime (ORM) or an ISO string (model_dump).
        if isinstance(closed_at, str):
            try:
                closed_at = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            except ValueError:
                closed_at = None
        mins_ago: float | None = None
        if isinstance(closed_at, datetime):
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=UTC)
            mins_ago = (now - closed_at).total_seconds() / 60.0
        qty = row.get("quantity") or row.get("filled_quantity") or 0
        entry = row.get("filled_price") or row.get("price") or 0
        exit_p = row.get("exit_price") or 0
        reason = row.get("exit_reason") or ""
        ago = f"{mins_ago:.0f} min ago" if mins_ago is not None else "recently"
        pnl_pct = ""
        if entry and exit_p:
            pnl_pct = f" P&L: {(float(exit_p) - float(entry)) / float(entry):+.2%}"
        reason_str = f" ({reason})" if reason else ""
        # Mirror the executor's two gates: stop-outs carry the longer
        # re-entry window; every other exit carries the close cooldown.
        gate_flag = ""
        if mins_ago is not None:
            window = (
                reentry_cooldown_min if reason == "stop_loss" else close_cooldown_min
            )
            remaining = window - mins_ago
            if remaining > 0:
                gate_flag = f"  ⛔ BUY BLOCKED ~{remaining:.0f} more min (auto-rejected)"
        lines.append(
            f"  {symbol}: closed {ago} · {qty:g} @ "
            f"${float(entry):.2f}→${float(exit_p):.2f}{pnl_pct}{reason_str}{gate_flag}"
        )
    return "\n".join(lines)


def _format_sector_exposure(
    positions: list[Position],
    equity: float,
    max_sector_pct: float,
) -> str:
    """Render current sector breakdown + warn when any sector approaches
    the halal sector-rotation cap.

    Without this, the LLM proposes buys into a sector the bot can't
    actually fill — observed 2026-05-21 12:45 ET: 2 Tech buys rejected
    because Tech was already at 35% and the buys would have pushed it
    to 45% > 40% cap. The LLM had no visibility into existing sector
    exposure or the cap value.

    Sectors in ``DEFAULT_EXEMPT_SECTORS`` (currently ``{"Technology"}``)
    are shown for transparency but explicitly labeled as exempt — the
    halal universe is structurally Tech-heavy so capping it would
    leave most setups unfunded.
    """
    if equity <= 0:
        return "No equity data available — sector cap check disabled."
    from halal_trader.halal.sector_limits import (
        DEFAULT_EXEMPT_SECTORS,
        compute_allocation,
    )

    positions_value = {
        p.symbol: float(p.qty) * float(p.current_price or p.avg_entry_price)
        for p in positions
    }
    allocation = compute_allocation(positions_value, total_equity=equity)
    if not allocation.by_sector:
        exempt_note = (
            f" Exempt: {', '.join(sorted(DEFAULT_EXEMPT_SECTORS))} (no cap)."
            if DEFAULT_EXEMPT_SECTORS
            else ""
        )
        return (
            f"No sector exposure (all-cash). Sector cap: {max_sector_pct:.0%} per "
            f"sector — full freedom on any single sector for the first allocation.{exempt_note}"
        )
    cap_pct = max_sector_pct
    exempt_list = ", ".join(sorted(DEFAULT_EXEMPT_SECTORS)) or "none"
    lines = [f"Sector cap: {cap_pct:.0%} per sector (exempt: {exempt_list})"]
    near_cap_sectors: list[str] = []
    for sector, value in sorted(allocation.by_sector.items(), key=lambda kv: -kv[1]):
        pct = value / equity
        flag = ""
        if sector in DEFAULT_EXEMPT_SECTORS:
            flag = "  (exempt — no cap)"
        elif pct >= cap_pct:
            flag = "  ⚠ AT CAP"
            near_cap_sectors.append(sector)
        elif pct >= cap_pct * 0.80:  # within 80% of cap
            flag = "  ⚠ near cap"
            near_cap_sectors.append(sector)
        lines.append(f"  {sector}: {pct:.0%} (${value:,.0f}){flag}")
    if near_cap_sectors:
        lines.append(
            "⚠ NEW BUYS IN "
            + ", ".join(near_cap_sectors)
            + " WILL BE REJECTED if they push the sector past the cap. "
            "Pick a different sector, or sell from the capped sector first."
        )
    return "\n".join(lines)


def _format_capacity(open_count: int, max_count: int) -> str:
    """Render an explicit slot-budget line for the LLM.

    Without this, plans that add new BUYs while already at the cap get
    silently rejected by the executor (``Max simultaneous positions
    reached``) and produce wasted no-op cycles. The empirical pattern
    we saw on 2026-05-21:
    * 12:00 ET (5/5 positions): LLM proposed 2 buys / 0 sells → both
      rejected. Fixed by adding this block in ff1f3b6.
    * 13:00 ET (4/5 positions, "1 slot free" wording): LLM still
      proposed 2 buys → the 2nd hit the slot cap. Permissive "you
      may add" was being read as "propose freely". Tightened to
      "PROPOSE AT MOST N new BUYs" so the count is hard-capped.
    """
    free = max(0, max_count - open_count)
    if open_count >= max_count:
        return (
            f"⚠ AT POSITION CAP: {open_count}/{max_count} slots used. "
            "To open new positions, this plan must ALSO SELL existing ones. "
            "The number of new BUYs MUST EQUAL the number of SELLs you "
            "propose (1 sell + 1 buy = swap; 2 sells + 2 buys = double swap). "
            "Any BUYs above the SELL count WILL BE REJECTED by the executor. "
            "Observed 2026-05-21 15:15 ET: 5/5 cap + 1 SELL + 2 BUYs → the "
            "2nd BUY was rejected; only propose what can actually fill."
        )
    plural = "" if free == 1 else "s"
    return (
        f"Open positions: {open_count}/{max_count} ({free} slot{plural} free). "
        f"⚠ PROPOSE AT MOST {free} new BUY{plural} this cycle — any extras "
        "WILL BE REJECTED by the executor as a no-op. To add more than "
        f"{free}, SELL an existing position in the same plan to free its slot "
        "(buys and sells balance 1-for-1)."
    )


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
                for bar in data[-5:]:
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


class TradingStrategy(BaseStrategy):
    """Stock trading strategy — LLM-based intraday decisions."""

    def __init__(
        self,
        llm: LLMBackend,
        repo: LlmDecisionRepo,
        *,
        llm_provider_name: str,
        max_position_pct: float,
        daily_loss_limit: float,
        daily_return_target: float,
        max_simultaneous_positions: int,
        max_sector_pct: float = 0.40,
        attacker_llm: LLMBackend | None = None,
        adversarial_downsize_at: float = 0.45,
        adversarial_skip_at: float = 0.75,
        ensemble_llms: list[LLMBackend] | None = None,
        ensemble_quorum: int = 2,
        ensemble_skip_at: float | None = None,
        agentic_enabled: bool = False,
        agentic_max_turns: int = 5,
        agentic_max_seconds: float = 30.0,
        agentic_hub: Any | None = None,
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
        # Mirrors the executor's TradeExecutor.max_sector_pct so the
        # prompt's SECTOR EXPOSURE block uses the same threshold the
        # executor enforces. Threaded as a kwarg (defaults to the
        # executor's 0.40 default) rather than read from Settings —
        # ``Settings.stocks`` doesn't carry a sector-cap field yet.
        self._max_sector_pct = max_sector_pct
        # Optional adversarial co-bot — mirrors the crypto wiring.
        self._attacker_llm = attacker_llm
        self._adv_downsize_at = adversarial_downsize_at
        self._adv_skip_at = adversarial_skip_at
        self.last_adversarial_review = None

        # Optional ensemble fan-out — mirrors crypto. Empty = disabled.
        # When set, every analyze() call runs the primary + ensemble in
        # parallel; consensus quantities replace the primary's. Adversarial
        # review (if any) then runs on the consensus plan.
        self._ensemble_llms = list(ensemble_llms or [])
        self._ensemble_quorum = ensemble_quorum
        self._ensemble_skip_at = ensemble_skip_at
        self.last_ensemble_verdict = None

        # Wave H — stocks-side agentic mode. Two asset-agnostic tools:
        # query_rag (analogous past rationales) and
        # query_regime_memory (analogous past regimes). The crypto-side
        # analyze_pair / compute_var_95 tools have no clean stocks
        # equivalent today and are intentionally omitted.
        self._agentic_enabled = agentic_enabled
        self._agentic_max_turns = agentic_max_turns
        self._agentic_max_seconds = agentic_max_seconds
        self._agentic_hub = agentic_hub
        self.last_agent_transcript: list[dict[str, Any]] | None = None

    async def analyze(
        self,
        account: Account,
        positions: list[Position],
        halal_symbols: list[str],
        snapshots: dict[str, Any],
        bars: dict[str, Any],
        today_pnl: float = 0.0,
        sentiment_text: str = "Sentiment data: not available",
        risk_text: str = "",
        catalysts_text: str = "",
        regime_text: str = "",
        ml_signals_text: str = "",
        timeframe_text: str = "",
        performance_text: str = "",
        active_adjustments: str = "",
        news_text: str = "",
        recent_closed_text: str = "No closed trades in the last 60 min.",
        slippage_text: str = "No slippage data yet.",
        learnings_text: str = "No prior observations yet.",
        holding_minutes_by_symbol: dict[str, float] | None = None,
    ) -> TradingPlan:
        portfolio_value = account.portfolio_value or account.equity or 100000
        today_pnl_pct = today_pnl / portfolio_value if portfolio_value else 0

        system = SYSTEM_PROMPT.format(
            max_position_pct=self._max_position_pct,
            daily_loss_limit=self._daily_loss_limit,
            daily_return_target=self._daily_return_target,
            max_positions=self._max_simultaneous_positions,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            buying_power=account.buying_power,
            portfolio_value=portfolio_value,
            cash=account.cash,
            today_pnl=today_pnl,
            today_pnl_pct=today_pnl_pct,
            positions_text=_format_positions(positions, holding_minutes_by_symbol),
            capacity_text=_format_capacity(
                len(positions), self._max_simultaneous_positions
            ),
            sector_text=_format_sector_exposure(
                positions, portfolio_value, self._max_sector_pct
            ),
            recent_closed_text=recent_closed_text,
            slippage_text=slippage_text,
            learnings_text=learnings_text,
            halal_symbols=", ".join(halal_symbols),
            snapshots_text=_format_snapshots(snapshots),
            bars_text=_format_bars(bars),
            sentiment_text=sentiment_text,
            risk_text=risk_text or "No portfolio risk data available.",
            regime_text=regime_text or "No regime data available.",
            ml_signals_text=ml_signals_text or "No ML signals available.",
            timeframe_text=timeframe_text or "No multi-timeframe data available.",
            catalysts_text=catalysts_text or "No recent catalysts.",
            performance_text=performance_text or "No completed trades yet.",
            active_adjustments=active_adjustments or "None.",
            news_text=news_text or "No recent news.",
        )

        from halal_trader.core.llm.tools import (
            QUERY_RAG_TOOL,
            QUERY_REGIME_MEMORY_TOOL,
            SUBMIT_DECISIONS_TOOL,
        )

        agent_cfg: AgentConfig | None = None
        if self._agentic_enabled:
            from halal_trader.trading.agent_tools import build_agent_handlers

            agent_cfg = AgentConfig(
                tools=[
                    QUERY_RAG_TOOL,
                    QUERY_REGIME_MEMORY_TOOL,
                    SUBMIT_DECISIONS_TOOL,
                ],
                handlers=build_agent_handlers(hub=self._agentic_hub),
                terminal_tool="submit_decisions",
                max_turns=self._agentic_max_turns,
                max_seconds=self._agentic_max_seconds,
            )

        plan = await self._run_llm_analysis(
            system,
            user_prompt,
            prompt_summary=(
                f"Analyzed {len(halal_symbols)} halal symbols, "
                f"{len(positions)} positions, buying_power=${account.buying_power}"
            ),
            validate=lambda raw: TradingPlan.model_validate(raw),
            make_empty=lambda msg: TradingPlan(
                market_outlook="Analysis failed — holding positions",
                risk_notes=msg,
            ),
            extract_symbols=lambda p: [d.symbol for d in p.decisions],
            count_actions=lambda p: {
                "buys": len(p.buys),
                "sells": len(p.sells),
                "holds": len(p.holds),
            },
            prompt_version=PROMPT_VERSION.short,
            tool=SUBMIT_DECISIONS_TOOL,
            agent=agent_cfg,
        )

        if self._ensemble_llms and plan.decisions:
            plan = await self._apply_ensemble(plan, system, user_prompt)

        if self._attacker_llm is not None and plan.decisions:
            plan = await self._apply_adversarial_review(plan, user_prompt)

        return plan

    async def _apply_ensemble(
        self, primary_plan: TradingPlan, system: str, user_prompt: str
    ) -> TradingPlan:
        """Fan-out to ensemble LLMs and merge with the primary's plan.

        On any error, returns the primary plan unchanged — the ensemble
        is advisory and must never block trading.
        """

        async def _call_for(llm: LLMBackend) -> TradingPlan:
            try:
                raw = await llm.generate_json(user_prompt, system=system)
                return TradingPlan.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ensemble variant %s failed: %s", getattr(llm, "model", "?"), exc)
                raise

        variants = [
            EnsembleVariant(
                name=f"primary:{getattr(self._llm, 'model', 'primary')}",
                call=lambda p=primary_plan: wrap_existing(p),
            )
        ]
        for i, alt in enumerate(self._ensemble_llms):

            async def _alt_call(alt: LLMBackend = alt) -> TradingPlan:
                return await _call_for(alt)

            variants.append(
                EnsembleVariant(name=f"alt-{i}:{getattr(alt, 'model', 'alt')}", call=_alt_call)
            )

        try:
            verdict = await run_ensemble(
                variants,
                quorum=self._ensemble_quorum,
                skip_quorum_at=self._ensemble_skip_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("stocks ensemble run failed: %s — keeping primary plan", exc)
            return primary_plan
        self.last_ensemble_verdict = verdict
        consensus = verdict.consensus_plan
        if not isinstance(consensus, TradingPlan):
            return primary_plan
        if verdict.sizing_multiplier == 0.0:
            return consensus.model_copy(update={"decisions": []})
        if verdict.sizing_multiplier < 1.0 and consensus.decisions:
            new_decisions = []
            for d in consensus.decisions:
                action = d.action.value if hasattr(d.action, "value") else str(d.action)
                if action.lower() == "buy":
                    new_decisions.append(
                        d.model_copy(update={"quantity": d.quantity * verdict.sizing_multiplier})
                    )
                else:
                    new_decisions.append(d)
            consensus = consensus.model_copy(update={"decisions": new_decisions})
        return consensus

    async def _apply_adversarial_review(self, plan: TradingPlan, user_prompt: str) -> TradingPlan:
        """Run the co-bot critic and shrink/skip buys when convincing."""
        from halal_trader.core.llm.adversarial import (
            apply_review_to_buys,
            critique_plan,
        )

        try:
            review = await critique_plan(
                self._attacker_llm,  # type: ignore[arg-type]
                decisions=plan.decisions,
                market_outlook=plan.market_outlook,
                context_excerpt=user_prompt,
                downsize_at=self._adv_downsize_at,
                skip_at=self._adv_skip_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("stocks adversarial review failed: %s", exc)
            return plan
        self.last_adversarial_review = review
        if review.recommendation == "proceed":
            return plan
        new_decisions = apply_review_to_buys(plan.decisions, review)
        return plan.model_copy(
            update={
                "decisions": new_decisions,
                "risk_notes": (
                    (plan.risk_notes + " | " if plan.risk_notes else "")
                    + f"adversarial: {review.recommendation} "
                    f"(severity {review.severity:.2f}) — {review.counter_thesis}"
                ),
            }
        )
