"""Practice mode harness — Round-5 Wave 20.F.

Paper-trading sessions with the full platform feature set, but the
broker connection is paper-only. Each user has a `PracticeSession`
that:

1. Starts with a seed cash balance.
2. Tracks paper trades + portfolio P&L.
3. Enforces feature-unlock guards keyed off the user's qualified tier
   from `education/academy.py` (Wave 20.A).
4. Reports progression milestones back to the learner-progress hook.

Pinned semantics:

- **Closed-set SessionStatus FSM** — ACTIVE → SUSPENDED → CONCLUDED.
- **Closed-set PracticeFeature ladder** — BASIC_TRADING /
  OPTIONS_PRACTICE / SUKUK_PRACTICE / MARGIN_PRACTICE /
  STRUCTURED_PRODUCTS / LIVE_PROMOTION.
- **Per-feature minimum tier**:
    BASIC_TRADING → BEGINNER (always allowed)
    OPTIONS_PRACTICE → INTERMEDIATE
    SUKUK_PRACTICE → INTERMEDIATE
    MARGIN_PRACTICE → ADVANCED (still paper — but the operator gates
       it because the user must understand the riba framing)
    STRUCTURED_PRODUCTS → ADVANCED
    LIVE_PROMOTION → EXPERT (auto-promote to real-money mode)
- **Session balance** = seed_cash + realised_pnl + unrealised_pnl.
- **Trades against a closed session are rejected.**
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — user IDs masked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum

from halal_trader.education.academy import Tier

# Tier ordering — duplicated locally so this module doesn't reach into
# academy's private internals.
_TIER_ORDER: dict[Tier, int] = {
    Tier.BEGINNER: 0,
    Tier.INTERMEDIATE: 1,
    Tier.ADVANCED: 2,
    Tier.EXPERT: 3,
}


class SessionStatus(str, Enum):
    """Closed-set practice-session FSM ladder."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    CONCLUDED = "concluded"


class PracticeFeature(str, Enum):
    """Closed-set practice-feature ladder."""

    BASIC_TRADING = "basic_trading"
    OPTIONS_PRACTICE = "options_practice"
    SUKUK_PRACTICE = "sukuk_practice"
    MARGIN_PRACTICE = "margin_practice"
    STRUCTURED_PRODUCTS = "structured_products"
    LIVE_PROMOTION = "live_promotion"


_FEATURE_MIN_TIER: dict[PracticeFeature, Tier] = {
    PracticeFeature.BASIC_TRADING: Tier.BEGINNER,
    PracticeFeature.OPTIONS_PRACTICE: Tier.INTERMEDIATE,
    PracticeFeature.SUKUK_PRACTICE: Tier.INTERMEDIATE,
    PracticeFeature.MARGIN_PRACTICE: Tier.ADVANCED,
    PracticeFeature.STRUCTURED_PRODUCTS: Tier.ADVANCED,
    PracticeFeature.LIVE_PROMOTION: Tier.EXPERT,
}


def min_tier_for(feature: PracticeFeature) -> Tier:
    return _FEATURE_MIN_TIER[feature]


def is_feature_unlocked(feature: PracticeFeature, qualified_tier: Tier) -> bool:
    """True iff the qualified_tier ≥ feature's minimum tier."""
    return _TIER_ORDER[qualified_tier] >= _TIER_ORDER[_FEATURE_MIN_TIER[feature]]


@dataclass(frozen=True)
class PracticeTrade:
    """One paper trade."""

    trade_id: str
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: float
    fill_price: float
    placed_at: datetime
    feature: PracticeFeature = PracticeFeature.BASIC_TRADING

    def __post_init__(self) -> None:
        if not self.trade_id or not self.trade_id.strip():
            raise ValueError("trade_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.fill_price <= 0:
            raise ValueError("fill_price must be positive")

    def notional(self) -> float:
        return self.quantity * self.fill_price


@dataclass(frozen=True)
class PracticeSession:
    """One user's paper-trading session."""

    session_id: str
    user_id: str
    seed_cash_usd: float
    qualified_tier: Tier
    started_on: date
    trades: tuple[PracticeTrade, ...] = ()
    realised_pnl_usd: float = 0.0
    unrealised_pnl_usd: float = 0.0
    status: SessionStatus = SessionStatus.ACTIVE
    concluded_on: date | None = None
    promoted_to_live: bool = False

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.seed_cash_usd <= 0:
            raise ValueError("seed_cash_usd must be positive")
        if self.seed_cash_usd > 10_000_000:
            raise ValueError("seed_cash > $10M is suspicious for practice")
        ids = [t.trade_id for t in self.trades]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate trade_id")
        if self.status is SessionStatus.CONCLUDED and self.concluded_on is None:
            raise ValueError("CONCLUDED requires concluded_on")
        if self.concluded_on is not None and self.concluded_on < self.started_on:
            raise ValueError("concluded_on must be ≥ started_on")
        if self.promoted_to_live and self.status is not SessionStatus.CONCLUDED:
            raise ValueError("promotion requires CONCLUDED status")

    def balance_usd(self) -> float:
        return self.seed_cash_usd + self.realised_pnl_usd + self.unrealised_pnl_usd


def start_session(
    *,
    session_id: str,
    user_id: str,
    seed_cash_usd: float,
    qualified_tier: Tier,
    started_on: date,
) -> PracticeSession:
    return PracticeSession(
        session_id=session_id,
        user_id=user_id,
        seed_cash_usd=seed_cash_usd,
        qualified_tier=qualified_tier,
        started_on=started_on,
    )


def place_trade(
    session: PracticeSession,
    trade: PracticeTrade,
) -> PracticeSession:
    """Append a paper trade.

    Pinned:
    - Session must be ACTIVE.
    - Trade's feature must be unlocked for the user's qualified tier.
    - Trade_id must be unique within the session.
    """
    if session.status is not SessionStatus.ACTIVE:
        raise ValueError(f"cannot trade in a {session.status.value} session")
    if not is_feature_unlocked(trade.feature, session.qualified_tier):
        raise ValueError(
            f"feature {trade.feature.value} requires tier "
            f"{min_tier_for(trade.feature).value} (you have {session.qualified_tier.value})"
        )
    if any(t.trade_id == trade.trade_id for t in session.trades):
        raise ValueError(f"duplicate trade_id {trade.trade_id}")
    return replace(session, trades=(*session.trades, trade))


def update_pnl(
    session: PracticeSession,
    *,
    realised: float | None = None,
    unrealised: float | None = None,
) -> PracticeSession:
    """Set realised / unrealised PnL; pass None to leave unchanged."""
    if session.status is SessionStatus.CONCLUDED:
        raise ValueError("cannot update PnL on a CONCLUDED session")
    new_r = realised if realised is not None else session.realised_pnl_usd
    new_u = unrealised if unrealised is not None else session.unrealised_pnl_usd
    if not -1e9 < new_r < 1e9:
        raise ValueError("realised PnL out of bounds")
    if not -1e9 < new_u < 1e9:
        raise ValueError("unrealised PnL out of bounds")
    return replace(session, realised_pnl_usd=new_r, unrealised_pnl_usd=new_u)


_LEGAL_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.ACTIVE: {SessionStatus.SUSPENDED, SessionStatus.CONCLUDED},
    SessionStatus.SUSPENDED: {SessionStatus.ACTIVE, SessionStatus.CONCLUDED},
    SessionStatus.CONCLUDED: set(),
}


def suspend_session(session: PracticeSession) -> PracticeSession:
    if session.status is not SessionStatus.ACTIVE:
        raise ValueError(f"suspend illegal from {session.status.value}")
    return replace(session, status=SessionStatus.SUSPENDED)


def resume_session(session: PracticeSession) -> PracticeSession:
    if session.status is not SessionStatus.SUSPENDED:
        raise ValueError(f"resume illegal from {session.status.value}")
    return replace(session, status=SessionStatus.ACTIVE)


def conclude_session(
    session: PracticeSession,
    *,
    on: date,
    promote_to_live: bool = False,
) -> PracticeSession:
    """Conclude a session. Optionally promote to live mode.

    Pinned: live promotion requires qualified_tier ≥ EXPERT (LIVE_PROMOTION
    feature). Soft-fail with explicit ValueError if the gate doesn't hold.
    """
    if session.status not in (
        SessionStatus.ACTIVE,
        SessionStatus.SUSPENDED,
    ):
        raise ValueError(f"conclude illegal from {session.status.value}")
    if on < session.started_on:
        raise ValueError("conclude on must be ≥ started_on")
    if promote_to_live and not is_feature_unlocked(
        PracticeFeature.LIVE_PROMOTION, session.qualified_tier
    ):
        raise ValueError("live promotion requires tier ≥ EXPERT")
    return replace(
        session,
        status=SessionStatus.CONCLUDED,
        concluded_on=on,
        promoted_to_live=promote_to_live,
    )


@dataclass(frozen=True)
class MilestoneReport:
    """Progression metrics emitted at session-close time."""

    user_id: str
    session_id: str
    n_trades: int
    final_balance_usd: float
    realised_pnl_usd: float
    return_pct: float
    unique_tickers: int


def session_milestones(session: PracticeSession) -> MilestoneReport:
    """Compute milestones from session state."""
    unique_tickers = len({t.ticker for t in session.trades})
    return_pct = (session.balance_usd() - session.seed_cash_usd) / session.seed_cash_usd
    return MilestoneReport(
        user_id=session.user_id,
        session_id=session.session_id,
        n_trades=len(session.trades),
        final_balance_usd=session.balance_usd(),
        realised_pnl_usd=session.realised_pnl_usd,
        return_pct=return_pct,
        unique_tickers=unique_tickers,
    )


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[SessionStatus, str] = {
    SessionStatus.ACTIVE: "🟢",
    SessionStatus.SUSPENDED: "⏸️",
    SessionStatus.CONCLUDED: "✅",
}


def render_session(session: PracticeSession) -> str:
    promo = " 🚀 promoted" if session.promoted_to_live else ""
    return (
        f"{_STATUS_EMOJI[session.status]} {session.session_id} "
        f"[{session.status.value}] {_mask(session.user_id)} "
        f"({session.qualified_tier.value}){promo}\n"
        f"  Seed ${session.seed_cash_usd:,.0f} | "
        f"trades {len(session.trades)} | "
        f"balance ${session.balance_usd():,.2f} | "
        f"realised P&L ${session.realised_pnl_usd:+,.2f}"
    )


def render_milestones(report: MilestoneReport) -> str:
    return (
        f"🏁 {report.session_id}: {report.n_trades} trades, "
        f"{report.unique_tickers} unique tickers, "
        f"final ${report.final_balance_usd:,.2f} "
        f"({report.return_pct * 100:+.2f}% return)"
    )
