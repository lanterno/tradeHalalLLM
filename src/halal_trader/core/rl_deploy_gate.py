"""Safe RL deployment gate — Round-5 Wave 9.G.

Before an RL agent ships to live trading, this gate enforces the
shadow-mode performance contract: the candidate agent must beat the
production baseline by `min_sharpe_delta` (default 3.0) over a
`min_shadow_days` window (default 90), with no catastrophic-drawdown
breaches.

Pinned semantics:

- **Closed-set GateStatus FSM** — SHADOW_RUNNING → ELIGIBLE → PROMOTED,
  with REJECTED as alternate terminal.
- **Closed-set RejectionReason ladder** — INSUFFICIENT_DAYS /
  SHARPE_DELTA_TOO_LOW / DRAWDOWN_BREACH / TRADE_COUNT_TOO_LOW /
  STILL_NEGATIVE.
- **Required shadow_days = 90 by default**. Operator-tunable.
- **Min sample size = 200 trades** by default.
- **Max allowed drawdown = 25%**; agents that breach are rejected
  regardless of Sharpe.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — agent IDs masked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class GateStatus(str, Enum):
    """Closed-set RL-gate FSM ladder."""

    SHADOW_RUNNING = "shadow_running"
    ELIGIBLE = "eligible"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    """Closed-set rejection-reason ladder."""

    INSUFFICIENT_DAYS = "insufficient_days"
    SHARPE_DELTA_TOO_LOW = "sharpe_delta_too_low"
    DRAWDOWN_BREACH = "drawdown_breach"
    TRADE_COUNT_TOO_LOW = "trade_count_too_low"
    STILL_NEGATIVE = "still_negative"


@dataclass(frozen=True)
class GatePolicy:
    """Operator-tunable RL-gate policy."""

    min_shadow_days: int = 90
    min_sharpe_delta: float = 3.0
    """Candidate's Sharpe must beat baseline by ≥ this."""
    max_drawdown_pct: float = 0.25
    """Candidate's realised max drawdown must be ≤ this."""
    min_trade_count: int = 200
    require_positive_sharpe: bool = True
    """If True, candidate's *own* Sharpe must be > 0 in addition to
    the delta."""

    def __post_init__(self) -> None:
        if self.min_shadow_days <= 0:
            raise ValueError("min_shadow_days must be positive")
        if self.min_sharpe_delta <= 0:
            raise ValueError("min_sharpe_delta must be positive")
        if not 0.0 < self.max_drawdown_pct <= 1.0:
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        if self.min_trade_count <= 0:
            raise ValueError("min_trade_count must be positive")


@dataclass(frozen=True)
class ShadowMetrics:
    """Realised shadow-mode metrics for the candidate agent."""

    candidate_sharpe: float
    baseline_sharpe: float
    candidate_max_drawdown_pct: float
    n_trades: int
    shadow_started_on: date
    shadow_last_active_on: date

    def __post_init__(self) -> None:
        if not -5.0 <= self.candidate_sharpe <= 10.0:
            raise ValueError("candidate_sharpe outside reasonable bounds")
        if not -5.0 <= self.baseline_sharpe <= 10.0:
            raise ValueError("baseline_sharpe outside reasonable bounds")
        if not 0.0 <= self.candidate_max_drawdown_pct <= 1.0:
            raise ValueError("candidate_max_drawdown_pct must be in [0, 1]")
        if self.n_trades < 0:
            raise ValueError("n_trades must be ≥ 0")
        if self.shadow_last_active_on < self.shadow_started_on:
            raise ValueError("shadow_last_active_on must be ≥ shadow_started_on")

    def shadow_days(self) -> int:
        return (self.shadow_last_active_on - self.shadow_started_on).days

    def sharpe_delta(self) -> float:
        return self.candidate_sharpe - self.baseline_sharpe


@dataclass(frozen=True)
class GateRecord:
    """Persistent state of one RL-gate evaluation."""

    candidate_id: str
    baseline_id: str
    metrics: ShadowMetrics
    status: GateStatus = GateStatus.SHADOW_RUNNING
    rejection_reasons: tuple[RejectionReason, ...] = ()
    promoted_on: date | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.baseline_id or not self.baseline_id.strip():
            raise ValueError("baseline_id must be non-empty")
        if self.candidate_id == self.baseline_id:
            raise ValueError("candidate and baseline must differ")
        if self.status is GateStatus.REJECTED and not self.rejection_reasons:
            raise ValueError("REJECTED requires ≥ 1 rejection_reason")
        if self.status is not GateStatus.REJECTED and self.rejection_reasons:
            raise ValueError("rejection_reasons can only be set on REJECTED records")
        if self.status is GateStatus.PROMOTED and self.promoted_on is None:
            raise ValueError("PROMOTED requires promoted_on")
        if self.promoted_on is not None and self.promoted_on < self.metrics.shadow_last_active_on:
            raise ValueError("promoted_on must be ≥ shadow_last_active_on")


def evaluate(
    candidate_id: str,
    baseline_id: str,
    metrics: ShadowMetrics,
    *,
    policy: GatePolicy | None = None,
) -> GateRecord:
    """Apply the policy and emit a `GateRecord` in ELIGIBLE or REJECTED state.

    Pinned: this function does NOT promote — promotion is a separate
    `mark_promoted` call after the operator confirms.
    """
    pol = policy if policy is not None else GatePolicy()
    reasons: list[RejectionReason] = []
    if metrics.shadow_days() < pol.min_shadow_days:
        reasons.append(RejectionReason.INSUFFICIENT_DAYS)
    if metrics.n_trades < pol.min_trade_count:
        reasons.append(RejectionReason.TRADE_COUNT_TOO_LOW)
    if metrics.candidate_max_drawdown_pct > pol.max_drawdown_pct + 1e-9:
        reasons.append(RejectionReason.DRAWDOWN_BREACH)
    if metrics.sharpe_delta() < pol.min_sharpe_delta - 1e-9:
        reasons.append(RejectionReason.SHARPE_DELTA_TOO_LOW)
    if pol.require_positive_sharpe and metrics.candidate_sharpe <= 0:
        reasons.append(RejectionReason.STILL_NEGATIVE)
    if reasons:
        return GateRecord(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            metrics=metrics,
            status=GateStatus.REJECTED,
            rejection_reasons=tuple(reasons),
        )
    return GateRecord(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        metrics=metrics,
        status=GateStatus.ELIGIBLE,
    )


_LEGAL_TRANSITIONS: dict[GateStatus, set[GateStatus]] = {
    GateStatus.SHADOW_RUNNING: {GateStatus.ELIGIBLE, GateStatus.REJECTED},
    GateStatus.ELIGIBLE: {GateStatus.PROMOTED, GateStatus.REJECTED},
    GateStatus.PROMOTED: set(),
    GateStatus.REJECTED: set(),
}


def mark_promoted(record: GateRecord, *, on: date) -> GateRecord:
    """ELIGIBLE → PROMOTED. Operator-pulled trigger after review."""
    if record.status is not GateStatus.ELIGIBLE:
        raise ValueError(f"mark_promoted illegal from {record.status.value}")
    if on < record.metrics.shadow_last_active_on:
        raise ValueError("promoted_on cannot precede shadow_last_active_on")
    return replace(record, status=GateStatus.PROMOTED, promoted_on=on)


def mark_rejected_late(record: GateRecord, *, reasons: Iterable[RejectionReason]) -> GateRecord:
    """Reject an ELIGIBLE record after a post-eligibility review.

    Used when the operator catches a soft-fail after `evaluate` passed
    (e.g. ancillary halal-compliance concern). PROMOTED + REJECTED are
    terminal.
    """
    if record.status not in (GateStatus.ELIGIBLE, GateStatus.SHADOW_RUNNING):
        raise ValueError(f"mark_rejected_late illegal from {record.status.value}")
    reasons_t = tuple(reasons)
    if not reasons_t:
        raise ValueError("must supply ≥ 1 rejection_reason")
    return replace(record, status=GateStatus.REJECTED, rejection_reasons=reasons_t)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_STATUS_EMOJI: dict[GateStatus, str] = {
    GateStatus.SHADOW_RUNNING: "🌒",
    GateStatus.ELIGIBLE: "🟢",
    GateStatus.PROMOTED: "🚀",
    GateStatus.REJECTED: "❌",
}


def render_record(record: GateRecord) -> str:
    m = record.metrics
    head = (
        f"{_STATUS_EMOJI[record.status]} {_mask(record.candidate_id)} "
        f"vs {_mask(record.baseline_id)} [{record.status.value}]\n"
        f"  Shadow: {m.shadow_days()}d, {m.n_trades} trades | "
        f"sharpe Δ={m.sharpe_delta():+.2f} "
        f"(cand={m.candidate_sharpe:+.2f}, base={m.baseline_sharpe:+.2f}) | "
        f"max DD={m.candidate_max_drawdown_pct * 100:.2f}%"
    )
    if record.rejection_reasons:
        head += "\n  Rejections:"
        for r in record.rejection_reasons:
            head += f"\n    • {r.value}"
    if record.promoted_on is not None:
        head += f"\n  Promoted on {record.promoted_on.isoformat()}"
    return head
