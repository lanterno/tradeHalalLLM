"""Continuous re-screening on corporate-action / 8-K events.

Round-5 Wave 1.H primitive. The standard halal screen runs once at
universe-load time. When a screened company files an 8-K announcing
a debt issuance, large acquisition, or business-mix shift, its
ratios change overnight — but the universe cache keeps treating it
as compliant until the next refresh.

This module is the **screening rules engine** that the live event
feed (8-K subscription / earnings-release polling) calls into. It
is deliberately I/O-free + stateless: callers pass in the previous
verdict + the post-event snapshot, the engine returns a re-screen
verdict + a structured "what changed" payload. Persistence + alerts
live one layer up.

Pinned semantics:

- **Closed-set CorporateAction ladder.** Adding a new action is a
  code review change.
- **Closed-set ScreenStatus ladder.** PASSED / FLAGGED / FAILED.
- **`reassess` is pure** — no DB, no clock, no network. Deterministic
  for replay.
- **Stale-window pin.** Re-screens older than `stale_after_days` are
  rejected with `is_stale=True` so the operator knows to refresh
  the data feed rather than trust an old verdict.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class CorporateAction(str, Enum):
    """Closed-set actions that trigger a re-screen."""

    DEBT_ISSUANCE = "debt_issuance"
    DEBT_RETIREMENT = "debt_retirement"
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    DIVIDEND_DECLARATION = "dividend_declaration"
    EARNINGS_RELEASE = "earnings_release"
    SECTOR_RECLASSIFICATION = "sector_reclassification"
    BUSINESS_MIX_CHANGE = "business_mix_change"
    BANKRUPTCY_FILING = "bankruptcy_filing"
    GOING_PRIVATE = "going_private"


class ScreenStatus(str, Enum):
    """Closed-set re-screen outcome."""

    PASSED = "passed"
    FLAGGED = "flagged"
    FAILED = "failed"


@dataclass(frozen=True)
class ScreenSnapshot:
    """A point-in-time snapshot of the data the screener evaluates."""

    ticker: str
    debt_to_market_cap: float  # 0.30 = 30%
    interest_income_to_revenue: float
    liquid_assets_to_market_cap: float
    non_halal_revenue_pct: float  # 0.05 = 5%
    sector_is_halal: bool
    snapshot_date: date

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        for name, val in (
            ("debt_to_market_cap", self.debt_to_market_cap),
            ("interest_income_to_revenue", self.interest_income_to_revenue),
            ("liquid_assets_to_market_cap", self.liquid_assets_to_market_cap),
            ("non_halal_revenue_pct", self.non_halal_revenue_pct),
        ):
            if val < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class ScreenPolicy:
    """Operator-tunable thresholds — defaults match AAOIFI Standard 21."""

    debt_ratio_cap: float = 0.30
    interest_income_cap: float = 0.05
    liquid_assets_cap: float = 0.30
    non_halal_revenue_cap: float = 0.05
    flag_buffer: float = 0.02  # within this fraction of cap → FLAGGED
    stale_after_days: int = 7

    def __post_init__(self) -> None:
        for name, val in (
            ("debt_ratio_cap", self.debt_ratio_cap),
            ("interest_income_cap", self.interest_income_cap),
            ("liquid_assets_cap", self.liquid_assets_cap),
            ("non_halal_revenue_cap", self.non_halal_revenue_cap),
        ):
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.flag_buffer < 0 or self.flag_buffer >= min(
            self.debt_ratio_cap,
            self.interest_income_cap,
            self.liquid_assets_cap,
            self.non_halal_revenue_cap,
        ):
            raise ValueError("flag_buffer must be in [0, min_cap)")
        if self.stale_after_days <= 0:
            raise ValueError("stale_after_days must be positive")


@dataclass(frozen=True)
class CorporateEvent:
    """A single corporate-action event triggering re-screen."""

    ticker: str
    action: CorporateAction
    event_date: date
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")


@dataclass(frozen=True)
class ScreenChange:
    """What changed about a single metric across the re-screen."""

    metric: str
    previous: float
    current: float
    delta: float
    crossed_cap: bool

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric must be non-empty")


@dataclass(frozen=True)
class RescreenResult:
    """Result of running a re-screen — what status is, what changed."""

    ticker: str
    new_status: ScreenStatus
    previous_status: ScreenStatus
    changes: tuple[ScreenChange, ...]
    triggering_event: CorporateEvent
    is_stale: bool

    def status_flipped(self) -> bool:
        """True if the verdict changed across the re-screen."""
        return self.new_status is not self.previous_status

    def regression(self) -> bool:
        """True if the verdict moved towards FAILED (PASSED→FLAGGED→FAILED)."""
        order = {ScreenStatus.PASSED: 0, ScreenStatus.FLAGGED: 1, ScreenStatus.FAILED: 2}
        return order[self.new_status] > order[self.previous_status]


def _classify(snapshot: ScreenSnapshot, policy: ScreenPolicy) -> ScreenStatus:
    """Classify a snapshot against the policy."""
    if not snapshot.sector_is_halal:
        return ScreenStatus.FAILED

    metrics = (
        (snapshot.debt_to_market_cap, policy.debt_ratio_cap),
        (snapshot.interest_income_to_revenue, policy.interest_income_cap),
        (snapshot.liquid_assets_to_market_cap, policy.liquid_assets_cap),
        (snapshot.non_halal_revenue_pct, policy.non_halal_revenue_cap),
    )
    if any(val > cap for val, cap in metrics):
        return ScreenStatus.FAILED
    if any(val > cap - policy.flag_buffer for val, cap in metrics):
        return ScreenStatus.FLAGGED
    return ScreenStatus.PASSED


def _diff_snapshots(
    previous: ScreenSnapshot,
    current: ScreenSnapshot,
    policy: ScreenPolicy,
) -> tuple[ScreenChange, ...]:
    out: list[ScreenChange] = []
    triples = (
        ("debt_to_market_cap", previous.debt_to_market_cap, current.debt_to_market_cap, policy.debt_ratio_cap),
        (
            "interest_income_to_revenue",
            previous.interest_income_to_revenue,
            current.interest_income_to_revenue,
            policy.interest_income_cap,
        ),
        (
            "liquid_assets_to_market_cap",
            previous.liquid_assets_to_market_cap,
            current.liquid_assets_to_market_cap,
            policy.liquid_assets_cap,
        ),
        (
            "non_halal_revenue_pct",
            previous.non_halal_revenue_pct,
            current.non_halal_revenue_pct,
            policy.non_halal_revenue_cap,
        ),
    )
    for name, prev, curr, cap in triples:
        delta = curr - prev
        crossed = (prev <= cap < curr) or (curr <= cap < prev)
        if delta != 0 or crossed:
            out.append(
                ScreenChange(
                    metric=name,
                    previous=prev,
                    current=curr,
                    delta=delta,
                    crossed_cap=crossed,
                )
            )
    if previous.sector_is_halal != current.sector_is_halal:
        out.append(
            ScreenChange(
                metric="sector_is_halal",
                previous=1.0 if previous.sector_is_halal else 0.0,
                current=1.0 if current.sector_is_halal else 0.0,
                delta=(1.0 if current.sector_is_halal else 0.0)
                - (1.0 if previous.sector_is_halal else 0.0),
                crossed_cap=True,
            )
        )
    return tuple(out)


def reassess(
    previous: ScreenSnapshot,
    current: ScreenSnapshot,
    event: CorporateEvent,
    *,
    today: date,
    previous_status: ScreenStatus | None = None,
    policy: ScreenPolicy | None = None,
) -> RescreenResult:
    """Re-screen a candidate after a corporate action and return the result."""
    if previous.ticker != current.ticker or current.ticker != event.ticker:
        raise ValueError("ticker mismatch across previous / current / event")
    pol = policy if policy is not None else ScreenPolicy()
    prev_status = previous_status if previous_status is not None else _classify(previous, pol)
    new_status = _classify(current, pol)
    changes = _diff_snapshots(previous, current, pol)
    is_stale = (today - current.snapshot_date) > timedelta(days=pol.stale_after_days)
    return RescreenResult(
        ticker=event.ticker,
        new_status=new_status,
        previous_status=prev_status,
        changes=changes,
        triggering_event=event,
        is_stale=is_stale,
    )


def reassess_batch(
    pairs: Iterable[tuple[ScreenSnapshot, ScreenSnapshot, CorporateEvent]],
    *,
    today: date,
    policy: ScreenPolicy | None = None,
) -> tuple[RescreenResult, ...]:
    return tuple(reassess(prev, curr, ev, today=today, policy=policy) for prev, curr, ev in pairs)


def filter_regressions(results: Iterable[RescreenResult]) -> tuple[RescreenResult, ...]:
    """Return only the results whose status moved towards FAILED — operator alert list."""
    return tuple(r for r in results if r.regression())


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_result(result: RescreenResult) -> str:
    if result.regression():
        emoji = "⚠️"
    elif result.status_flipped():
        emoji = "ℹ️"
    else:
        emoji = "✅"
    stale = " [STALE DATA]" if result.is_stale else ""
    head = (
        f"{emoji} {result.ticker} {result.previous_status.value}→{result.new_status.value}"
        f" (event: {result.triggering_event.action.value}){stale}"
    )
    lines = [head]
    for ch in result.changes:
        marker = "‼" if ch.crossed_cap else "·"
        lines.append(
            f"  {marker} {ch.metric}: {ch.previous:.4f}→{ch.current:.4f} "
            f"(Δ{ch.delta:+.4f})"
        )
    return _scrub("\n".join(lines))
