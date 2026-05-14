"""Strategy leaderboard with opt-in privacy.

The roadmap pins Wave 3.H: "Aggregated (anonymised) leaderboard:
top 10 strategies by Sharpe this quarter, by win rate this month.
Lets users discover what's working and clone a public strategy
template. Privacy: opt-in only." This module is the **pure-Python
ranking + anonymisation engine** that the public leaderboard route
consumes.

Picked a focused engine over a "hand-roll a SQL query per route"
approach because (a) the privacy contract is the load-bearing
surface — opt-in is a per-strategy switch, *not* a per-user-account
switch; an account that opted out at signup but later opts in for
a single strategy must surface only that strategy and nothing else.
A single chokepoint that filters opt-in at the boundary is far
safer than a JOIN in 8 different routes that has to remember to
filter; (b) the ranking metrics (Sharpe / win rate / total return)
need consistent boundary semantics across windows (quarterly /
monthly / yearly) — pure functions of (entries, window, metric)
let us regression-pin those boundaries; (c) anonymisation needs
a minimum sample threshold (k-anonymity-like) — a leaderboard
with one entry isn't a leaderboard, it's a personal page; (d) a
clone-template surface needs to surface the strategy *config*
without the operator's identity — pinned via test that the
template never carries user_id / email.

Pinned semantics:
- **Opt-in is per-strategy, not per-user.** A user with two
  strategies can opt in to one and not the other; the engine
  surfaces only opted-in strategies. Default opt-in=False — the
  conservative default; surfacing without explicit consent is the
  privacy failure mode this guards against.
- **Minimum 5 entries to publish a leaderboard.** Below 5,
  individual entries become re-identifiable by elimination
  (especially when paired with the strategy_kind hint). The
  threshold is operator-configurable via `LeaderboardPolicy` but
  defaults to 5 — pinned via test that 4 entries return an empty
  leaderboard.
- **Display name is NEVER the user_id.** Each entry surfaces a
  `display_handle` chosen by the user (or auto-generated as
  `strategist_{stable_hash[:8]}`); user_id never appears in render
  output. Pinned via no-leak regression test.
- **Top-N takes ties via stable secondary sort.** When two
  strategies have the same Sharpe, the older strategy ranks
  higher (more battle-tested). Operators expect deterministic
  ordering across renders.
- **Render output never includes user_id / email / Stripe ID /
  position values.** Mirrors the no-secret patterns of Wave 3.B
  vault + Wave 3.F billing + Wave 3.G admin console.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class LeaderboardMetric(str, Enum):
    """The metric used to rank leaderboard entries.

    Pinned string values for JSON / DB stability. Adding a metric
    is a code review change.
    """

    SHARPE = "sharpe"
    WIN_RATE = "win_rate"
    TOTAL_RETURN_PCT = "total_return_pct"


class LeaderboardWindow(str, Enum):
    """Time window over which the metric is aggregated."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


_WINDOW_DURATIONS: dict[LeaderboardWindow, timedelta | None] = {
    LeaderboardWindow.MONTHLY: timedelta(days=30),
    LeaderboardWindow.QUARTERLY: timedelta(days=90),
    LeaderboardWindow.YEARLY: timedelta(days=365),
    LeaderboardWindow.ALL_TIME: None,
}


_MIN_K_ANONYMITY_DEFAULT = 5
_TOP_N_DEFAULT = 10


@dataclass(frozen=True)
class LeaderboardPolicy:
    """Operator-tunable leaderboard policy.

    `min_entries_to_publish` enforces the k-anonymity-like floor;
    `top_n` is the rendered cohort size; `min_sample_size` is the
    per-entry trade count threshold below which an entry is excluded
    (e.g. a strategy with 3 trades has too-noisy a Sharpe to rank).
    """

    min_entries_to_publish: int = _MIN_K_ANONYMITY_DEFAULT
    top_n: int = _TOP_N_DEFAULT
    min_sample_size: int = 10

    def __post_init__(self) -> None:
        if self.min_entries_to_publish < 2:
            raise ValueError("min_entries_to_publish must be >= 2")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.min_sample_size < 0:
            raise ValueError("min_sample_size must be non-negative")


DEFAULT_POLICY = LeaderboardPolicy()


@dataclass(frozen=True)
class StrategyEntry:
    """One strategy's leaderboard candidate row.

    The user_id is recorded for opt-in lookup + audit; the
    `display_handle` is what surfaces in render output. The
    privacy contract: user_id NEVER appears in rendered output.

    `strategy_kind` is a short tag (e.g. "momentum", "mean_reversion",
    "halal_dca") that groups strategies for the clone-template
    surface; pinned non-empty.
    """

    user_id: str
    strategy_id: str
    display_handle: str
    strategy_kind: str
    opt_in: bool
    created_at: datetime
    last_traded_at: datetime
    sharpe: float
    win_rate: float
    total_return_pct: float
    sample_size: int

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.strategy_id or not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty")
        if not self.display_handle or not self.display_handle.strip():
            raise ValueError("display_handle must be non-empty")
        if not self.strategy_kind or not self.strategy_kind.strip():
            raise ValueError("strategy_kind must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.last_traded_at.tzinfo is None:
            raise ValueError("last_traded_at must be timezone-aware")
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"win_rate {self.win_rate} must be in [0, 1]")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")


@dataclass(frozen=True)
class LeaderboardRow:
    """Anonymised leaderboard row for render.

    `user_id` is intentionally absent — only the display_handle
    appears. The strategy_id is preserved so the clone-template
    surface can fetch the public template by ID.
    """

    rank: int
    display_handle: str
    strategy_id: str
    strategy_kind: str
    metric: LeaderboardMetric
    metric_value: float
    sample_size: int


@dataclass(frozen=True)
class Leaderboard:
    """The full leaderboard view-model."""

    metric: LeaderboardMetric
    window: LeaderboardWindow
    generated_at: datetime
    rows: tuple[LeaderboardRow, ...]
    eligible_count: int
    excluded_below_min_sample: int
    suppressed_below_k_anonymity: bool

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")


def _entry_metric(entry: StrategyEntry, metric: LeaderboardMetric) -> float:
    if metric is LeaderboardMetric.SHARPE:
        return entry.sharpe
    if metric is LeaderboardMetric.WIN_RATE:
        return entry.win_rate
    if metric is LeaderboardMetric.TOTAL_RETURN_PCT:
        return entry.total_return_pct
    raise ValueError(f"unknown metric {metric!r}")


def _within_window(entry: StrategyEntry, *, now: datetime, window: LeaderboardWindow) -> bool:
    duration = _WINDOW_DURATIONS[window]
    if duration is None:  # ALL_TIME
        return True
    return entry.last_traded_at >= now - duration


def build_leaderboard(
    entries: Iterable[StrategyEntry],
    *,
    metric: LeaderboardMetric,
    window: LeaderboardWindow,
    now: datetime,
    policy: LeaderboardPolicy = DEFAULT_POLICY,
) -> Leaderboard:
    """Aggregate entries into a ranked, anonymised leaderboard.

    The pipeline:
    1. Filter to opt-in entries only.
    2. Filter to entries within the time window.
    3. Filter to entries with sample_size >= policy.min_sample_size.
    4. If fewer than policy.min_entries_to_publish remain, suppress
       the leaderboard (return empty rows + suppressed=True).
    5. Sort by metric descending; tie-break by older created_at first.
    6. Take top policy.top_n.
    7. Strip user_id at the LeaderboardRow boundary.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    entry_list = list(entries)

    opted_in = [e for e in entry_list if e.opt_in]
    in_window = [e for e in opted_in if _within_window(e, now=now, window=window)]

    sample_size_filtered = [e for e in in_window if e.sample_size >= policy.min_sample_size]
    excluded_below_min_sample = len(in_window) - len(sample_size_filtered)

    eligible_count = len(sample_size_filtered)

    if eligible_count < policy.min_entries_to_publish:
        return Leaderboard(
            metric=metric,
            window=window,
            generated_at=now,
            rows=(),
            eligible_count=eligible_count,
            excluded_below_min_sample=excluded_below_min_sample,
            suppressed_below_k_anonymity=True,
        )

    # Sort by metric descending; ties broken by older created_at first
    # (more battle-tested), then by strategy_id for full determinism.
    ranked = sorted(
        sample_size_filtered,
        key=lambda e: (-_entry_metric(e, metric), e.created_at, e.strategy_id),
    )

    top_n = ranked[: policy.top_n]
    rows = tuple(
        LeaderboardRow(
            rank=idx + 1,
            display_handle=entry.display_handle,
            strategy_id=entry.strategy_id,
            strategy_kind=entry.strategy_kind,
            metric=metric,
            metric_value=_entry_metric(entry, metric),
            sample_size=entry.sample_size,
        )
        for idx, entry in enumerate(top_n)
    )

    return Leaderboard(
        metric=metric,
        window=window,
        generated_at=now,
        rows=rows,
        eligible_count=eligible_count,
        excluded_below_min_sample=excluded_below_min_sample,
        suppressed_below_k_anonymity=False,
    )


def auto_handle(user_id: str) -> str:
    """Generate an auto display handle from user_id.

    Uses the leading 8 chars of SHA-256(user_id) so the handle is
    stable across renders but doesn't leak the user_id. Operators
    can override by setting an explicit `display_handle` on the
    entry.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return f"strategist_{digest[:8]}"


@dataclass(frozen=True)
class StrategyTemplate:
    """A clone-able strategy template.

    Carries the strategy *config* without the operator's identity.
    The privacy contract: never includes user_id / email / position
    values / capital amounts. Just the kind + display handle + the
    public-facing config blob (a frozen tuple of (key, value) pairs).
    """

    template_id: str
    display_handle: str
    strategy_kind: str
    config: tuple[tuple[str, str], ...]
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.template_id or not self.template_id.strip():
            raise ValueError("template_id must be non-empty")
        if not self.display_handle or not self.display_handle.strip():
            raise ValueError("display_handle must be non-empty")
        if not self.strategy_kind or not self.strategy_kind.strip():
            raise ValueError("strategy_kind must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        # Reject any private-looking config keys
        forbidden = {"user_id", "email", "broker_api_key", "stripe_id"}
        for key, _value in self.config:
            if key.lower() in forbidden:
                raise ValueError(f"config key {key!r} is forbidden in a public template")


def _format_metric(metric: LeaderboardMetric, value: float) -> str:
    if metric is LeaderboardMetric.SHARPE:
        return f"{value:.2f}"
    if metric is LeaderboardMetric.WIN_RATE:
        return f"{value * 100:.1f}%"
    if metric is LeaderboardMetric.TOTAL_RETURN_PCT:
        return f"{value:+.1f}%"
    return f"{value:.4f}"


def render_leaderboard(leaderboard: Leaderboard) -> str:
    """Format the leaderboard for ops display.

    Pinned no-secret-leak: never includes user_id / email / Stripe
    customer ID / dollar amounts / position values. Renders rank,
    display_handle, strategy_kind, metric value, sample size.
    """

    metric_label = leaderboard.metric.value.replace("_", " ")
    window_label = leaderboard.window.value
    header = (
        f"🏆 Leaderboard — {metric_label} ({window_label}) "
        f"@ {leaderboard.generated_at.date().isoformat()}"
    )
    lines = [header]

    if leaderboard.suppressed_below_k_anonymity:
        lines.append(
            f"  suppressed: only {leaderboard.eligible_count} eligible "
            f"(need >= {DEFAULT_POLICY.min_entries_to_publish} for privacy)"
        )
        return "\n".join(lines)

    if not leaderboard.rows:
        lines.append("  no entries")
        return "\n".join(lines)

    for row in leaderboard.rows:
        formatted = _format_metric(row.metric, row.metric_value)
        lines.append(
            f"  #{row.rank} {row.display_handle} ({row.strategy_kind}) — "
            f"{formatted} over {row.sample_size} trades"
        )
    return "\n".join(lines)


def render_template(template: StrategyTemplate) -> str:
    """Format a strategy template for ops / share-link display."""

    lines = [
        f"📋 {template.strategy_kind} template by {template.display_handle}",
        f"  template_id: {template.template_id}",
        f"  created: {template.created_at.date().isoformat()}",
    ]
    if template.config:
        lines.append("  config:")
        for key, value in template.config:
            lines.append(f"    {key}: {value}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "Leaderboard",
    "LeaderboardMetric",
    "LeaderboardPolicy",
    "LeaderboardRow",
    "LeaderboardWindow",
    "StrategyEntry",
    "StrategyTemplate",
    "auto_handle",
    "build_leaderboard",
    "render_leaderboard",
    "render_template",
]
