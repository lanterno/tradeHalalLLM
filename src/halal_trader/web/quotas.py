"""Per-user resource quotas.

The single-operator laptop bot has one global LLM cost cap
(`LLM_DAILY_USD_CAP`) and a single Binance / CoinGecko / Zoya rate
limit shared across the whole instance. The moment the bot grows a
multi-user surface (Wave 3.A user accounts + 3.B vault), one user's
runaway prompt loop or screener storm could exhaust the whole
instance's daily budget. This module is the per-user quota
accounting layer: tier-based daily limits, rolling 24-hour
windows, auto-reset on window expiry, and a clean exceeded
exception so downstream callers (the LLM router, the screener,
the executor) gate-keep their own spend without each re-deriving
the budget math.

Picked rolling 24-hour windows over calendar days because users
sit across time zones — a calendar-day window resets at the
operator's UTC midnight, which is mid-afternoon for a US-East
user and could wipe out their morning's cumulative usage in a way
that feels random. A rolling 24-hour window means each user's
budget refreshes 24 hours after they started spending, regardless
of clock time. Pinned via test.

Not in scope: the Wave 7.D research-API token-bucket limiter
handles *external* per-key rate limits at the public research API
boundary. This module handles *internal* per-user quotas — the
budget gate the LLM / screener / broker call sites consult before
spending. They share no state and have different semantics
(token bucket vs rolling-window accounting).

Pinned semantics:
- **Rolling 24-hour windows.** A `ResourceUsage.window_started_at`
  is set on first consume; consume calls within the window
  accumulate; calls past `window_started_at + 24h` reset the
  window with the consumed amount as the new starting usage.
- **WARNING band starts at 80%, EXCEEDED at 100%.** Both
  thresholds inclusive; pinned via test.
- **Negative consumption rejected.** Refunds are not a quota
  concern — the caller persists corrected usage rows directly
  (and operators audit the refund event separately). This guard
  prevents an integer-overflow / accidental-credit class of bug.
- **Tier limits are immutable.** `TierLimits` is a frozen
  dataclass and the tier→limits map handed to the tracker is a
  read-only mapping in spirit; the tracker copies inputs into a
  frozen registry at construction.
- **Render output never contains other users' usage.** Each
  rendered line is for one user only; operator audit dashboards
  iterate per user rather than concatenating across users.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Callable


class Tier(str, Enum):
    """Pricing / quota tier for a user.

    Pinned string values for DB / JSON serialisation stability.
    Operators add new tiers via code + review (rather than DB
    insert) so a tier drift can't silently change limit math.
    """

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ResourceKind(str, Enum):
    """Resource categories the quota engine tracks.

    Pinned string values for DB / JSON stability. The set is
    deliberately small — every new resource adds a column to the
    `ResourceUsage` row and a field to `TierLimits`, so adding one
    is a code review change, not a runtime config knob.
    """

    LLM_USD = "llm_usd"
    LLM_TOKENS = "llm_tokens"
    BROKER_API_CALLS = "broker_api_calls"
    SCREENER_API_CALLS = "screener_api_calls"
    CYCLE_RUNS = "cycle_runs"


class QuotaState(str, Enum):
    """Three-band state for a quota check.

    `OK` is below the warning threshold; `WARNING` is between
    warning and exceeded; `EXCEEDED` is at or above the limit.
    Operators key on the state values for routing (Telegram WARN
    notification on WARNING, halt on EXCEEDED).
    """

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


class QuotaExceededError(Exception):
    """Raised when consume() would push usage past the limit.

    Carries the resource + tier + remaining headroom so the caller's
    handler can render an actionable message.
    """

    def __init__(
        self,
        *,
        user_id: str,
        resource: ResourceKind,
        tier: Tier,
        used: float,
        limit: float,
    ) -> None:
        super().__init__(
            f"user {user_id!r} exceeded {resource.value} quota: "
            f"{used:.4f}/{limit:.4f} ({tier.value} tier)"
        )
        self.user_id = user_id
        self.resource = resource
        self.tier = tier
        self.used = used
        self.limit = limit


@dataclass(frozen=True)
class TierLimits:
    """Per-tier daily quotas.

    Pinned positive-or-zero invariants on every limit; zero is
    valid (used to disable a resource for a tier — e.g., the FREE
    tier has zero broker_api_calls because free-tier users only get
    paper trading via a shared simulator).
    """

    tier: Tier
    llm_usd_daily: float
    llm_tokens_daily: int
    broker_api_calls_daily: int
    screener_api_calls_daily: int
    cycle_runs_daily: int

    def __post_init__(self) -> None:
        if self.llm_usd_daily < 0:
            raise ValueError("llm_usd_daily must be non-negative")
        if self.llm_tokens_daily < 0:
            raise ValueError("llm_tokens_daily must be non-negative")
        if self.broker_api_calls_daily < 0:
            raise ValueError("broker_api_calls_daily must be non-negative")
        if self.screener_api_calls_daily < 0:
            raise ValueError("screener_api_calls_daily must be non-negative")
        if self.cycle_runs_daily < 0:
            raise ValueError("cycle_runs_daily must be non-negative")

    def for_resource(self, resource: ResourceKind) -> float:
        """Return the daily limit for the given resource."""

        if resource is ResourceKind.LLM_USD:
            return self.llm_usd_daily
        if resource is ResourceKind.LLM_TOKENS:
            return float(self.llm_tokens_daily)
        if resource is ResourceKind.BROKER_API_CALLS:
            return float(self.broker_api_calls_daily)
        if resource is ResourceKind.SCREENER_API_CALLS:
            return float(self.screener_api_calls_daily)
        if resource is ResourceKind.CYCLE_RUNS:
            return float(self.cycle_runs_daily)
        raise ValueError(f"unknown resource {resource!r}")


# Default per-tier limits. Operators override via the `limits_by_tier`
# arg to `QuotaTracker`. The numbers are documented best-guesses
# for a multi-user paper-trading deployment — production deployments
# should tune them to actual cost recovery.
DEFAULT_TIER_LIMITS: dict[Tier, TierLimits] = {
    Tier.FREE: TierLimits(
        tier=Tier.FREE,
        llm_usd_daily=0.50,
        llm_tokens_daily=200_000,
        broker_api_calls_daily=0,
        screener_api_calls_daily=200,
        cycle_runs_daily=24,
    ),
    Tier.PRO: TierLimits(
        tier=Tier.PRO,
        llm_usd_daily=10.00,
        llm_tokens_daily=2_000_000,
        broker_api_calls_daily=10_000,
        screener_api_calls_daily=2_000,
        cycle_runs_daily=288,  # one every 5 minutes
    ),
    Tier.ENTERPRISE: TierLimits(
        tier=Tier.ENTERPRISE,
        llm_usd_daily=100.00,
        llm_tokens_daily=20_000_000,
        broker_api_calls_daily=100_000,
        screener_api_calls_daily=20_000,
        cycle_runs_daily=1_440,  # one per minute
    ),
}


@dataclass(frozen=True)
class ResourceUsage:
    """Current usage for one user inside their rolling 24h window.

    `window_started_at` is set on first consume; subsequent consume
    calls within the 24h window add to the running totals; calls
    past the window roll the start forward and reset the totals.
    """

    user_id: str
    tier: Tier
    window_started_at: datetime
    llm_usd_used: float = 0.0
    llm_tokens_used: int = 0
    broker_api_calls_used: int = 0
    screener_api_calls_used: int = 0
    cycle_runs_used: int = 0

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if self.window_started_at.tzinfo is None:
            raise ValueError("window_started_at must be timezone-aware")
        if self.llm_usd_used < 0:
            raise ValueError("llm_usd_used must be non-negative")
        if self.llm_tokens_used < 0:
            raise ValueError("llm_tokens_used must be non-negative")
        if self.broker_api_calls_used < 0:
            raise ValueError("broker_api_calls_used must be non-negative")
        if self.screener_api_calls_used < 0:
            raise ValueError("screener_api_calls_used must be non-negative")
        if self.cycle_runs_used < 0:
            raise ValueError("cycle_runs_used must be non-negative")

    def used_for(self, resource: ResourceKind) -> float:
        """Return the current usage for the given resource."""

        if resource is ResourceKind.LLM_USD:
            return self.llm_usd_used
        if resource is ResourceKind.LLM_TOKENS:
            return float(self.llm_tokens_used)
        if resource is ResourceKind.BROKER_API_CALLS:
            return float(self.broker_api_calls_used)
        if resource is ResourceKind.SCREENER_API_CALLS:
            return float(self.screener_api_calls_used)
        if resource is ResourceKind.CYCLE_RUNS:
            return float(self.cycle_runs_used)
        raise ValueError(f"unknown resource {resource!r}")


@dataclass(frozen=True)
class QuotaCheckResult:
    """A read-only snapshot of one user's quota state for one resource."""

    user_id: str
    resource: ResourceKind
    tier: Tier
    used: float
    limit: float
    remaining: float
    pct_used: float
    state: QuotaState
    window_started_at: datetime
    warnings: tuple[str, ...] = field(default_factory=tuple)


_WINDOW_HOURS = 24
_WARNING_THRESHOLD_PCT = 80.0


def _classify(used: float, limit: float) -> tuple[QuotaState, float]:
    """Return (state, pct_used) per the three-band ladder.

    Pinned: WARNING and EXCEEDED bands are inclusive at threshold —
    80% triggers WARNING, 100% triggers EXCEEDED. Zero-limit
    resources (FREE tier broker_api_calls) immediately classify
    as EXCEEDED on any non-zero use; zero-limit + zero-use lands
    OK with pct_used = 0 by definition.
    """

    if limit <= 0:
        if used <= 0:
            return QuotaState.OK, 0.0
        return QuotaState.EXCEEDED, 100.0
    pct = (used / limit) * 100.0
    if pct >= 100.0:
        return QuotaState.EXCEEDED, pct
    if pct >= _WARNING_THRESHOLD_PCT:
        return QuotaState.WARNING, pct
    return QuotaState.OK, pct


class QuotaTracker:
    """Stateless quota math + rolling-window accounting.

    Construction takes a per-tier limits map (defaults to
    `DEFAULT_TIER_LIMITS`) and an optional `now_fn` for
    deterministic tests. The tracker is stateless — usage rows
    flow through it (caller persists; tracker computes); a future
    DB-backed adapter would simply pass `ResourceUsage` rows from
    Postgres.
    """

    def __init__(
        self,
        *,
        limits_by_tier: Mapping[Tier, TierLimits] | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._limits = dict(limits_by_tier or DEFAULT_TIER_LIMITS)
        for tier in Tier:
            if tier not in self._limits:
                raise ValueError(f"limits_by_tier missing tier {tier.value!r}")
        self._now_fn = now_fn

    def limits_for(self, tier: Tier) -> TierLimits:
        return self._limits[tier]

    def _maybe_reset_window(self, usage: ResourceUsage) -> ResourceUsage:
        """Roll the window forward if 24h has elapsed."""

        now = self._now_fn()
        if now - usage.window_started_at >= timedelta(hours=_WINDOW_HOURS):
            return ResourceUsage(
                user_id=usage.user_id,
                tier=usage.tier,
                window_started_at=now,
            )
        return usage

    def check(
        self,
        usage: ResourceUsage,
        *,
        resource: ResourceKind,
        requested: float = 0.0,
    ) -> QuotaCheckResult:
        """Read-only quota check for a hypothetical (or zero) request.

        With `requested=0` (the default) returns the user's current
        state. With a positive `requested`, returns the state that
        *would* result from that consume — useful for "can I afford
        this LLM call?" pre-checks without committing usage.

        Auto-resets the window in the returned snapshot if 24h has
        elapsed; the *input* usage row is not mutated.
        """

        if requested < 0:
            raise ValueError("requested must be non-negative")
        rolled = self._maybe_reset_window(usage)
        used = rolled.used_for(resource) + requested
        limits = self.limits_for(rolled.tier)
        limit = limits.for_resource(resource)
        state, pct = _classify(used, limit)
        remaining = max(limit - used, 0.0)
        warnings: list[str] = []
        if state is QuotaState.WARNING:
            warnings.append(f"{resource.value} at {pct:.1f}% of {rolled.tier.value} tier limit")
        elif state is QuotaState.EXCEEDED:
            warnings.append(f"{resource.value} EXCEEDED ({used:.4f} / {limit:.4f})")
        return QuotaCheckResult(
            user_id=rolled.user_id,
            resource=resource,
            tier=rolled.tier,
            used=used,
            limit=limit,
            remaining=remaining,
            pct_used=pct,
            state=state,
            window_started_at=rolled.window_started_at,
            warnings=tuple(warnings),
        )

    def consume(
        self,
        usage: ResourceUsage,
        *,
        resource: ResourceKind,
        amount: float,
    ) -> ResourceUsage:
        """Consume `amount` of `resource` and return the new usage row.

        Raises `QuotaExceededError` if the consume would push usage
        past the limit. The caller persists the returned row.

        Auto-resets the window if 24h has elapsed since the row's
        `window_started_at`. The reset window starts with the just-
        consumed amount so the user gets credited for the new
        window's first call.
        """

        if amount < 0:
            raise ValueError("amount must be non-negative")
        rolled = self._maybe_reset_window(usage)
        limits = self.limits_for(rolled.tier)
        limit = limits.for_resource(resource)
        used_before = rolled.used_for(resource)
        used_after = used_before + amount
        if used_after > limit:
            raise QuotaExceededError(
                user_id=rolled.user_id,
                resource=resource,
                tier=rolled.tier,
                used=used_after,
                limit=limit,
            )
        return _apply_amount(rolled, resource, amount)

    def remaining(self, usage: ResourceUsage, *, resource: ResourceKind) -> float:
        """Convenience: remaining quota for the resource."""

        return self.check(usage, resource=resource).remaining


def _apply_amount(usage: ResourceUsage, resource: ResourceKind, amount: float) -> ResourceUsage:
    """Return a new ResourceUsage with `amount` added to the right field.

    Token / call counts are integer-typed at the dataclass level so
    we coerce the amount to int via floor — pinned via test that a
    fractional token count rounds down (consuming "half a token"
    consumes zero tokens so the user gets a small benefit-of-the-
    doubt). USD is float-typed so it's added directly.
    """

    if resource is ResourceKind.LLM_USD:
        return ResourceUsage(
            user_id=usage.user_id,
            tier=usage.tier,
            window_started_at=usage.window_started_at,
            llm_usd_used=usage.llm_usd_used + amount,
            llm_tokens_used=usage.llm_tokens_used,
            broker_api_calls_used=usage.broker_api_calls_used,
            screener_api_calls_used=usage.screener_api_calls_used,
            cycle_runs_used=usage.cycle_runs_used,
        )
    if resource is ResourceKind.LLM_TOKENS:
        return ResourceUsage(
            user_id=usage.user_id,
            tier=usage.tier,
            window_started_at=usage.window_started_at,
            llm_usd_used=usage.llm_usd_used,
            llm_tokens_used=usage.llm_tokens_used + int(amount),
            broker_api_calls_used=usage.broker_api_calls_used,
            screener_api_calls_used=usage.screener_api_calls_used,
            cycle_runs_used=usage.cycle_runs_used,
        )
    if resource is ResourceKind.BROKER_API_CALLS:
        return ResourceUsage(
            user_id=usage.user_id,
            tier=usage.tier,
            window_started_at=usage.window_started_at,
            llm_usd_used=usage.llm_usd_used,
            llm_tokens_used=usage.llm_tokens_used,
            broker_api_calls_used=usage.broker_api_calls_used + int(amount),
            screener_api_calls_used=usage.screener_api_calls_used,
            cycle_runs_used=usage.cycle_runs_used,
        )
    if resource is ResourceKind.SCREENER_API_CALLS:
        return ResourceUsage(
            user_id=usage.user_id,
            tier=usage.tier,
            window_started_at=usage.window_started_at,
            llm_usd_used=usage.llm_usd_used,
            llm_tokens_used=usage.llm_tokens_used,
            broker_api_calls_used=usage.broker_api_calls_used,
            screener_api_calls_used=usage.screener_api_calls_used + int(amount),
            cycle_runs_used=usage.cycle_runs_used,
        )
    if resource is ResourceKind.CYCLE_RUNS:
        return ResourceUsage(
            user_id=usage.user_id,
            tier=usage.tier,
            window_started_at=usage.window_started_at,
            llm_usd_used=usage.llm_usd_used,
            llm_tokens_used=usage.llm_tokens_used,
            broker_api_calls_used=usage.broker_api_calls_used,
            screener_api_calls_used=usage.screener_api_calls_used,
            cycle_runs_used=usage.cycle_runs_used + int(amount),
        )
    raise ValueError(f"unknown resource {resource!r}")


_STATE_EMOJI: dict[QuotaState, str] = {
    QuotaState.OK: "✅",
    QuotaState.WARNING: "⚠️",
    QuotaState.EXCEEDED: "🚫",
}


def render_quota_check(result: QuotaCheckResult) -> str:
    """Render-safe one-user quota summary."""

    emoji = _STATE_EMOJI[result.state]
    if result.resource is ResourceKind.LLM_USD:
        used = f"${result.used:.4f}"
        limit = f"${result.limit:.2f}"
        remaining = f"${result.remaining:.4f}"
    else:
        used = f"{result.used:.0f}"
        limit = f"{result.limit:.0f}"
        remaining = f"{result.remaining:.0f}"
    lines = [
        f"{emoji} {result.user_id} ({result.tier.value}) {result.resource.value}: "
        f"{used}/{limit} ({result.pct_used:.1f}%) — {result.state.value.upper()}"
    ]
    lines.append(f"  remaining: {remaining}")
    lines.append(f"  window started: {result.window_started_at.isoformat()}")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_TIER_LIMITS",
    "QuotaCheckResult",
    "QuotaExceededError",
    "QuotaState",
    "QuotaTracker",
    "ResourceKind",
    "ResourceUsage",
    "Tier",
    "TierLimits",
    "render_quota_check",
]
