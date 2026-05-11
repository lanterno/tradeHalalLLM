"""Halal ETF basket builder — Round-5 Wave 21.D.

User defines a basket of halal-compliant constituents with target
weights; the platform treats the basket as a single tradable position
+ auto-rebalances on schedule. Two structural pins:

1. **Continuous halal screen** — when a constituent's halal status
   changes, the basket re-runs the screen and proposes an action
   (HOLD / REWEIGHT / DIVEST).
2. **Drift-based rebalance** — when any constituent's actual weight
   drifts more than `drift_threshold` from target, the next scheduled
   rebalance is brought forward.

Pinned semantics:

- **Closed-set RebalanceCadence** — DAILY / WEEKLY / MONTHLY /
  QUARTERLY.
- **Closed-set ScreenAction** — HOLD / REWEIGHT / DIVEST. DIVEST is
  triggered when a constituent fails the halal screen post-listing.
- **Target weights must sum to 1.0** (±1e-6).
- **Min weight per constituent = 1%** by default — prevents dust.
- **Drift threshold default = 5pp** absolute deviation per constituent.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum


class RebalanceCadence(str, Enum):
    """Closed-set cadence ladder."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


_CADENCE_DAYS: dict[RebalanceCadence, int] = {
    RebalanceCadence.DAILY: 1,
    RebalanceCadence.WEEKLY: 7,
    RebalanceCadence.MONTHLY: 30,
    RebalanceCadence.QUARTERLY: 90,
}


class ScreenAction(str, Enum):
    """Closed-set screen action ladder."""

    HOLD = "hold"
    REWEIGHT = "reweight"
    DIVEST = "divest"


@dataclass(frozen=True)
class Constituent:
    """One basket member with its target weight."""

    ticker: str
    target_weight: float
    sector: str = "unknown"
    halal_compliant: bool = True

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if not 0.0 < self.target_weight <= 1.0:
            raise ValueError("target_weight must be in (0, 1]")
        if not self.sector or not self.sector.strip():
            raise ValueError("sector must be non-empty")


@dataclass(frozen=True)
class BasketDefinition:
    """A frozen halal-ETF basket definition."""

    basket_id: str
    name: str
    author_id: str
    constituents: tuple[Constituent, ...]
    cadence: RebalanceCadence
    drift_threshold: float = 0.05
    """Per-constituent absolute weight drift triggering rebalance."""
    min_weight: float = 0.01
    """Floor on constituent weight; below = dust + rejected."""
    created_on: date | None = None

    def __post_init__(self) -> None:
        if not self.basket_id or not self.basket_id.strip():
            raise ValueError("basket_id must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if len(self.name) > 120:
            raise ValueError("name must be ≤ 120 chars")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.constituents:
            raise ValueError("basket must have at least one constituent")
        # Unique tickers.
        tickers = [c.ticker for c in self.constituents]
        if len(set(tickers)) != len(tickers):
            raise ValueError("duplicate ticker in basket")
        # All target weights ≥ min_weight.
        if not 0.0 < self.min_weight <= 1.0:
            raise ValueError("min_weight must be in (0, 1]")
        for c in self.constituents:
            if c.target_weight < self.min_weight - 1e-12:
                raise ValueError(
                    f"{c.ticker} target_weight {c.target_weight:.4f} < "
                    f"min_weight {self.min_weight:.4f}"
                )
        # Targets sum to 1.0.
        s = sum(c.target_weight for c in self.constituents)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"target_weights must sum to 1.0; got {s:.6f}")
        if not 0.0 < self.drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be in (0, 1]")


def assert_basket_halal(basket: BasketDefinition) -> None:
    """All constituents must be halal_compliant at construction."""
    bad = [c.ticker for c in basket.constituents if not c.halal_compliant]
    if bad:
        raise ValueError(f"basket contains non-halal tickers at construction: {bad}")


# --- Drift detection ---------------------------------------------


@dataclass(frozen=True)
class DriftReport:
    """Output of `compute_drift`."""

    ticker: str
    target_weight: float
    actual_weight: float
    drift: float
    """actual − target; signed."""
    needs_rebalance: bool


def compute_drift(
    basket: BasketDefinition,
    actual_weights: Mapping[str, float],
) -> tuple[DriftReport, ...]:
    """Compute per-constituent drift vs target.

    Pinned: tickers in `actual_weights` not in the basket are ignored
    (the operator may have a residual position from a previous basket).
    """
    out: list[DriftReport] = []
    for c in basket.constituents:
        actual = float(actual_weights.get(c.ticker, 0.0))
        if actual < 0:
            raise ValueError(f"actual_weight for {c.ticker} cannot be negative")
        drift = actual - c.target_weight
        needs = abs(drift) > basket.drift_threshold
        out.append(
            DriftReport(
                ticker=c.ticker,
                target_weight=c.target_weight,
                actual_weight=actual,
                drift=drift,
                needs_rebalance=needs,
            )
        )
    return tuple(out)


def any_drift_exceeds(basket: BasketDefinition, actual_weights: Mapping[str, float]) -> bool:
    return any(d.needs_rebalance for d in compute_drift(basket, actual_weights))


# --- Continuous halal-screen ---------------------------------------


@dataclass(frozen=True)
class ScreenIssue:
    """Output of `screen_basket`."""

    ticker: str
    action: ScreenAction
    reason: str


def screen_basket(
    basket: BasketDefinition,
    *,
    is_ticker_halal: Callable[[str], bool],
) -> tuple[ScreenIssue, ...]:
    """Re-run the halal screen on each constituent.

    Pinned:
    - Tickers that fail → DIVEST.
    - Tickers that pass but had previously been marked non-halal in
      the basket definition → REWEIGHT (operator can opt to add the
      ticker back).
    - Otherwise HOLD (no issue emitted).
    """
    issues: list[ScreenIssue] = []
    for c in basket.constituents:
        currently_halal = is_ticker_halal(c.ticker)
        if not currently_halal:
            issues.append(
                ScreenIssue(
                    ticker=c.ticker,
                    action=ScreenAction.DIVEST,
                    reason="ticker failed halal screen after listing",
                )
            )
        elif not c.halal_compliant:
            issues.append(
                ScreenIssue(
                    ticker=c.ticker,
                    action=ScreenAction.REWEIGHT,
                    reason="ticker passed halal screen but was marked non-compliant",
                )
            )
    return tuple(issues)


def divest_failing(
    basket: BasketDefinition,
    *,
    is_ticker_halal: Callable[[str], bool],
) -> BasketDefinition:
    """Return a new basket with non-halal constituents removed and the
    remaining weights renormalised.

    Raises if every constituent fails the screen.
    """
    surviving = [c for c in basket.constituents if is_ticker_halal(c.ticker)]
    if not surviving:
        raise ValueError("all constituents failed halal screen")
    total = sum(c.target_weight for c in surviving)
    new_constituents = tuple(replace(c, target_weight=c.target_weight / total) for c in surviving)
    return replace(basket, constituents=new_constituents)


# --- Schedule ---------------------------------------------------


def next_rebalance(
    basket: BasketDefinition,
    *,
    last_rebalance: date,
    actual_weights: Mapping[str, float] | None = None,
) -> date:
    """Compute the next scheduled rebalance date.

    Pinned: if any constituent's drift exceeds `drift_threshold`, the
    rebalance is pulled forward to `last_rebalance + 1` day. Otherwise
    it falls on the standard cadence interval.
    """
    days = _CADENCE_DAYS[basket.cadence]
    standard = last_rebalance + timedelta(days=days)
    if actual_weights is not None and any_drift_exceeds(basket, actual_weights):
        urgent = last_rebalance + timedelta(days=1)
        return min(standard, urgent)
    return standard


# --- Render ---------------------------------------------------


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_ACTION_EMOJI: dict[ScreenAction, str] = {
    ScreenAction.HOLD: "✅",
    ScreenAction.REWEIGHT: "🔁",
    ScreenAction.DIVEST: "⛔",
}


def render_basket(basket: BasketDefinition) -> str:
    head = (
        f"🧺 {basket.basket_id}: {basket.name} ({basket.cadence.value}) "
        f"— author {_mask(basket.author_id)}, "
        f"{len(basket.constituents)} constituents"
    )
    lines = [head]
    for c in basket.constituents:
        flag = "" if c.halal_compliant else " [non-halal]"
        lines.append(f"  • {c.ticker} ({c.sector}): {c.target_weight * 100:.2f}%{flag}")
    return "\n".join(lines)


def render_drift(reports: Iterable[DriftReport]) -> str:
    rs = tuple(reports)
    if not rs:
        return "📊 No drift reports."
    lines = [f"📊 Drift across {len(rs)} constituent(s):"]
    for r in rs:
        marker = "⚠️" if r.needs_rebalance else "—"
        lines.append(
            f"  {marker} {r.ticker}: target={r.target_weight * 100:.2f}%, "
            f"actual={r.actual_weight * 100:.2f}%, "
            f"drift={r.drift * 100:+.2f}%"
        )
    return "\n".join(lines)


def render_screen(issues: Iterable[ScreenIssue]) -> str:
    its = tuple(issues)
    if not its:
        return "✅ Basket halal screen: clean."
    lines = [f"🔍 Basket halal screen: {len(its)} issue(s)"]
    for it in its:
        lines.append(f"  {_ACTION_EMOJI[it.action]} {it.ticker}: {it.action.value} — {it.reason}")
    return "\n".join(lines)
