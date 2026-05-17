"""Sukuk laddering + roll strategy — Round-5 Wave 3.D.

A sukuk ladder is N tradable sukuk holdings staggered by maturity, so
that one matures every period and the proceeds roll into a new
N-period sukuk at the longest end. Smooths reinvestment risk + locks
in average yield across the curve.

This module ships the **construction + roll engine**. Pricing uses
`markets/sukuk_pricing.py`; persistence + broker dispatch live one
layer up.

Pinned semantics:

- **Closed-set RollPolicy ladder** (LONGEST_TENOR / EVEN_DISTRIBUTION).
- **`build_ladder` enforces tradability** — non-tradable sukuk types
  (Murabaha / Salam) are rejected at construction; the ladder only
  holds Ijara / Mudarabah / Musharakah / Wakalah / Istisna.
- **`maturity_today` is pure** — no clock side-effects; caller passes
  ``today``.
- **`roll(ladder, today, replacement)` returns a new tuple** — the
  data structure is immutable, mirroring the rest of the codebase.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from halal_trader.halal.aaoifi_standard_17 import (
    SukukType,
    is_tradable_in_secondary,
)


class RollPolicy(str, Enum):
    """Closed-set roll strategies."""

    LONGEST_TENOR = "longest_tenor"
    EVEN_DISTRIBUTION = "even_distribution"


@dataclass(frozen=True)
class LadderRung:
    """A single rung of the ladder — one sukuk holding."""

    issuer: str
    sukuk_type: SukukType
    face_value: float
    coupon_rate: float
    issue_date: date
    maturity_date: date

    def __post_init__(self) -> None:
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")
        if not is_tradable_in_secondary(self.sukuk_type):
            raise ValueError(
                f"{self.sukuk_type.value} is not tradable on secondary; "
                "ladders cannot hold pure-Murabaha or Salam"
            )
        if self.face_value <= 0:
            raise ValueError("face_value must be positive")
        if not 0.0 <= self.coupon_rate <= 0.50:
            raise ValueError("coupon_rate outside reasonable bounds")
        if self.maturity_date <= self.issue_date:
            raise ValueError("maturity_date must be after issue_date")

    def tenor_days(self) -> int:
        return (self.maturity_date - self.issue_date).days


@dataclass(frozen=True)
class Ladder:
    """A sukuk ladder — sorted tuple of rungs by maturity ascending."""

    rungs: tuple[LadderRung, ...]
    base_currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("ladder must have at least one rung")
        if not self.base_currency or len(self.base_currency) > 8:
            raise ValueError("base_currency must be a non-empty short code")
        mats = [r.maturity_date for r in self.rungs]
        if mats != sorted(mats):
            raise ValueError("rungs must be sorted by maturity_date")

    def total_face(self) -> float:
        return sum(r.face_value for r in self.rungs)

    def average_coupon(self) -> float:
        total_face = self.total_face()
        if total_face == 0:
            return 0.0
        return sum(r.face_value * r.coupon_rate for r in self.rungs) / total_face

    def matured_rungs(self, today: date) -> tuple[LadderRung, ...]:
        return tuple(r for r in self.rungs if r.maturity_date <= today)

    def active_rungs(self, today: date) -> tuple[LadderRung, ...]:
        return tuple(r for r in self.rungs if r.maturity_date > today)


def build_ladder(
    rungs: Iterable[LadderRung],
    *,
    base_currency: str = "USD",
) -> Ladder:
    """Sort + validate rungs into a ladder."""
    sorted_rungs = tuple(sorted(rungs, key=lambda r: r.maturity_date))
    return Ladder(rungs=sorted_rungs, base_currency=base_currency)


def even_distribution_target_tenors(
    n_rungs: int, max_tenor_years: int
) -> tuple[int, ...]:
    """Returns target tenors in years for an even-distribution ladder."""
    if n_rungs <= 0:
        raise ValueError("n_rungs must be positive")
    if max_tenor_years <= 0:
        raise ValueError("max_tenor_years must be positive")
    if n_rungs == 1:
        return (max_tenor_years,)
    step = max_tenor_years / n_rungs
    return tuple(round(step * (i + 1)) for i in range(n_rungs))


def roll(
    ladder: Ladder,
    *,
    today: date,
    replacement: LadderRung,
) -> Ladder:
    """Replace matured rungs with a single new long-end rung.

    A LONGEST_TENOR roll merges all matured face value into the
    replacement; EVEN_DISTRIBUTION is achieved by calling roll
    repeatedly with separate replacements at the appropriate tenors.
    """
    matured = ladder.matured_rungs(today)
    if not matured:
        return ladder

    # The replacement must mature later than every active rung.
    active = ladder.active_rungs(today)
    if active and replacement.maturity_date <= active[-1].maturity_date:
        raise ValueError(
            "replacement maturity must be after the longest active rung's maturity"
        )

    new_rungs = list(active) + [replacement]
    return build_ladder(new_rungs, base_currency=ladder.base_currency)


def render_ladder(ladder: Ladder, *, today: date | None = None) -> str:
    head = (
        f"📊 Ladder: {len(ladder.rungs)} rungs, "
        f"face={ladder.total_face():.2f} {ladder.base_currency}, "
        f"avg coupon={ladder.average_coupon() * 100:.2f}%"
    )
    lines = [head]
    for r in ladder.rungs:
        marker = ""
        if today is not None:
            marker = " [matured]" if r.maturity_date <= today else ""
        lines.append(
            f"  • {r.issuer} {r.sukuk_type.value} face={r.face_value:.2f} "
            f"coupon={r.coupon_rate * 100:.2f}% "
            f"matures {r.maturity_date.isoformat()}{marker}"
        )
    return "\n".join(lines)
