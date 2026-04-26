"""Setup-typed stop-loss / take-profit profiles.

A flat 1% SL / 2% TP applied to every trade is a known leak: a
mean-reversion entry at a deep oversold reading needs a *tight* SL
(price has nowhere to go but slightly lower before our thesis is
invalidated) and a *modest* TP (we're harvesting noise, not chasing a
move). A breakout, by contrast, needs a *wider* SL (initial pullback
into the broken level is normal) and a *larger* TP (we're paid for
catching the leg). Same story for momentum vs. range trades.

This module owns those profiles in one place so:

  * Strategy code asks for (sl_pct, tp_pct) by setup_type instead of
    hard-coding values per call site.
  * Operators can tune the table without rebuilding prompts.
  * Backtests can replay historic decisions against alternative
    profiles to see what the same trades would have returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class SetupType(str, Enum):
    """The four setups we differentiate today.

    ``UNKNOWN`` is the back-compat fallback for any decision that
    pre-dates the schema bump or whose LLM output omits the field.
    """

    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    RANGE = "range"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SLTPProfile:
    """Stop-loss & take-profit distances expressed as fractions of entry."""

    stop_loss_pct: float
    take_profit_pct: float

    @property
    def reward_risk(self) -> float:
        if self.stop_loss_pct <= 0:
            return 0.0
        return self.take_profit_pct / self.stop_loss_pct


# Defaults tuned to match the existing config knobs (`stop_loss_pct=0.01`,
# `take_profit_pct=0.02`) so swapping in a profile-driven path is a no-op
# for the UNKNOWN/legacy case. Specific setups deviate from there.
_DEFAULT_PROFILES: Final[dict[SetupType, SLTPProfile]] = {
    SetupType.BREAKOUT: SLTPProfile(stop_loss_pct=0.012, take_profit_pct=0.030),
    SetupType.MEAN_REVERSION: SLTPProfile(stop_loss_pct=0.005, take_profit_pct=0.012),
    SetupType.MOMENTUM: SLTPProfile(stop_loss_pct=0.010, take_profit_pct=0.025),
    SetupType.RANGE: SLTPProfile(stop_loss_pct=0.006, take_profit_pct=0.010),
    SetupType.UNKNOWN: SLTPProfile(stop_loss_pct=0.010, take_profit_pct=0.020),
}


def coerce_setup_type(raw: str | None) -> SetupType:
    """Normalise an LLM-supplied string into a :class:`SetupType`.

    The LLM may return ``"breakout"``, ``"BREAKOUT"``, ``"mean reversion"``,
    or anything else. Unknown values fall back to ``UNKNOWN`` rather than
    raising — the strategy already validates the rest of the plan, and
    we'd rather size a trade conservatively than abort the whole cycle on
    a typo.
    """
    if not raw:
        return SetupType.UNKNOWN
    normalised = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return SetupType(normalised)
    except ValueError:
        return SetupType.UNKNOWN


def profile_for(setup_type: str | SetupType | None) -> SLTPProfile:
    """Return the SL/TP profile for a setup, falling back to UNKNOWN."""
    if isinstance(setup_type, str):
        setup_type = coerce_setup_type(setup_type)
    if setup_type is None:
        setup_type = SetupType.UNKNOWN
    return _DEFAULT_PROFILES[setup_type]


def derive_sl_tp(
    entry_price: float,
    setup_type: str | SetupType | None,
    side: str = "buy",
) -> tuple[float, float]:
    """Return (stop_loss_price, take_profit_price) for a long entry.

    Shorts are not currently supported (halal: no naked shorts) so the
    ``side`` argument exists only to make the asymmetry explicit at call
    sites and to fail loudly if someone does try to short later.
    """
    if side != "buy":
        raise NotImplementedError(
            f"derive_sl_tp only supports long entries (no shorts allowed by halal); got {side!r}"
        )
    profile = profile_for(setup_type)
    sl = entry_price * (1 - profile.stop_loss_pct)
    tp = entry_price * (1 + profile.take_profit_pct)
    return sl, tp
