"""Online concept-drift detection.

Today the retrainer fires every N closed trades regardless of whether the
market regime is stable or shifting. That is fine when nothing changes,
but it's late-firing when a regime breaks (we keep predicting with a
stale model) and wasteful when nothing has actually moved.

This module gives the rest of the system a small, dependency-free drift
signal:

* :class:`PageHinkleyDetector` — lightweight, mean-shift detector based on
  the Page–Hinkley test. Cheap to keep online.
* :class:`AdwinLiteDetector` — variant of ADWIN: maintain a sliding window
  of recent observations and split-test the most recent half against the
  older half via a Welch-t-style heuristic.
* :class:`DriftMonitor` — convenience wrapper combining both: ``observe``
  one residual / win-rate signal per step, ``state`` returns ``"stable"``,
  ``"drift"``, or ``"warming_up"``.

Plug into the live pipeline by feeding either:

* per-trade residual (predicted_pnl - actual_pnl), or
* rolling 1/0 win indicator after each closed trade.

Either signal exposes regime drift sooner than waiting for the cumulative
P&L curve to bend.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

DriftState = Literal["warming_up", "stable", "drift"]


@dataclass
class PageHinkleyDetector:
    """Page–Hinkley one-sided change detector.

    ``delta`` is a "tolerance" — drift only registers when the
    cumulative deviation exceeds the running minimum by ``threshold``.
    ``alpha`` smooths the running mean (set to 1 for a flat mean).

    Tuned to be sensitive but not jumpy at our cycle cadence; default
    tunings are starting points — wire to live data and adjust.
    """

    threshold: float = 50.0
    delta: float = 0.005
    alpha: float = 0.999
    n: int = 0
    mean: float = 0.0
    cumulative: float = 0.0
    minimum: float = 0.0
    last_drift_at: int | None = None

    def observe(self, value: float) -> bool:
        """Feed one observation. Returns True the step drift is detected."""
        self.n += 1
        # incremental mean
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value if self.n > 1 else value
        if self.n <= 1:
            self.cumulative = 0.0
            self.minimum = 0.0
            return False
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        if (self.cumulative - self.minimum) > self.threshold:
            self.last_drift_at = self.n
            # reset so we don't fire on every subsequent obs
            self.cumulative = 0.0
            self.minimum = 0.0
            return True
        return False

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.last_drift_at = None


@dataclass
class AdwinLiteDetector:
    """A small ADWIN-style window-split detector.

    Keeps a fixed-size deque; when the recent half's mean diverges from
    the older half's by more than ``z`` standard errors, drift fires.

    Not as sophisticated as the original ADWIN (which tries every cut
    point) but markedly cheaper and good enough for a one-stream
    sanity check.
    """

    window: int = 64
    z: float = 2.5
    min_obs: int = 24
    _buf: deque[float] = field(default_factory=deque)
    last_drift_at: int | None = None
    n: int = 0

    def observe(self, value: float) -> bool:
        self.n += 1
        self._buf.append(value)
        while len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) < self.min_obs:
            return False
        half = len(self._buf) // 2
        old = list(self._buf)[:half]
        new = list(self._buf)[half:]
        mu_old = _mean(old)
        mu_new = _mean(new)
        var_old = _var(old, mu_old)
        var_new = _var(new, mu_new)
        # Welch-style standard error
        se = math.sqrt(var_old / max(1, len(old)) + var_new / max(1, len(new)))
        if se == 0:
            return False
        score = abs(mu_new - mu_old) / se
        if score > self.z:
            self.last_drift_at = self.n
            self._buf.clear()
            return True
        return False

    def reset(self) -> None:
        self._buf.clear()
        self.last_drift_at = None
        self.n = 0


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _var(xs: list[float], mu: float) -> float:
    if len(xs) < 2:
        return 0.0
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


@dataclass
class DriftMonitor:
    """Combine PH + ADWIN-lite into one user-facing detector.

    Drift fires when either underlying detector trips on the same step;
    this is intentionally OR-style — false negatives in one detector
    are corrected by the other. After a drift event the monitor enters
    a short cooldown so a single regime change doesn't fire repeatedly.
    """

    page_hinkley: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)
    adwin: AdwinLiteDetector = field(default_factory=AdwinLiteDetector)
    cooldown: int = 5
    n: int = 0
    last_drift_at: int | None = None
    drift_count: int = 0

    @property
    def state(self) -> DriftState:
        if self.n < self.adwin.min_obs:
            return "warming_up"
        if self.last_drift_at is not None and (self.n - self.last_drift_at) < self.cooldown:
            return "drift"
        return "stable"

    def observe(self, value: float) -> bool:
        self.n += 1
        ph_drift = self.page_hinkley.observe(value)
        adwin_drift = self.adwin.observe(value)
        if ph_drift or adwin_drift:
            self.last_drift_at = self.n
            self.drift_count += 1
            return True
        return False

    def reset(self) -> None:
        self.page_hinkley.reset()
        self.adwin.reset()
        self.n = 0
        self.last_drift_at = None
        self.drift_count = 0


# ── Risk policy hook ──────────────────────────────────────────────


@dataclass(frozen=True)
class DriftRiskPolicy:
    """Maps a DriftMonitor state to a sizing multiplier + SL tightener.

    Pure data — easy to swap. The cycle / risk engine reads this on every
    cycle to compute ``effective_max_position_pct`` and ``effective_sl_pct``.
    """

    drift_size_multiplier: float = 0.5
    drift_sl_tighten: float = 0.7  # multiply current SL distance by this
    warming_up_size_multiplier: float = 0.75

    def size_multiplier(self, state: DriftState) -> float:
        if state == "drift":
            return self.drift_size_multiplier
        if state == "warming_up":
            return self.warming_up_size_multiplier
        return 1.0

    def sl_multiplier(self, state: DriftState) -> float:
        if state == "drift":
            return self.drift_sl_tighten
        return 1.0
