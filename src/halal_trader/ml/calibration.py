"""Confidence calibration — map raw LLM/model confidence to empirical win-rate.

The strategy LLM emits a 0..1 confidence score and our sizing engine
treats it as if it meant "probability of profitable trade". It almost
never does — language-model-emitted confidences are notoriously
miscalibrated, and the bias is rarely zero (most over-confident, some
sandbagging on certain setup types).

This module fits a calibrator from past trades:

    fit_isotonic(samples) -> CalibrationCurve
    fit_platt(samples) -> CalibrationCurve

Both produce the same :class:`CalibrationCurve` interface, so callers
can swap methods without changing downstream code. The curve maps a raw
confidence to a calibrated win-probability:

    p_calibrated = curve.predict(p_raw)

…and the sizing engine uses ``calibrated`` (not raw) as its scaling.

Why two methods?
* Platt is parametric (one sigmoid) — robust on small samples, smooth.
* Isotonic is non-parametric — better when the bias has a non-monotonic
  shape, but needs more data to behave.

The default policy: try isotonic when ``n >= 200``, fall back to Platt
otherwise. Below ``n_min`` we return the identity curve (no calibration)
rather than make something up.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# ── Sample container ──────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationSample:
    """One closed trade contributing to the calibration fit."""

    raw_confidence: float
    win: bool


# ── Curve interface ───────────────────────────────────────────────


@dataclass
class CalibrationCurve:
    """Piecewise-linear lookup from raw → calibrated confidence.

    Stored as monotone (raw, calibrated) anchor points; predictions
    interpolate linearly between adjacent anchors. Identity is just
    the anchors [(0,0), (1,1)].
    """

    anchors: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    method: str = "identity"
    n_samples: int = 0

    @classmethod
    def identity(cls) -> "CalibrationCurve":
        return cls()

    def predict(self, x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        anchors = self.anchors
        if not anchors:
            return x
        if x <= anchors[0][0]:
            return anchors[0][1]
        if x >= anchors[-1][0]:
            return anchors[-1][1]
        # binary search would be faster; linear is fine at our scale
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return x  # unreachable, but defensive

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "anchors": [list(a) for a in self.anchors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationCurve":
        return cls(
            anchors=[(float(a[0]), float(a[1])) for a in data.get("anchors", [])],
            method=str(data.get("method", "identity")),
            n_samples=int(data.get("n_samples", 0)),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationCurve":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ── Fitting ───────────────────────────────────────────────────────


def fit_platt(
    samples: Sequence[CalibrationSample],
    *,
    n_anchors: int = 11,
    max_iter: int = 200,
    lr: float = 0.05,
) -> CalibrationCurve:
    """Fit a one-parameter Platt sigmoid by gradient descent.

    Solves ``sigmoid(a*(x - b))`` for (a, b) minimising binary cross-
    entropy on the (raw_conf, win) pairs. Pure stdlib — no numpy.
    """
    if len(samples) < 2:
        return CalibrationCurve.identity()
    a, b = 1.0, 0.5  # init: pass-through-ish
    n = len(samples)
    # gradient descent on negative log-likelihood
    for _ in range(max_iter):
        ga = 0.0
        gb = 0.0
        for s in samples:
            x = s.raw_confidence
            y = 1.0 if s.win else 0.0
            z = a * (x - b)
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y
            ga += err * (x - b)
            gb += err * (-a)
        a -= lr * ga / n
        b -= lr * gb / n
    # build piecewise-linear anchors
    xs = [i / (n_anchors - 1) for i in range(n_anchors)]
    anchors = [(x, 1.0 / (1.0 + math.exp(-a * (x - b)))) for x in xs]
    anchors = _enforce_monotone(anchors)
    return CalibrationCurve(anchors=anchors, method="platt", n_samples=len(samples))


def fit_isotonic(
    samples: Sequence[CalibrationSample],
    *,
    n_bins: int = 10,
    min_per_bin: int = 5,
) -> CalibrationCurve:
    """Bin raw confidences and fit a monotone-non-decreasing win-rate curve.

    Bins by equal-width on [0, 1]. Within each bin, win-rate = wins /
    samples. PAV (pool-adjacent-violators) merges any bin pair that
    breaks monotonicity. Bins below ``min_per_bin`` are merged with
    their nearest neighbour.
    """
    if len(samples) < n_bins:
        return fit_platt(samples)

    # bin
    bins: list[tuple[int, int]] = [(0, 0) for _ in range(n_bins)]  # (wins, n)
    for s in samples:
        idx = min(n_bins - 1, max(0, int(s.raw_confidence * n_bins)))
        wins, count = bins[idx]
        bins[idx] = (wins + (1 if s.win else 0), count + 1)

    # merge sparse bins
    edges = [i / n_bins for i in range(n_bins + 1)]
    grouped = [
        {"x_lo": edges[i], "x_hi": edges[i + 1], "wins": w, "n": n} for i, (w, n) in enumerate(bins)
    ]
    grouped = _merge_sparse(grouped, min_per_bin=min_per_bin)
    # PAV — pool adjacent violators
    grouped = _pav(grouped)
    # build anchors at bin centers
    anchors: list[tuple[float, float]] = [(0.0, _safe_rate(grouped[0]))]
    for g in grouped:
        center = 0.5 * (g["x_lo"] + g["x_hi"])
        anchors.append((center, _safe_rate(g)))
    anchors.append((1.0, _safe_rate(grouped[-1])))
    anchors = _enforce_monotone(_dedupe_x(anchors))
    return CalibrationCurve(anchors=anchors, method="isotonic", n_samples=len(samples))


def fit_auto(
    samples: Sequence[CalibrationSample], *, isotonic_threshold: int = 200
) -> CalibrationCurve:
    """Pick the calibration method based on sample count."""
    n = len(samples)
    if n < 30:
        return CalibrationCurve.identity()
    if n >= isotonic_threshold:
        return fit_isotonic(samples)
    return fit_platt(samples)


# ── Internals ─────────────────────────────────────────────────────


def _safe_rate(g: dict) -> float:
    n = g["n"]
    if n == 0:
        return 0.0
    return g["wins"] / n


def _merge_sparse(groups: list[dict], *, min_per_bin: int) -> list[dict]:
    out = list(groups)
    i = 0
    while i < len(out):
        if out[i]["n"] >= min_per_bin or len(out) == 1:
            i += 1
            continue
        # merge with left neighbour if exists, else right
        if i > 0:
            left = out[i - 1]
            left["wins"] += out[i]["wins"]
            left["n"] += out[i]["n"]
            left["x_hi"] = out[i]["x_hi"]
            del out[i]
            i = max(0, i - 1)
        else:
            right = out[i + 1]
            right["wins"] += out[i]["wins"]
            right["n"] += out[i]["n"]
            right["x_lo"] = out[i]["x_lo"]
            del out[i]
    return out


def _pav(groups: list[dict]) -> list[dict]:
    """Pool-adjacent-violators to enforce non-decreasing rates."""
    out = [dict(g) for g in groups]
    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 1):
            if _safe_rate(out[i]) > _safe_rate(out[i + 1]):
                out[i]["wins"] += out[i + 1]["wins"]
                out[i]["n"] += out[i + 1]["n"]
                out[i]["x_hi"] = out[i + 1]["x_hi"]
                del out[i + 1]
                changed = True
                break
    return out


def _enforce_monotone(anchors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort by x and clamp y to be non-decreasing."""
    anchors = sorted(anchors, key=lambda p: p[0])
    out: list[tuple[float, float]] = []
    last_y = -1.0
    for x, y in anchors:
        y = max(last_y, max(0.0, min(1.0, y)))
        out.append((x, y))
        last_y = y
    return out


def _dedupe_x(anchors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: dict[float, float] = {}
    for x, y in anchors:
        if x in seen:
            seen[x] = max(seen[x], y)
        else:
            seen[x] = y
    return [(x, seen[x]) for x in sorted(seen)]


# ── Sizing application ────────────────────────────────────────────


def apply_calibration(curve: CalibrationCurve, raw_conf: float) -> float:
    """Convenience: ``curve.predict(raw_conf)`` clamped to [0, 1]."""
    return max(0.0, min(1.0, curve.predict(raw_conf)))


def calibration_metrics(
    curve: CalibrationCurve, samples: Iterable[CalibrationSample], *, n_bins: int = 10
) -> dict[str, float]:
    """Reliability metrics for monitoring drift in calibrator quality.

    Returns ``{ece, brier, n}`` — ECE = Expected Calibration Error, Brier =
    mean squared error of calibrated prob vs outcome.
    """
    samples = list(samples)
    if not samples:
        return {"ece": 0.0, "brier": 0.0, "n": 0}
    bins = [{"sum_p": 0.0, "wins": 0, "n": 0} for _ in range(n_bins)]
    brier = 0.0
    for s in samples:
        p = curve.predict(s.raw_confidence)
        brier += (p - (1.0 if s.win else 0.0)) ** 2
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx]["sum_p"] += p
        bins[idx]["wins"] += 1 if s.win else 0
        bins[idx]["n"] += 1
    n = len(samples)
    ece = 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        avg_p = b["sum_p"] / b["n"]
        win_rate = b["wins"] / b["n"]
        ece += (b["n"] / n) * abs(avg_p - win_rate)
    return {"ece": ece, "brier": brier / n, "n": n}
