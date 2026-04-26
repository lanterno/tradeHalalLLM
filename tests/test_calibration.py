"""Tests for confidence calibration."""

from __future__ import annotations

import random
from pathlib import Path

from halal_trader.ml.calibration import (
    CalibrationCurve,
    CalibrationSample,
    apply_calibration,
    calibration_metrics,
    fit_auto,
    fit_isotonic,
    fit_platt,
)


def _generate_overconfident(n: int, seed: int = 0) -> list[CalibrationSample]:
    """Synthetic LLM that says 0.9 but actually wins 60%."""
    rng = random.Random(seed)
    out: list[CalibrationSample] = []
    for _ in range(n):
        raw = rng.uniform(0.6, 0.95)
        # actual win prob = roughly raw * 0.65 (over-confident bias)
        p_true = raw * 0.65
        out.append(CalibrationSample(raw_confidence=raw, win=rng.random() < p_true))
    return out


def _generate_calibrated(n: int, seed: int = 1) -> list[CalibrationSample]:
    rng = random.Random(seed)
    out: list[CalibrationSample] = []
    for _ in range(n):
        raw = rng.uniform(0.05, 0.95)
        out.append(CalibrationSample(raw_confidence=raw, win=rng.random() < raw))
    return out


# ── Curve interface ───────────────────────────────────────────────


def test_identity_curve_returns_input() -> None:
    c = CalibrationCurve.identity()
    assert c.predict(0.0) == 0.0
    assert c.predict(0.5) == 0.5
    assert c.predict(1.0) == 1.0


def test_curve_clamps_out_of_range() -> None:
    c = CalibrationCurve.identity()
    assert c.predict(-0.5) == 0.0
    assert c.predict(1.5) == 1.0


def test_curve_interpolates_linearly() -> None:
    c = CalibrationCurve(anchors=[(0.0, 0.0), (0.5, 0.2), (1.0, 0.6)])
    assert c.predict(0.0) == 0.0
    assert c.predict(0.25) == 0.1
    assert c.predict(0.75) == 0.4
    assert c.predict(1.0) == 0.6


def test_curve_round_trip_via_dict() -> None:
    c = CalibrationCurve(
        anchors=[(0.0, 0.1), (0.5, 0.4), (1.0, 0.7)],
        method="platt",
        n_samples=123,
    )
    d = c.to_dict()
    back = CalibrationCurve.from_dict(d)
    assert back.method == "platt"
    assert back.n_samples == 123
    assert back.anchors == c.anchors


def test_curve_save_load(tmp_path: Path) -> None:
    p = tmp_path / "cal.json"
    c = CalibrationCurve(anchors=[(0.0, 0.0), (1.0, 0.7)], method="iso", n_samples=42)
    c.save(p)
    back = CalibrationCurve.load(p)
    assert back.method == "iso"
    assert back.n_samples == 42


# ── Fitting ───────────────────────────────────────────────────────


def test_fit_platt_compresses_overconfidence() -> None:
    samples = _generate_overconfident(800)
    curve = fit_platt(samples)
    assert curve.method == "platt"
    # raw 0.9 should map to noticeably less (true win rate ~0.585)
    assert curve.predict(0.9) < 0.85
    assert curve.predict(0.9) > 0.4


def test_fit_isotonic_monotone() -> None:
    samples = _generate_overconfident(1000)
    curve = fit_isotonic(samples)
    last = -1.0
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = curve.predict(x)
        assert y >= last - 1e-9
        last = y


def test_fit_isotonic_compresses_overconfidence() -> None:
    samples = _generate_overconfident(1000)
    curve = fit_isotonic(samples)
    assert curve.predict(0.9) < 0.85


def test_fit_auto_uses_isotonic_for_large_n() -> None:
    samples = _generate_overconfident(500)
    curve = fit_auto(samples)
    assert curve.method == "isotonic"


def test_fit_auto_uses_platt_for_small_n() -> None:
    samples = _generate_overconfident(50)
    curve = fit_auto(samples)
    assert curve.method == "platt"


def test_fit_auto_returns_identity_below_threshold() -> None:
    samples = _generate_overconfident(20)
    curve = fit_auto(samples)
    assert curve.method == "identity"
    assert curve.predict(0.7) == 0.7


def test_apply_calibration_clamps() -> None:
    c = CalibrationCurve(anchors=[(0.0, 0.0), (1.0, 1.5)])
    # anchor's y is clamped to 1.0 by enforce_monotone — but use the raw curve
    assert apply_calibration(c, 1.0) <= 1.0
    assert apply_calibration(c, -1.0) >= 0.0


# ── Metrics ───────────────────────────────────────────────────────


def test_metrics_identity_brier_high_when_overconfident() -> None:
    samples = _generate_overconfident(500)
    m_identity = calibration_metrics(CalibrationCurve.identity(), samples)
    fitted = fit_platt(samples)
    m_fitted = calibration_metrics(fitted, samples)
    # The fitted curve should be at least as good as identity on Brier+ECE.
    assert m_fitted["brier"] <= m_identity["brier"]
    assert m_fitted["ece"] <= m_identity["ece"]


def test_metrics_empty_samples() -> None:
    m = calibration_metrics(CalibrationCurve.identity(), [])
    assert m["n"] == 0
    assert m["ece"] == 0.0


def test_calibrated_curve_close_to_identity_on_well_calibrated_data() -> None:
    samples = _generate_calibrated(800)
    fitted = fit_platt(samples)
    # On well-calibrated data, predict(0.5) should be near 0.5
    assert abs(fitted.predict(0.5) - 0.5) < 0.15
