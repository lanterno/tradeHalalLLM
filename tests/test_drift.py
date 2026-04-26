"""Tests for online concept-drift detection."""

from __future__ import annotations

import random

from halal_trader.ml.drift import (
    AdwinLiteDetector,
    DriftMonitor,
    DriftRiskPolicy,
    PageHinkleyDetector,
)


def test_page_hinkley_quiet_stream_no_drift() -> None:
    rng = random.Random(0)
    det = PageHinkleyDetector(threshold=10.0)
    for _ in range(200):
        assert det.observe(rng.gauss(0, 0.1)) is False


def test_page_hinkley_detects_step_change() -> None:
    rng = random.Random(0)
    det = PageHinkleyDetector(threshold=5.0, delta=0.001, alpha=0.99)
    fired = False
    for i in range(400):
        # mean shifts +1.0 after step 200
        v = rng.gauss(0, 0.1) + (1.0 if i >= 200 else 0.0)
        if det.observe(v):
            fired = True
            assert i >= 200
            break
    assert fired, "PH should detect a sustained 1-sigma+ shift"


def test_adwin_lite_quiet_stream_no_drift() -> None:
    rng = random.Random(0)
    det = AdwinLiteDetector(window=40, min_obs=20)
    for _ in range(200):
        assert det.observe(rng.gauss(0, 0.1)) is False


def test_adwin_lite_detects_mean_shift() -> None:
    rng = random.Random(0)
    det = AdwinLiteDetector(window=40, z=3.0, min_obs=20)
    # Phase 1: calm
    for _ in range(100):
        det.observe(rng.gauss(0, 0.1))
    # Phase 2: shifted by 1.5 (15× σ) — must fire at least once
    fired_in_shift = False
    for _ in range(100):
        if det.observe(rng.gauss(1.5, 0.1)):
            fired_in_shift = True
            break
    assert fired_in_shift, "ADWIN-lite should detect a sustained 15σ shift"


def test_drift_monitor_warming_up_then_stable() -> None:
    mon = DriftMonitor()
    rng = random.Random(0)
    assert mon.state == "warming_up"
    for _ in range(mon.adwin.min_obs):
        mon.observe(rng.gauss(0, 0.1))
    assert mon.state == "stable"


def test_drift_monitor_state_drift_after_shift() -> None:
    mon = DriftMonitor()
    rng = random.Random(0)
    for _ in range(60):
        mon.observe(rng.gauss(0, 0.05))
    assert mon.state == "stable"
    assert mon.drift_count == 0
    fired = False
    for i in range(120):
        if mon.observe(rng.gauss(2.0, 0.1)):
            fired = True
            break
    assert fired
    assert mon.state == "drift"
    assert mon.last_drift_at is not None
    assert mon.drift_count >= 1


def test_drift_monitor_returns_to_stable_after_cooldown() -> None:
    mon = DriftMonitor(cooldown=3)
    rng = random.Random(0)
    # warm up then induce drift
    for _ in range(60):
        mon.observe(rng.gauss(0, 0.05))
    for _ in range(80):
        if mon.observe(rng.gauss(2.0, 0.05)):
            break
    assert mon.state == "drift"
    drift_at = mon.last_drift_at
    # Feed a few more samples — cooldown still active
    mon.observe(0.0)
    mon.observe(0.0)
    assert mon.state == "drift"
    # Past cooldown
    for _ in range(5):
        mon.observe(0.0)
    assert mon.n - drift_at > mon.cooldown
    assert mon.state == "stable"


def test_drift_risk_policy_multipliers() -> None:
    pol = DriftRiskPolicy(
        drift_size_multiplier=0.5,
        drift_sl_tighten=0.7,
        warming_up_size_multiplier=0.8,
    )
    assert pol.size_multiplier("stable") == 1.0
    assert pol.size_multiplier("drift") == 0.5
    assert pol.size_multiplier("warming_up") == 0.8
    assert pol.sl_multiplier("drift") == 0.7
    assert pol.sl_multiplier("stable") == 1.0


def test_detectors_resettable() -> None:
    ph = PageHinkleyDetector()
    for v in [0.0, 1.0, -1.0, 2.0]:
        ph.observe(v)
    ph.reset()
    assert ph.n == 0
    assert ph.cumulative == 0.0

    aw = AdwinLiteDetector()
    for _ in range(10):
        aw.observe(1.0)
    aw.reset()
    assert aw.n == 0
    assert len(aw._buf) == 0

    mon = DriftMonitor()
    for _ in range(40):
        mon.observe(0.0)
    mon.reset()
    assert mon.n == 0
    assert mon.last_drift_at is None
    assert mon.drift_count == 0
