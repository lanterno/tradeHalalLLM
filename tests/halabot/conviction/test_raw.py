"""Raw conviction + identity calibrator (REARCHITECTURE B.2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.belief.schema import EvidenceItem
from halabot.conviction.raw import IdentityCalibrator, conviction_raw

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _ev(direction, weight=1.0, *, source="x", directional=True):
    return EvidenceItem(
        source=source, direction=direction, weight=weight, ts=T0, directional=directional
    )


def test_empty_evidence_is_zero():
    assert conviction_raw([], 1.0) == 0.0


def test_long_only_net_bearish_is_zero():
    assert conviction_raw([_ev(-1.0)], 1.0) == 0.0


def test_net_flat_is_zero():
    assert conviction_raw([_ev(1.0), _ev(-1.0)], 1.0) == 0.0


def test_strong_unanimous_bullish_high_conviction():
    # signed=1, agreement=1 → factor 1.0; regime_conf=1 → raw=1.0
    assert conviction_raw([_ev(1.0), _ev(1.0)], 1.0) == 1.0


def test_regime_confidence_scales_down():
    full = conviction_raw([_ev(1.0)], 1.0)
    half = conviction_raw([_ev(1.0)], 0.5)
    assert half == pytest.approx(full * 0.5)


def test_drift_flag_applies_penalty():
    base = conviction_raw([_ev(1.0)], 1.0)
    drifted = conviction_raw([_ev(1.0)], 1.0, drift_flag=True)
    assert drifted == pytest.approx(base * 0.7)


def test_anomaly_flag_applies_penalty():
    base = conviction_raw([_ev(1.0)], 1.0)
    anom = conviction_raw([_ev(1.0)], 1.0, anomaly_flag=True)
    assert anom == pytest.approx(base * 0.6)


def test_raw_is_clamped_to_unit_interval():
    out = conviction_raw([_ev(1.0)], 5.0)  # regime_conf clamps to 1.0
    assert 0.0 <= out <= 1.0


def test_disagreement_lowers_conviction_vs_unanimous():
    unanimous = conviction_raw([_ev(1.0), _ev(1.0)], 1.0)
    mixed = conviction_raw([_ev(1.0), _ev(1.0), _ev(-0.2, 0.3)], 1.0)
    assert mixed < unanimous


@pytest.mark.asyncio
async def test_identity_calibrator_passes_through_clamped():
    c = IdentityCalibrator()
    assert await c.calibrate("NVDA", 0.42, features={}) == 0.42
    assert await c.calibrate("NVDA", 1.5, features={}) == 1.0
