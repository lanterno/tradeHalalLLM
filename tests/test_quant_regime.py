"""Tests for quant/regime.py — VIX term-structure regime gate."""

from __future__ import annotations

import pytest

from halal_trader.quant.regime import (
    CAUTION,
    RISK_OFF,
    RISK_ON,
    RegimeReading,
    _parse_cboe_csv,
    classify_raw,
    format_for_prompt,
    regime_from_series,
)


class TestClassifyRaw:
    def test_contango_is_risk_on(self):
        assert classify_raw(0.88, 0.85) == RISK_ON  # VIX well below VIX3M

    def test_flattening_is_caution(self):
        assert classify_raw(0.97, 0.95) == CAUTION

    def test_backwardation_is_risk_off(self):
        assert classify_raw(1.05, 1.10) == RISK_OFF  # VIX above VIX3M

    def test_gate_ignores_fast_ratio(self):
        # A screaming-hot r_fast can't move the gate on its own; r_slow rules.
        assert classify_raw(0.88, 1.5) == RISK_ON


def _flat_series(r_slow: float, r_fast: float, n: int):
    """n identical days at the given ratios (vix=20 fixed)."""
    vix = [20.0] * n
    vix3m = [20.0 / r_slow] * n
    vix9d = [20.0 * r_fast] * n
    return vix9d, vix, vix3m


class TestRegimeFromSeries:
    def test_steady_risk_on(self):
        r = regime_from_series(*_flat_series(0.88, 0.85, 10))
        assert r is not None
        assert r.regime == RISK_ON
        assert r.r_slow == pytest.approx(0.88, abs=0.01)
        assert r.fast_inverted is False

    def test_steady_risk_off_and_inversion_flag(self):
        r = regime_from_series(*_flat_series(1.05, 1.10, 10))
        assert r.regime == RISK_OFF
        assert r.fast_inverted is True

    def test_hysteresis_holds_through_single_day_spike(self):
        # 8 calm days, then ONE risk-off day, then back — must stay risk_on.
        v9, v, v3 = _flat_series(0.88, 0.85, 8)
        # inject a single risk-off day at the end-1, calm day at the end
        v9 += [20.0 * 1.10, 20.0 * 0.85]
        v += [20.0, 20.0]
        v3 += [20.0 / 1.05, 20.0 / 0.88]
        r = regime_from_series(v9, v, v3)
        assert r.regime == RISK_ON  # one day can't flip a 2-day-confirmed gate

    def test_hysteresis_switches_after_two_confirming_days(self):
        v9, v, v3 = _flat_series(0.88, 0.85, 6)
        # two consecutive risk-off days at the end → switch
        for _ in range(2):
            v9.append(20.0 * 1.10)
            v.append(20.0)
            v3.append(20.0 / 1.05)
        r = regime_from_series(v9, v, v3)
        assert r.regime == RISK_OFF

    def test_empty_and_nonpositive(self):
        assert regime_from_series([], [], []) is None
        assert regime_from_series([0.0], [20.0], [22.0]) is None


class TestParseCsv:
    def test_parses_cboe_shape(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\n07/13/2026,16.32,17.41,16.03,17.16\n"
        out = _parse_cboe_csv(csv)
        assert out == {"2026-07-13": 17.16}

    def test_skips_header_and_junk(self):
        csv = "DATE,OPEN,HIGH,LOW,CLOSE\ngarbage\n07/10/2026,1,2,3,15.03\n,,,,\n"
        assert _parse_cboe_csv(csv) == {"2026-07-10": 15.03}


class TestFormatPrompt:
    def test_risk_off_prompt_mentions_backwardation(self):
        r = RegimeReading(regime=RISK_OFF, r_slow=1.05, r_fast=1.10, fast_inverted=True, vix=28.0)
        text = format_for_prompt(r)
        assert "RISK-OFF" in text and "BACKWARDATION" in text
        assert "short-end inverted" in text

    def test_risk_on_prompt_is_calm(self):
        r = RegimeReading(regime=RISK_ON, r_slow=0.88, r_fast=0.85, fast_inverted=False, vix=15.0)
        text = format_for_prompt(r)
        assert "RISK-ON" in text and "short-end inverted" not in text
