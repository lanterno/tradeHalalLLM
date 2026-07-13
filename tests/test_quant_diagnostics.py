"""Tests for quant/diagnostics.py + the split-gap labeling guard."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.diagnostics import (
    overnight_intraday_split,
    suspect_split_gaps,
)
from halal_trader.recommendation.scorecard import _forward_outcomes, _ohlc_by_date


class TestOvernightSplit:
    def test_hand_computed_decomposition(self):
        # Day 1: close 100. Day 2: opens 102 (overnight +2%), closes 104.04
        # (intraday +2%). Day 3: opens 104.04 (overnight 0%), closes 109.242
        # (intraday +5%).
        opens = [100.0, 102.0, 104.04]
        closes = [100.0, 104.04, 109.242]
        split = overnight_intraday_split(opens, closes)
        assert split.n_days == 2
        assert split.mean_overnight_pct == pytest.approx(1.0)  # (2% + 0%) / 2
        assert split.mean_intraday_pct == pytest.approx(3.5)  # (2% + 5%) / 2
        assert split.cum_overnight_pct == pytest.approx(2.0)
        assert split.cum_intraday_pct == pytest.approx(7.1)

    def test_validation(self):
        with pytest.raises(ValueError):
            overnight_intraday_split([100.0], [100.0])
        with pytest.raises(ValueError):
            overnight_intraday_split([100.0, -1.0], [100.0, 100.0])


class TestSuspectSplitGaps:
    def test_detects_split_shaped_gap(self):
        # 2:1 split: close 200, next open 100 → −50% overnight gap.
        opens = [200.0, 200.0, 100.0, 101.0]
        closes = [200.0, 200.0, 101.0, 102.0]
        gaps = suspect_split_gaps(opens, closes)
        assert gaps == [(2, -0.5)]

    def test_normal_moves_pass(self):
        rng = np.random.default_rng(0)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
        opens = np.concatenate([[100.0], closes[:-1] * np.exp(rng.normal(0, 0.005, 199))])
        assert suspect_split_gaps(opens, closes) == []

    def test_big_but_real_crash_passes(self):
        # A META-style −26% day must NOT be flagged (threshold is 40%).
        opens = [300.0, 222.0]
        closes = [300.0, 220.0]
        assert suspect_split_gaps(opens, closes) == []

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            suspect_split_gaps([1.0, 1.0], [1.0, 1.0], threshold=1.5)


class TestLabelingGuard:
    def test_forward_outcomes_refuses_to_label_across_split(self):
        # Entry at 200; a 2:1 split gap two bars later inside the window.
        bars = [
            {"t": "2026-05-01", "o": 200.0, "c": 200.0, "h": 201.0, "l": 199.0},
            {"t": "2026-05-02", "o": 201.0, "c": 202.0, "h": 203.0, "l": 200.0},
            {"t": "2026-05-03", "o": 101.0, "c": 102.0, "h": 103.0, "l": 100.0},
            *[
                {"t": f"2026-05-{d:02d}", "o": 102.0, "c": 102.0, "h": 103.0, "l": 101.0}
                for d in range(4, 12)
            ],
        ]
        fr = _forward_outcomes(_ohlc_by_date(bars), "2026-05-01")
        assert fr == {"suspect_gap": True}

    def test_gap_before_entry_does_not_block(self):
        # The split happened BEFORE the rec date: the post-split window is
        # internally consistent and labels fine.
        bars = [
            {"t": "2026-05-01", "o": 200.0, "c": 200.0, "h": 201.0, "l": 199.0},
            {"t": "2026-05-02", "o": 100.0, "c": 100.0, "h": 101.0, "l": 99.0},
            *[
                {"t": f"2026-05-{d:02d}", "o": 100.0, "c": 100.0, "h": 101.0, "l": 99.0}
                for d in range(3, 30)
            ],
        ]
        fr = _forward_outcomes(_ohlc_by_date(bars), "2026-05-02")
        assert fr is not None
        assert fr.get("suspect_gap") is None
        assert fr["entry_close"] == 100.0
