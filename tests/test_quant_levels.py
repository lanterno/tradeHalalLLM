"""Tests for quant/levels.py and quant/level_eval.py."""

from __future__ import annotations

import numpy as np
import pytest

from halal_trader.quant.level_eval import (
    evaluate_family,
    merge_stats,
    placebo_uplift,
)
from halal_trader.quant.levels import (
    Level,
    atr_series,
    level_map,
    prior_extreme_levels,
    round_number_levels,
    swing_zones,
)


class TestAtrSeries:
    def test_constant_range_converges_to_range(self):
        n = 200
        h = np.full(n, 102.0)
        lo = np.full(n, 98.0)
        c = np.full(n, 100.0)
        atr = atr_series(h, lo, c)
        assert atr[-1] == pytest.approx(4.0, rel=1e-6)

    def test_gap_counts_via_prev_close(self):
        # Second bar gaps: TR = max(h-l, |h-prev_c|, |l-prev_c|) = 110-100=10
        h = np.array([101.0, 110.0])
        lo = np.array([99.0, 109.0])
        c = np.array([100.0, 109.5])
        atr = atr_series(h, lo, c, window=14)
        assert atr[1] > atr[0]  # the gap inflated the estimate

    def test_validation(self):
        with pytest.raises(ValueError):
            atr_series([1.0], [1.0, 2.0], [1.0])


class TestPriorExtremes:
    def test_day_week_month_levels(self):
        # Two ISO weeks (Mon-Fri) spanning a month boundary.
        dates = [
            "2026-04-27",
            "2026-04-28",
            "2026-04-29",
            "2026-04-30",
            "2026-05-01",
            "2026-05-04",
            "2026-05-05",
            "2026-05-06",
        ]
        highs = np.array([101, 102, 110, 103, 104, 105, 106, 107], dtype=float)
        lows = np.array([99, 98, 90, 97, 96, 95, 94, 93], dtype=float)
        levels = {lvl.kind: lvl.price for lvl in prior_extreme_levels(dates, highs, lows)}
        assert levels["prior_day_high"] == 107.0
        assert levels["prior_day_low"] == 93.0
        # Last bar is in ISO week of 2026-05-04; prior completed week is
        # 04-27..05-01 → high 110, low 90.
        assert levels["prior_week_high"] == 110.0
        assert levels["prior_week_low"] == 90.0
        # Prior completed month is April (04-27..04-30) → high 110, low 90.
        assert levels["prior_month_high"] == 110.0
        assert levels["prior_month_low"] == 90.0

    def test_no_prior_period_omits_levels(self):
        dates = ["2026-05-04", "2026-05-05"]
        highs = np.array([101.0, 102.0])
        lows = np.array([99.0, 98.0])
        kinds = {lvl.kind for lvl in prior_extreme_levels(dates, highs, lows)}
        assert "prior_day_high" in kinds
        assert "prior_week_high" not in kinds  # no completed prior week in window
        assert "prior_month_high" not in kinds


class TestRoundNumbers:
    def test_magnitude_scaled_grids(self):
        kinds = {(lvl.kind, lvl.price) for lvl in round_number_levels(103.7)}
        assert ("round_100", 100.0) in kinds
        assert ("round_100", 200.0) in kinds
        assert ("round_5", 105.0) in kinds
        # $1 grid is too fine for a $103 stock (103/1 > 60): absent.
        assert not any(k == "round_1" for k, _ in kinds)

    def test_low_price_gets_fine_grid(self):
        kinds = {lvl.kind for lvl in round_number_levels(23.4)}
        assert "round_1" in kinds

    def test_validation(self):
        with pytest.raises(ValueError):
            round_number_levels(0.0)


class TestSwingZones:
    def test_repeated_touches_cluster_and_rank(self):
        # Price oscillates: repeated swing lows near 100, one high near 120.
        lows, highs = [], []
        for cycle in range(5):
            lows += [110, 105, 100 + 0.1 * cycle, 105, 110]
            highs += [115, 112, 104, 118, 120]
        h = np.array(highs, dtype=float)
        lo = np.array(lows, dtype=float)
        zones = swing_zones(h, lo, atr=2.0, confirm=2)
        assert zones, "expected at least one zone"
        strongest = zones[0]
        assert strongest.strength >= 3
        assert 99.5 <= strongest.price <= 101.0  # the repeated ~100 swing low

    def test_confirmation_excludes_recent_bars(self):
        # A huge spike in the last `confirm` bars must NOT create a zone.
        h = np.concatenate([np.full(30, 100.0), [200.0, 100.0]])
        lo = h - 2
        zones = swing_zones(h, lo, atr=2.0, confirm=3)
        assert all(abs(z.price - 200.0) > 1 for z in zones)

    def test_short_series_empty(self):
        assert swing_zones(np.ones(4), np.ones(4), atr=1.0, confirm=3) == []


class TestLevelMap:
    def test_snaps_to_round_numbers(self):
        dates = [f"2026-05-{d:02d}" for d in range(1, 11)]
        highs = np.full(10, 100.2)  # prior-day high 100.2, near round 100
        lows = np.full(10, 95.7)
        closes = np.full(10, 98.0)
        levels = level_map(dates, highs, lows, closes, atr=2.0)
        pdh = next(lvl for lvl in levels if lvl.kind.startswith("prior_day_high"))
        assert pdh.price == 100.0  # snapped (|100.2-100| = 0.2 <= 0.15*2.0)
        assert "+round" in pdh.kind

    def test_sorted_by_price(self):
        dates = [f"2026-05-{d:02d}" for d in range(1, 11)]
        rng = np.random.default_rng(0)
        closes = 100 + rng.normal(0, 1, 10).cumsum()
        highs = closes + 1
        lows = closes - 1
        levels = level_map(dates, highs, lows, closes, atr=1.5)
        prices = [lvl.price for lvl in levels]
        assert prices == sorted(prices)


def _flat_dates(n: int) -> list[str]:
    return [f"2026-{2 + i // 28:02d}-{(i % 28) + 1:02d}" for i in range(n)]


class TestTouchAndHold:
    def _support_bounce_series(self, n_cycles: int = 12):
        """Support at 100 that always rejects: dip to 100, rally to 106."""
        h, lo, c = [], [], []
        for _ in range(n_cycles):
            for bar_h, bar_l, bar_c in (
                (106, 103, 104),
                (104, 100.2, 101),  # touches the 100 zone
                (106, 101, 105.5),  # closes 1+ ATR above → reject
                (107, 104, 106),
            ):
                h.append(bar_h)
                lo.append(bar_l)
                c.append(bar_c)
        return np.array(h, float), np.array(lo, float), np.array(c, float)

    def test_perfect_support_scores_high_hold_rate(self):
        h, lo, c = self._support_bounce_series()
        dates = _flat_dates(len(c))
        stats = evaluate_family(
            dates,
            h,
            lo,
            c,
            lambda d, hh, ll, cc, atr: [100.0],
            label="fixture",
            horizon=4,
            warmup=8,
        )
        assert stats.touches > 0
        assert stats.hold_rate is not None
        assert stats.hold_rate > 0.9
        assert stats.breaks == 0

    def test_breaking_level_scores_low(self):
        # Downtrend slicing through every level placed above the lows.
        n = 60
        c = np.linspace(150, 90, n)
        h = c + 1
        lo = c - 1
        dates = _flat_dates(n)
        stats = evaluate_family(
            dates,
            h,
            lo,
            c,
            lambda d, hh, ll, cc, atr: [float(cc[-1]) - 2.0],  # support 2$ below
            label="fixture",
            horizon=5,
            warmup=10,
        )
        assert stats.touches > 0
        assert stats.hold_rate is not None
        assert stats.hold_rate < 0.5

    def test_placebo_is_deterministic_and_distance_matched(self):
        h, lo, c = self._support_bounce_series()
        dates = _flat_dates(len(c))
        fam = lambda d, hh, ll, cc, atr: [100.0]  # noqa: E731
        p1 = evaluate_family(dates, h, lo, c, fam, label="p", horizon=4, warmup=8, placebo_seed=7)
        p2 = evaluate_family(dates, h, lo, c, fam, label="p", horizon=4, warmup=8, placebo_seed=7)
        assert p1 == p2  # seeded → reproducible
        real = evaluate_family(dates, h, lo, c, fam, label="r", horizon=4, warmup=8)
        up = placebo_uplift(real, p1)
        # The fixture's support is real by construction; placebo shouldn't win.
        assert up is None or up >= 0

    def test_unreachable_levels_are_skipped(self):
        h, lo, c = self._support_bounce_series()
        dates = _flat_dates(len(c))
        stats = evaluate_family(
            dates,
            h,
            lo,
            c,
            lambda d, hh, ll, cc, atr: [1000.0],  # far beyond 3·ATR reach
            label="far",
            horizon=4,
            warmup=8,
        )
        assert stats.n_level_days == 0

    def test_merge_stats_pools(self):
        a = evaluate_family(
            _flat_dates(48),
            *self._support_bounce_series(),
            lambda d, hh, ll, cc, atr: [100.0],
            label="a",
            horizon=4,
            warmup=8,
        )
        merged = merge_stats("pool", [a, a])
        assert merged.touches == 2 * a.touches
        assert merged.hold_rate == a.hold_rate

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_family(
                ["2026-01-01"],
                np.ones(2),
                np.ones(2),
                np.ones(2),
                lambda d, hh, ll, cc, atr: [],
                label="x",
            )


def test_level_dataclass_frozen():
    lvl = Level(100.0, "prior_day_high")
    with pytest.raises(AttributeError):
        lvl.price = 101.0  # type: ignore[misc]
