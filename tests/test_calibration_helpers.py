"""Tests for the private helpers in :mod:`ml.calibration`.

The public `fit_platt` / `fit_isotonic` / `fit_auto` are integration-
tested in `test_calibration.py`. This file pins the small numeric
helpers underneath: `_safe_rate`, `_merge_sparse`, `_pav`,
`_enforce_monotone`, `_dedupe_x`.
"""

from __future__ import annotations

from halal_trader.ml.calibration import (
    _dedupe_x,
    _enforce_monotone,
    _merge_sparse,
    _pav,
    _safe_rate,
)

# ── _safe_rate ──────────────────────────────────────────────


def test_safe_rate_zero_n_returns_zero():
    """Defensive: an empty bin yields 0 win rate, not a divide-by-zero."""
    assert _safe_rate({"n": 0, "wins": 0}) == 0.0


def test_safe_rate_basic():
    assert _safe_rate({"n": 10, "wins": 6}) == 0.6


def test_safe_rate_perfect_winner():
    assert _safe_rate({"n": 5, "wins": 5}) == 1.0


# ── _merge_sparse ──────────────────────────────────────────


def test_merge_sparse_no_change_when_all_full():
    """If every bin meets the floor, no merging happens."""
    groups = [
        {"x_lo": 0.0, "x_hi": 0.5, "n": 10, "wins": 5},
        {"x_lo": 0.5, "x_hi": 1.0, "n": 10, "wins": 7},
    ]
    out = _merge_sparse(groups, min_per_bin=5)
    assert len(out) == 2


def test_merge_sparse_merges_undersized_bin_left():
    """A bin under the floor merges into its left neighbour."""
    groups = [
        {"x_lo": 0.0, "x_hi": 0.5, "n": 10, "wins": 5},
        {"x_lo": 0.5, "x_hi": 1.0, "n": 1, "wins": 0},  # too few
    ]
    out = _merge_sparse(groups, min_per_bin=5)
    assert len(out) == 1
    assert out[0]["n"] == 11  # 10 + 1
    assert out[0]["wins"] == 5
    assert out[0]["x_hi"] == 1.0  # extended to absorb the merged bin


def test_merge_sparse_first_bin_merges_right_when_no_left():
    """When the undersized bin has no left neighbour, it merges right."""
    groups = [
        {"x_lo": 0.0, "x_hi": 0.5, "n": 1, "wins": 0},  # too few, no left
        {"x_lo": 0.5, "x_hi": 1.0, "n": 10, "wins": 5},
    ]
    out = _merge_sparse(groups, min_per_bin=5)
    assert len(out) == 1
    assert out[0]["n"] == 11
    assert out[0]["x_lo"] == 0.0  # extended to absorb the merged bin


def test_merge_sparse_keeps_a_lone_bin():
    """If only one bin exists, it's kept even if it's under the floor —
    can't merge a bin with itself."""
    groups = [{"x_lo": 0.0, "x_hi": 1.0, "n": 1, "wins": 0}]
    out = _merge_sparse(groups, min_per_bin=10)
    assert out == groups


# ── _pav ───────────────────────────────────────────────────


def test_pav_already_monotone_no_change():
    groups = [
        {"x_lo": 0.0, "x_hi": 0.5, "n": 10, "wins": 3},  # 0.3
        {"x_lo": 0.5, "x_hi": 1.0, "n": 10, "wins": 7},  # 0.7
    ]
    out = _pav(groups)
    assert len(out) == 2
    assert _safe_rate(out[0]) == 0.3
    assert _safe_rate(out[1]) == 0.7


def test_pav_pools_when_left_higher_than_right():
    """A backwards step (rate dropping left → right) pools the two bins."""
    groups = [
        {"x_lo": 0.0, "x_hi": 0.5, "n": 10, "wins": 8},  # 0.8 ← higher
        {"x_lo": 0.5, "x_hi": 1.0, "n": 10, "wins": 3},  # 0.3
    ]
    out = _pav(groups)
    assert len(out) == 1
    assert out[0]["wins"] == 11
    assert out[0]["n"] == 20


def test_pav_idempotent_after_one_pass():
    """Running PAV twice yields the same result as once."""
    groups = [
        {"x_lo": 0.0, "x_hi": 0.5, "n": 10, "wins": 8},
        {"x_lo": 0.5, "x_hi": 1.0, "n": 10, "wins": 3},
    ]
    one = _pav(groups)
    two = _pav(one)
    assert one == two


# ── _enforce_monotone ─────────────────────────────────────


def test_enforce_monotone_sorts_by_x():
    out = _enforce_monotone([(0.5, 0.3), (0.1, 0.2)])
    assert out[0][0] == 0.1
    assert out[1][0] == 0.5


def test_enforce_monotone_clamps_y_in_unit_interval():
    out = _enforce_monotone([(0.1, -0.5), (0.5, 1.5)])
    assert out[0][1] == 0.0  # clamped from negative
    assert out[1][1] == 1.0  # clamped from > 1


def test_enforce_monotone_makes_y_non_decreasing():
    """A backwards-y point gets pulled up to match the prior max."""
    out = _enforce_monotone([(0.1, 0.7), (0.5, 0.3)])
    assert out[1][1] == 0.7  # raised to match prior


# ── _dedupe_x ──────────────────────────────────────────────


def test_dedupe_x_keeps_max_y_for_duplicate_x():
    out = _dedupe_x([(0.5, 0.2), (0.5, 0.7), (0.5, 0.4)])
    assert len(out) == 1
    assert out[0] == (0.5, 0.7)


def test_dedupe_x_sorts_output_by_x():
    out = _dedupe_x([(0.9, 0.1), (0.1, 0.2), (0.5, 0.3)])
    assert [p[0] for p in out] == [0.1, 0.5, 0.9]


def test_dedupe_x_passes_unique_x_through():
    out = _dedupe_x([(0.1, 0.2), (0.5, 0.3)])
    assert out == [(0.1, 0.2), (0.5, 0.3)]
