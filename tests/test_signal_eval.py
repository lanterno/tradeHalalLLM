"""Tests for the IC / ICIR signal-evaluation harness."""

from __future__ import annotations

import pytest

from halal_trader.core.signal_eval import (
    information_coefficient,
)


def test_ic_perfect_monotonic_is_one():
    assert information_coefficient([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_ic_perfect_inverse_is_minus_one():
    assert information_coefficient([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_ic_monotonic_but_nonlinear_still_one():
    # Spearman (rank) → monotonic non-linear still scores 1.0.
    assert information_coefficient([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)


def test_ic_no_relation_near_zero():
    signal = list(range(20))
    outcomes = [3, -1, 2, 0, -2, 1, 4, -3, 0, 2, -1, 1, 3, -2, 0, 2, -1, 1, -3, 4]
    assert abs(information_coefficient(signal, outcomes)) < 0.5


def test_ic_handles_ties():
    # Ties in the signal must not crash and stay in [-1, 1].
    ic = information_coefficient([1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6])
    assert -1.0 <= ic <= 1.0
    assert ic > 0  # still positively associated


def test_ic_degenerate_inputs():
    assert information_coefficient([1, 2], [1, 2]) == 0.0  # < 3 points
    assert information_coefficient([5, 5, 5, 5], [1, 2, 3, 4]) == 0.0  # constant signal
    assert information_coefficient([1, 2, 3], [1, 2]) == 0.0  # shape mismatch


def test_ic_filters_nan():
    ic = information_coefficient([1, 2, float("nan"), 4, 5], [10, 20, 30, 40, 50])
    assert ic == pytest.approx(1.0)  # the nan pair is dropped, rest perfect
