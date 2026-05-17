"""Tests for trading/queue_position.py — Round-5 Wave 12.J."""

from __future__ import annotations

import math

import pytest

from halal_trader.trading.queue_position import (
    QueueAssessment,
    QueueInputs,
    estimate,
    render_assessment,
)


def _inputs(**overrides) -> QueueInputs:
    base = {
        "order_id": "Q-001",
        "arrival_size_ahead": 1000.0,
        "current_size_ahead": 800.0,
        "current_total_at_level": 1500.0,
        "market_volume_rate_at_level": 50.0,
    }
    base.update(overrides)
    return QueueInputs(**base)


# --- Validation -----------------------------------------------------


def test_inputs_empty_id_rejected():
    with pytest.raises(ValueError):
        _inputs(order_id="")


def test_inputs_negative_arrival_rejected():
    with pytest.raises(ValueError):
        _inputs(arrival_size_ahead=-1)


def test_inputs_zero_total_rejected():
    with pytest.raises(ValueError):
        _inputs(current_total_at_level=0)


def test_inputs_negative_rate_rejected():
    with pytest.raises(ValueError):
        _inputs(market_volume_rate_at_level=-1)


def test_inputs_current_exceeds_arrival_rejected():
    with pytest.raises(ValueError):
        _inputs(arrival_size_ahead=100, current_size_ahead=200)


def test_assessment_position_outside_unit_rejected():
    with pytest.raises(ValueError):
        QueueAssessment(
            order_id="x",
            queue_position=1.5,
            fraction_consumed=0.0,
            expected_seconds_to_fill=10,
        )


def test_assessment_negative_seconds_rejected():
    with pytest.raises(ValueError):
        QueueAssessment(
            order_id="x",
            queue_position=0.5,
            fraction_consumed=0.0,
            expected_seconds_to_fill=-1,
        )


# --- Estimation -----------------------------------------------------


def test_basic_estimation():
    a = estimate(_inputs())
    # current=800, total=1500 → queue=53%
    assert a.queue_position == pytest.approx(800 / 1500)
    # arrival=1000, current=800 → consumed=20%
    assert a.fraction_consumed == pytest.approx(0.20)
    # 800 / 50 = 16s
    assert a.expected_seconds_to_fill == pytest.approx(16.0)


def test_estimation_at_front_zero_position():
    a = estimate(_inputs(current_size_ahead=0))
    assert a.queue_position == 0.0
    assert a.fraction_consumed == 1.0
    assert a.expected_seconds_to_fill == 0.0


def test_estimation_no_consumption_at_arrival():
    a = estimate(_inputs(current_size_ahead=1000.0))  # nothing consumed
    assert a.fraction_consumed == 0.0


def test_estimation_zero_volume_infinite_eta():
    a = estimate(_inputs(market_volume_rate_at_level=0.0, current_size_ahead=100))
    assert math.isinf(a.expected_seconds_to_fill)


def test_estimation_zero_volume_at_front_zero_eta():
    a = estimate(_inputs(market_volume_rate_at_level=0.0, current_size_ahead=0))
    assert a.expected_seconds_to_fill == 0.0


def test_estimation_zero_arrival_treats_consumed_as_one():
    a = estimate(_inputs(arrival_size_ahead=0, current_size_ahead=0))
    assert a.fraction_consumed == 1.0


def test_estimation_position_clamped_to_one():
    """If current_size_ahead exceeds total_at_level (data anomaly), clamp at 1."""
    a = estimate(
        _inputs(
            arrival_size_ahead=1000,
            current_size_ahead=900,
            current_total_at_level=500,  # smaller than current_size_ahead
        )
    )
    assert a.queue_position == 1.0


# --- Render --------------------------------------------------------


def test_render_includes_summary():
    a = estimate(_inputs())
    out = render_assessment(a)
    assert "Q-001" in out
    assert "queue_pos" in out
    assert "ETA" in out


def test_render_infinite_eta():
    a = estimate(_inputs(market_volume_rate_at_level=0.0, current_size_ahead=100))
    out = render_assessment(a)
    assert "∞" in out


def test_render_no_secret_leak():
    a = estimate(_inputs())
    out = render_assessment(a)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ---------------------------------------------------------


def test_e2e_queue_progresses_to_front():
    """Walk from queue=53% → queue=0% as orders ahead consume."""
    arrival_position = 1000.0
    progressions = [800, 600, 400, 200, 0]
    etas = []
    for current in progressions:
        a = estimate(
            QueueInputs(
                order_id="Q",
                arrival_size_ahead=arrival_position,
                current_size_ahead=current,
                current_total_at_level=1500.0,
                market_volume_rate_at_level=50.0,
            )
        )
        etas.append(a.expected_seconds_to_fill)
    # ETAs should decrease as we approach the front
    assert etas == sorted(etas, reverse=True)


def test_replay_consistency():
    a = estimate(_inputs())
    b = estimate(_inputs())
    assert a == b
