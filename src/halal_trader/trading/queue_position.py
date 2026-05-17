"""Queue-position model for limit orders — Round-5 Wave 12.J.

When a limit order rests on the book, its fill probability depends on
its position in the queue at that price level. This module ships the
**queue-position estimator + expected-fill model**: given the order's
arrival time, the size of the resting book at the limit price, and
the recent volume rate, it estimates queue position + expected time
to fill.

Pinned semantics:

- **Queue position** = ``orders_ahead_at_arrival / current_total_at_level``.
- **Expected time to fill** uses simple linear approximation:
  ``orders_ahead_size / market_volume_rate_at_level``.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueueInputs:
    """Inputs for queue-position estimation."""

    order_id: str
    arrival_size_ahead: float  # orders ahead at arrival time
    current_size_ahead: float  # orders still ahead now
    current_total_at_level: float
    market_volume_rate_at_level: float  # in size-per-second

    def __post_init__(self) -> None:
        if not self.order_id or not self.order_id.strip():
            raise ValueError("order_id must be non-empty")
        if self.arrival_size_ahead < 0:
            raise ValueError("arrival_size_ahead must be non-negative")
        if self.current_size_ahead < 0:
            raise ValueError("current_size_ahead must be non-negative")
        if self.current_total_at_level <= 0:
            raise ValueError("current_total_at_level must be positive")
        if self.market_volume_rate_at_level < 0:
            raise ValueError("market_volume_rate_at_level must be non-negative")
        if self.current_size_ahead > self.arrival_size_ahead + 1e-9:
            raise ValueError(
                "current_size_ahead cannot exceed arrival_size_ahead "
                "(queue can only shrink for a resting order)"
            )


@dataclass(frozen=True)
class QueueAssessment:
    """Result of queue-position estimation."""

    order_id: str
    queue_position: float  # 0.0 = front, 1.0 = back
    fraction_consumed: float  # how much of original ahead is gone
    expected_seconds_to_fill: float  # inf if no volume

    def __post_init__(self) -> None:
        if not 0.0 <= self.queue_position <= 1.0:
            raise ValueError("queue_position must be in [0, 1]")
        if not 0.0 <= self.fraction_consumed <= 1.0:
            raise ValueError("fraction_consumed must be in [0, 1]")
        if self.expected_seconds_to_fill < 0:
            raise ValueError("expected_seconds_to_fill must be non-negative")


def estimate(inputs: QueueInputs) -> QueueAssessment:
    """Estimate queue position + expected time to fill."""
    queue_pos = (
        inputs.current_size_ahead / inputs.current_total_at_level
        if inputs.current_total_at_level > 0
        else 0.0
    )
    queue_pos = min(1.0, queue_pos)

    if inputs.arrival_size_ahead > 0:
        consumed = (
            inputs.arrival_size_ahead - inputs.current_size_ahead
        ) / inputs.arrival_size_ahead
    else:
        consumed = 1.0  # already at front
    consumed = max(0.0, min(1.0, consumed))

    if inputs.market_volume_rate_at_level <= 0:
        expected_seconds = float("inf") if inputs.current_size_ahead > 0 else 0.0
    else:
        expected_seconds = (
            inputs.current_size_ahead / inputs.market_volume_rate_at_level
        )

    return QueueAssessment(
        order_id=inputs.order_id,
        queue_position=queue_pos,
        fraction_consumed=consumed,
        expected_seconds_to_fill=expected_seconds,
    )


def render_assessment(a: QueueAssessment) -> str:
    if a.expected_seconds_to_fill == float("inf"):
        eta = "∞"
    else:
        eta = f"{a.expected_seconds_to_fill:.1f}s"
    return (
        f"📥 {a.order_id} queue_pos={a.queue_position * 100:.1f}% "
        f"consumed={a.fraction_consumed * 100:.1f}% ETA={eta}"
    )
