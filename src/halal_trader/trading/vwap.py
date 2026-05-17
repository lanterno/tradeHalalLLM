"""Volume-Weighted Average Price (VWAP) execution algorithm — Round-5 Wave 12.B.

VWAP slices a parent order in proportion to historical / forecasted
volume profile across the trading day. The intuition is "move with
the crowd": when 5% of the day's volume happens between 9:30–9:35,
that bucket gets 5% of the parent quantity. Standard execution algo
for institutional flow that doesn't want to leak its presence.

This module ships the **slicer**: given parent quantity, a volume
profile (per-bucket fractions), and the bucket times, it returns a
deterministic schedule of child orders. Broker dispatch + fill
reconciliation live one layer up.

Pinned semantics:

- **Volume-profile fractions sum to 1.0 within tolerance.** A profile
  that doesn't normalise is rejected at construction.
- **Each bucket is (start_time, fraction).** Children submit at the
  bucket's start.
- **Closed-set Side ladder** (BUY / SELL) — re-exports `trading.twap.Side`.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from halal_trader.trading.twap import Side


@dataclass(frozen=True)
class VolumeBucket:
    """A single (start_time, expected-fraction-of-day-volume) bucket."""

    start_time: datetime
    fraction: float

    def __post_init__(self) -> None:
        if self.start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")


@dataclass(frozen=True)
class VolumeProfile:
    """A normalised volume profile — sequence of buckets summing to ~1.0."""

    buckets: tuple[VolumeBucket, ...]
    sum_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("profile must have at least one bucket")
        starts = [b.start_time for b in self.buckets]
        if starts != sorted(starts):
            raise ValueError("buckets must be sorted by start_time")
        total = sum(b.fraction for b in self.buckets)
        if abs(total - 1.0) > self.sum_tolerance:
            raise ValueError(
                f"profile fractions must sum to 1.0 (within {self.sum_tolerance}); got {total}"
            )

    def __len__(self) -> int:
        return len(self.buckets)


@dataclass(frozen=True)
class VwapPolicy:
    """Operator-tunable VWAP policy."""

    min_slice_quantity: float = 0.0
    max_buckets: int = 1000
    skip_zero_fraction: bool = True

    def __post_init__(self) -> None:
        if self.min_slice_quantity < 0:
            raise ValueError("min_slice_quantity must be non-negative")
        if self.max_buckets <= 0:
            raise ValueError("max_buckets must be positive")


@dataclass(frozen=True)
class VwapInputs:
    """Inputs for a VWAP slice."""

    parent_id: str
    symbol: str
    side: Side
    parent_quantity: float
    profile: VolumeProfile

    def __post_init__(self) -> None:
        if not self.parent_id or not self.parent_id.strip():
            raise ValueError("parent_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.parent_quantity <= 0:
            raise ValueError("parent_quantity must be positive")


@dataclass(frozen=True)
class ChildOrder:
    parent_id: str
    bucket_index: int
    symbol: str
    side: Side
    quantity: float
    submit_time: datetime

    def __post_init__(self) -> None:
        if self.bucket_index < 0:
            raise ValueError("bucket_index must be non-negative")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.submit_time.tzinfo is None:
            raise ValueError("submit_time must be timezone-aware")


def slice_vwap(
    inputs: VwapInputs, *, policy: VwapPolicy | None = None
) -> tuple[ChildOrder, ...]:
    """Slice a parent order in proportion to the volume profile."""
    pol = policy if policy is not None else VwapPolicy()
    if len(inputs.profile) > pol.max_buckets:
        raise ValueError(f"profile has {len(inputs.profile)} buckets > max {pol.max_buckets}")

    children: list[ChildOrder] = []
    cumulative = 0.0
    for i, bucket in enumerate(inputs.profile.buckets):
        qty = inputs.parent_quantity * bucket.fraction
        if qty <= 0 and pol.skip_zero_fraction:
            continue
        if 0 < qty < pol.min_slice_quantity:
            # Roll forward into the next bucket; here we just skip + accumulate
            # the un-allocated fraction. The remainder is added at the end.
            continue
        children.append(
            ChildOrder(
                parent_id=inputs.parent_id,
                bucket_index=i,
                symbol=inputs.symbol,
                side=inputs.side,
                quantity=qty,
                submit_time=bucket.start_time,
            )
        )
        cumulative += qty

    # Distribute any rounding remainder into the largest bucket
    remainder = inputs.parent_quantity - cumulative
    if children and abs(remainder) > 1e-9:
        # Find largest bucket and adjust by the residual.
        biggest_idx = max(range(len(children)), key=lambda i: children[i].quantity)
        adjusted = list(children)
        original = adjusted[biggest_idx]
        new_qty = original.quantity + remainder
        if new_qty <= 0:
            raise ValueError("rounding correction would yield non-positive quantity")
        adjusted[biggest_idx] = ChildOrder(
            parent_id=original.parent_id,
            bucket_index=original.bucket_index,
            symbol=original.symbol,
            side=original.side,
            quantity=new_qty,
            submit_time=original.submit_time,
        )
        children = adjusted

    return tuple(children)


def cumulative_quantity(children: Iterable[ChildOrder]) -> float:
    return sum(c.quantity for c in children)


def filter_due(children: Iterable[ChildOrder], *, now: datetime) -> tuple[ChildOrder, ...]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return tuple(c for c in children if c.submit_time <= now)


# --- Build common volume profiles ------------------------------------------


def uniform_profile(buckets: Iterable[datetime]) -> VolumeProfile:
    """Build a flat profile (each bucket gets equal share)."""
    bucket_list = sorted(buckets)
    if not bucket_list:
        raise ValueError("at least one bucket required")
    n = len(bucket_list)
    fraction = 1.0 / n
    return VolumeProfile(
        buckets=tuple(VolumeBucket(start_time=t, fraction=fraction) for t in bucket_list),
    )


def u_shape_profile(buckets: Iterable[datetime]) -> VolumeProfile:
    """Build a U-shape profile (heavier at open + close).

    Standard equity-market intraday distribution. The implementation
    weights buckets by a parabolic ``(2x-1)^2 + 0.4``, then normalises.
    """
    bucket_list = sorted(buckets)
    n = len(bucket_list)
    if n == 0:
        raise ValueError("at least one bucket required")
    if n == 1:
        return VolumeProfile(buckets=(VolumeBucket(bucket_list[0], 1.0),))
    raw_weights = []
    for i in range(n):
        x = i / (n - 1)
        raw_weights.append((2 * x - 1) ** 2 + 0.4)
    total = sum(raw_weights)
    fractions = [w / total for w in raw_weights]
    return VolumeProfile(
        buckets=tuple(
            VolumeBucket(start_time=t, fraction=f)
            for t, f in zip(bucket_list, fractions)
        ),
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_schedule(children: tuple[ChildOrder, ...]) -> str:
    if not children:
        return "VWAP schedule: empty"
    head = (
        f"VWAP {children[0].parent_id} {children[0].symbol} {children[0].side.value}: "
        f"{len(children)} buckets, total={cumulative_quantity(children):.4f}"
    )
    lines = [head]
    for c in children:
        lines.append(
            f"  • bucket {c.bucket_index}: {c.quantity:.4f} @ "
            f"{c.submit_time.isoformat()}"
        )
    return _scrub("\n".join(lines))
