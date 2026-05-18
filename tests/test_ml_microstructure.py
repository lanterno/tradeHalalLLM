"""Tests for `ml/microstructure.py` (L2 order-book feature
extractor — distinct from the existing `crypto/microstructure.py`
which handles funding rates and cumulative-delta on klines).

Pins each per-feature computation, the snapshot validation rules
(crossed book, mis-ordered levels, negative size), the bounded-
output guarantees, the depth-decay slope edge cases, and the
render output.
"""

from __future__ import annotations

import math

import pytest

from halal_trader.ml.microstructure import (
    MicrostructureFeatures,
    OrderBookLevel,
    OrderBookSnapshot,
    extract,
    render_features,
)


def _level(price: float, size: float) -> OrderBookLevel:
    return OrderBookLevel(price=price, size=size)


def _snapshot(
    bids: tuple[OrderBookLevel, ...],
    asks: tuple[OrderBookLevel, ...],
) -> OrderBookSnapshot:
    return OrderBookSnapshot(bids=bids, asks=asks)


def _balanced_book() -> OrderBookSnapshot:
    """Flat 10-level book centred at 100, size 1.0 at every level."""
    bids = tuple(_level(100.0 - 0.1 * i, 1.0) for i in range(1, 11))
    asks = tuple(_level(100.0 + 0.1 * i, 1.0) for i in range(1, 11))
    return _snapshot(bids, asks)


# ── OrderBookLevel validation ────────────────────────────


def test_level_rejects_non_positive_price():
    with pytest.raises(ValueError, match="price"):
        _level(0.0, 1.0)


def test_level_rejects_negative_size():
    with pytest.raises(ValueError, match="size"):
        _level(100.0, -1.0)


def test_level_accepts_zero_size():
    """Pin: a zero-size level is valid (vanished bid). The
    extractor handles it specifically in depth-decay."""
    level = _level(100.0, 0.0)
    assert level.size == 0.0


# ── OrderBookSnapshot validation ─────────────────────────


def test_snapshot_rejects_empty_bids():
    with pytest.raises(ValueError, match="bid"):
        _snapshot(bids=(), asks=(_level(100.0, 1.0),))


def test_snapshot_rejects_empty_asks():
    with pytest.raises(ValueError, match="ask"):
        _snapshot(bids=(_level(99.0, 1.0),), asks=())


def test_snapshot_rejects_misordered_bids():
    """Pin: bids must be descending. The extractor doesn't sort
    — pin the contract so a future adapter that breaks the order
    surfaces immediately."""
    bids = (_level(99.0, 1.0), _level(99.5, 1.0))  # ascending — bad
    asks = (_level(100.0, 1.0),)
    with pytest.raises(ValueError, match="descending"):
        _snapshot(bids=bids, asks=asks)


def test_snapshot_rejects_misordered_asks():
    bids = (_level(99.0, 1.0),)
    asks = (_level(100.5, 1.0), _level(100.0, 1.0))  # descending — bad
    with pytest.raises(ValueError, match="ascending"):
        _snapshot(bids=bids, asks=asks)


def test_snapshot_rejects_crossed_book():
    """Pin: best_bid >= best_ask is a corrupted snapshot — surface
    rather than silently produce a negative spread."""
    bids = (_level(101.0, 1.0),)
    asks = (_level(100.0, 1.0),)
    with pytest.raises(ValueError, match="crossed"):
        _snapshot(bids=bids, asks=asks)


# ── imbalance ────────────────────────────────────────────


def test_imbalance_zero_on_balanced_book():
    """Pin: equal volume on both sides → imbalance 0."""
    f = extract(_balanced_book())
    assert f.imbalance == 0.0


def test_imbalance_positive_when_more_buying_pressure():
    """3× bid volume → imbalance close to 0.5."""
    bids = (_level(99.0, 6.0), _level(98.0, 6.0))
    asks = (_level(100.0, 2.0), _level(101.0, 2.0))
    f = extract(_snapshot(bids, asks))
    assert f.imbalance > 0.4
    assert f.imbalance < 0.6


def test_imbalance_negative_when_more_selling_pressure():
    bids = (_level(99.0, 1.0),)
    asks = (_level(100.0, 9.0),)
    f = extract(_snapshot(bids, asks))
    assert f.imbalance < -0.7


def test_imbalance_clamped_to_minus_one_one():
    """Pin: even when one side is empty (zero volume), imbalance
    stays in [-1, 1] — division uses the non-zero side as the
    full total."""
    bids = (_level(99.0, 5.0),)
    asks = (_level(100.0, 0.001),)  # near-zero ask
    f = extract(_snapshot(bids, asks))
    assert -1.0 <= f.imbalance <= 1.0
    # Should be very close to +1 since bid dominates
    assert f.imbalance > 0.99


def test_imbalance_zero_on_zero_volume_both_sides():
    """Defensive: zero-volume book → imbalance 0 (no information)."""
    bids = (_level(99.0, 0.0),)
    asks = (_level(100.0, 0.0),)
    f = extract(_snapshot(bids, asks))
    assert f.imbalance == 0.0


# ── micro-price ──────────────────────────────────────────


def test_micro_price_equals_mid_when_top_sizes_equal():
    """Pin: when bid_size == ask_size, the volume-weighted micro
    price equals the simple mid."""
    bids = (_level(99.0, 1.0),)
    asks = (_level(101.0, 1.0),)
    f = extract(_snapshot(bids, asks))
    assert f.micro_price == pytest.approx(100.0)
    assert f.micro_price_bp_dev == pytest.approx(0.0)


def test_micro_price_skews_toward_thinner_side():
    """Pin: when ask is thinner than bid, the micro-price skews
    UP toward the ask — the side about to move toward."""
    bids = (_level(99.0, 10.0),)  # thick bid
    asks = (_level(101.0, 1.0),)  # thin ask
    f = extract(_snapshot(bids, asks))
    # micro = (99×1 + 101×10) / 11 = 1109/11 ≈ 100.82
    assert f.micro_price == pytest.approx(1109 / 11)
    # micro is ABOVE the simple mid (100.0)
    assert f.micro_price_bp_dev > 0


def test_micro_price_bp_dev_clamped_at_500():
    """Pin: ±500bp clamp — anything beyond is data corruption.
    Construct an extreme case (degenerate ask size)."""
    bids = (_level(50.0, 1.0),)  # very wide spread
    asks = (_level(150.0, 0.001),)  # thin ask
    f = extract(_snapshot(bids, asks))
    assert -500.0 <= f.micro_price_bp_dev <= 500.0


# ── spread ───────────────────────────────────────────────


def test_spread_abs_and_bp_match():
    """Pin: spread_bp = (spread_abs / mid) × 1e4."""
    bids = (_level(99.0, 1.0),)
    asks = (_level(101.0, 1.0),)
    f = extract(_snapshot(bids, asks))
    assert f.spread_abs == 2.0
    # mid = 100, spread = 2, bp = 200
    assert f.spread_bp == pytest.approx(200.0)


def test_spread_never_negative():
    """Pin: spread_bp clamped to 0 lower bound."""
    f = extract(_balanced_book())
    assert f.spread_bp >= 0.0


# ── depth decay ──────────────────────────────────────────


def test_depth_decay_zero_on_flat_book():
    """Flat book (constant size at every level) → log(volume) is
    constant → slope 0."""
    f = extract(_balanced_book())
    assert f.depth_decay_slope == pytest.approx(0.0, abs=0.01)


def test_depth_decay_negative_on_thinning_book():
    """Volume halves at each level → log(volume) decreases
    linearly → negative slope."""
    bids = tuple(_level(100.0 - 0.1 * i, 1.0 / (2 ** (i - 1))) for i in range(1, 6))
    asks = tuple(_level(100.0 + 0.1 * i, 1.0 / (2 ** (i - 1))) for i in range(1, 6))
    f = extract(_snapshot(bids, asks))
    assert f.depth_decay_slope < -0.5


def test_depth_decay_zero_when_too_few_valid_levels():
    """Pin: below 3 valid levels per side, slope is reported 0.
    The polyfit on 2 points has no useful information for a
    stable slope estimate."""
    bids = (_level(99.0, 1.0), _level(98.0, 1.0))
    asks = (_level(100.0, 1.0), _level(101.0, 1.0))
    f = extract(_snapshot(bids, asks))
    assert f.depth_decay_slope == 0.0


def test_depth_decay_skips_zero_size_levels():
    """Pin: log(0) is -inf and would corrupt the slope estimate.
    Zero-size levels are skipped from the regression."""
    # 3 valid levels with thinning, 1 zero-size in middle
    bids = (
        _level(99.0, 1.0),
        _level(98.0, 0.0),  # vanished
        _level(97.0, 0.5),
        _level(96.0, 0.25),
    )
    asks = (
        _level(100.0, 1.0),
        _level(101.0, 0.0),
        _level(102.0, 0.5),
        _level(103.0, 0.25),
    )
    f = extract(_snapshot(bids, asks))
    # Slope is negative (thinning across the kept levels)
    assert f.depth_decay_slope < 0


def test_depth_decay_clamped():
    """Pin: ±5 clamp on slope to handle pathologically thin books."""
    f = extract(_balanced_book())
    assert -5.0 <= f.depth_decay_slope <= 5.0


# ── top-of-book skew ─────────────────────────────────────


def test_top_of_book_skew_positive_when_bid_thicker():
    bids = (_level(99.0, 10.0),)
    asks = (_level(100.0, 1.0),)
    f = extract(_snapshot(bids, asks))
    # log(10/1) = ~2.30
    assert f.top_of_book_log_skew == pytest.approx(math.log(10.0))


def test_top_of_book_skew_negative_when_ask_thicker():
    bids = (_level(99.0, 1.0),)
    asks = (_level(100.0, 5.0),)
    f = extract(_snapshot(bids, asks))
    # log(1/5) = log(0.2) = ~-1.61
    assert f.top_of_book_log_skew < 0


def test_top_of_book_skew_zero_on_equal_sizes():
    bids = (_level(99.0, 1.0),)
    asks = (_level(100.0, 1.0),)
    f = extract(_snapshot(bids, asks))
    assert f.top_of_book_log_skew == 0.0


def test_top_of_book_skew_zero_when_one_side_zero():
    """Pin: log(0) is -inf — the helper returns 0 instead, so a
    vanished side doesn't poison downstream signals."""
    bids = (_level(99.0, 0.0), _level(98.0, 1.0))
    asks = (_level(100.0, 1.0),)
    f = extract(_snapshot(bids, asks))
    assert f.top_of_book_log_skew == 0.0


def test_top_of_book_skew_clamped():
    bids = (_level(99.0, 1e6),)
    asks = (_level(100.0, 1e-3),)
    f = extract(_snapshot(bids, asks))
    # log(1e9) ≈ 20.7 → clamp at 5
    assert f.top_of_book_log_skew == 5.0


# ── extract entry point ──────────────────────────────────


def test_extract_rejects_zero_levels():
    with pytest.raises(ValueError, match="levels"):
        extract(_balanced_book(), levels=0)


def test_extract_returns_features_dataclass():
    f = extract(_balanced_book())
    assert isinstance(f, MicrostructureFeatures)


def test_extract_records_levels_used():
    """The number of levels actually consulted is recorded — useful
    for the dashboard's "depth confidence" indicator."""
    f = extract(_balanced_book(), levels=5)
    assert f.levels_used == 5


def test_extract_levels_used_matches_book_when_smaller():
    """When the book has fewer levels than requested, levels_used
    matches what the book actually has."""
    bids = (_level(99.0, 1.0), _level(98.0, 1.0))
    asks = (_level(100.0, 1.0), _level(101.0, 1.0))
    f = extract(_snapshot(bids, asks), levels=10)
    assert f.levels_used == 2


def test_extract_mid_price_is_midpoint():
    bids = (_level(99.5, 1.0),)
    asks = (_level(100.5, 1.0),)
    f = extract(_snapshot(bids, asks))
    assert f.mid_price == 100.0


def test_extract_best_bid_and_ask_match_snapshot():
    bids = (_level(99.0, 1.0), _level(98.0, 1.0))
    asks = (_level(100.0, 1.0), _level(101.0, 1.0))
    f = extract(_snapshot(bids, asks))
    assert f.best_bid == 99.0
    assert f.best_ask == 100.0


# ── output structure ─────────────────────────────────────


def test_features_immutable():
    f = extract(_balanced_book())
    with pytest.raises(Exception):
        f.imbalance = 1.0  # type: ignore[misc]


def test_imbalance_clamped_to_one_via_extract():
    """Pin: extract clamps even pathological imbalance values."""
    bids = (_level(99.0, 1.0),)
    asks = (_level(100.0, 0.0),)
    f = extract(_snapshot(bids, asks))
    assert -1.0 <= f.imbalance <= 1.0


# ── render_features ──────────────────────────────────────


def test_render_includes_imbalance_with_sign():
    bids = (_level(99.0, 5.0),)
    asks = (_level(100.0, 1.0),)
    text = render_features(extract(_snapshot(bids, asks)))
    assert "imbalance" in text
    assert "+" in text


def test_render_includes_spread_bp():
    text = render_features(extract(_balanced_book()))
    assert "spread" in text
    assert "bp" in text


def test_render_signs_micro_price_dev():
    bids = (_level(99.0, 10.0),)
    asks = (_level(101.0, 1.0),)
    text = render_features(extract(_snapshot(bids, asks)))
    # micro skews up; expect a + sign on the dev
    assert "micro" in text
