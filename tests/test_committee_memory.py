"""Tests for core/committee_memory.py — Round-5 Wave 8.C."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from halal_trader.core.committee_memory import (
    InMemoryStore,
    MemoryBias,
    MemoryEntry,
    OutcomeLabel,
    RegimeTag,
    RSIBucket,
    SetupFingerprint,
    VolumeBucket,
    bias_for_fingerprint,
    render_bias,
    rsi_to_bucket,
    volume_to_bucket,
)
from halal_trader.core.llm_committee import Stance


def _fp(
    regime: RegimeTag = RegimeTag.BULL_TREND,
    rsi_bucket: RSIBucket = RSIBucket.NEUTRAL,
    macd_positive: bool = True,
    volume_bucket: VolumeBucket = VolumeBucket.NORMAL,
    side: Stance = Stance.BUY,
    sector: str = "tech",
) -> SetupFingerprint:
    return SetupFingerprint(
        regime=regime,
        rsi_bucket=rsi_bucket,
        macd_positive=macd_positive,
        volume_bucket=volume_bucket,
        side=side,
        sector=sector,
    )


def _entry(
    fp: SetupFingerprint | None = None,
    decision_date: date = date(2026, 5, 1),
    stance: Stance = Stance.BUY,
    confidence: float = 0.7,
    outcome: OutcomeLabel = OutcomeLabel.WIN,
    return_pct: float = 0.05,
) -> MemoryEntry:
    return MemoryEntry(
        fingerprint=fp if fp is not None else _fp(),
        decision_date=decision_date,
        stance=stance,
        confidence=confidence,
        outcome=outcome,
        return_pct=return_pct,
    )


# --- Bucket helpers -----------------------------------------------------


def test_rsi_buckets():
    assert rsi_to_bucket(20) is RSIBucket.OVERSOLD
    assert rsi_to_bucket(40) is RSIBucket.NEUTRAL_LOW
    assert rsi_to_bucket(50) is RSIBucket.NEUTRAL
    assert rsi_to_bucket(65) is RSIBucket.NEUTRAL_HIGH
    assert rsi_to_bucket(80) is RSIBucket.OVERBOUGHT


def test_volume_buckets():
    assert volume_to_bucket(0.5) is VolumeBucket.LOW
    assert volume_to_bucket(1.0) is VolumeBucket.NORMAL
    assert volume_to_bucket(2.0) is VolumeBucket.HIGH


# --- SetupFingerprint validation ----------------------------------------


def test_fingerprint_valid():
    fp = _fp()
    assert fp.regime is RegimeTag.BULL_TREND


def test_fingerprint_empty_sector_rejected():
    with pytest.raises(ValueError):
        _fp(sector=" ")


def test_fingerprint_digest_stable():
    """Same fields → same digest."""
    fp1 = _fp()
    fp2 = _fp()
    assert fp1.digest() == fp2.digest()


def test_fingerprint_digest_changes_with_field():
    fp1 = _fp(regime=RegimeTag.BULL_TREND)
    fp2 = _fp(regime=RegimeTag.BEAR_TREND)
    assert fp1.digest() != fp2.digest()


def test_fingerprint_digest_length():
    fp = _fp()
    assert len(fp.digest()) == 16


def test_fingerprint_immutable():
    fp = _fp()
    with pytest.raises(AttributeError):
        fp.macd_positive = False  # type: ignore[misc]


# --- MemoryEntry validation ---------------------------------------------


def test_entry_valid():
    e = _entry()
    assert e.outcome is OutcomeLabel.WIN


def test_entry_invalid_confidence():
    with pytest.raises(ValueError):
        _entry(confidence=1.5)


def test_entry_open_with_return_rejected():
    """Pin: OPEN entries must have return_pct=0."""
    with pytest.raises(ValueError):
        MemoryEntry(
            fingerprint=_fp(),
            decision_date=date(2026, 5, 1),
            stance=Stance.BUY,
            confidence=0.7,
            outcome=OutcomeLabel.OPEN,
            return_pct=0.05,
        )


def test_entry_win_with_negative_return_rejected():
    with pytest.raises(ValueError):
        _entry(outcome=OutcomeLabel.WIN, return_pct=-0.05)


def test_entry_loss_with_positive_return_rejected():
    with pytest.raises(ValueError):
        _entry(outcome=OutcomeLabel.LOSS, return_pct=0.05)


def test_entry_unreasonable_return_rejected():
    with pytest.raises(ValueError):
        _entry(outcome=OutcomeLabel.WIN, return_pct=10.0)


# --- InMemoryStore --------------------------------------------------------


def test_store_insert_and_query():
    store = InMemoryStore()
    e = _entry()
    store.insert(e)
    digest = e.fingerprint.digest()
    out = store.query(digest)
    assert len(out) == 1
    assert out[0].fingerprint.digest() == digest


def test_store_query_newest_first():
    store = InMemoryStore()
    fp = _fp()
    old = _entry(fp=fp, decision_date=date(2026, 1, 1))
    new = _entry(fp=fp, decision_date=date(2026, 5, 1))
    store.insert(old)
    store.insert(new)
    out = store.query(fp.digest())
    assert out[0].decision_date == date(2026, 5, 1)
    assert out[1].decision_date == date(2026, 1, 1)


def test_store_query_k_limit():
    store = InMemoryStore()
    fp = _fp()
    for i in range(5):
        store.insert(_entry(fp=fp, decision_date=date(2026, 1, 1) + timedelta(days=i)))
    out = store.query(fp.digest(), k=3)
    assert len(out) == 3


def test_store_query_filters_by_digest():
    store = InMemoryStore()
    fp_a = _fp(sector="tech")
    fp_b = _fp(sector="energy")
    store.insert(_entry(fp=fp_a))
    store.insert(_entry(fp=fp_b))
    out_a = store.query(fp_a.digest())
    assert len(out_a) == 1
    assert out_a[0].fingerprint.sector == "tech"


def test_store_update_outcome():
    store = InMemoryStore()
    fp = _fp()
    e = MemoryEntry(
        fingerprint=fp,
        decision_date=date(2026, 5, 1),
        stance=Stance.BUY,
        confidence=0.7,
        outcome=OutcomeLabel.OPEN,
        return_pct=0.0,
    )
    store.insert(e)
    n = store.update_outcome(fp.digest(), date(2026, 5, 1), OutcomeLabel.WIN, 0.10)
    assert n == 1
    out = store.query(fp.digest())
    assert out[0].outcome is OutcomeLabel.WIN
    assert out[0].return_pct == 0.10


def test_store_update_outcome_no_match():
    store = InMemoryStore()
    n = store.update_outcome("nonexistent", date(2026, 5, 1), OutcomeLabel.WIN, 0.10)
    assert n == 0


def test_store_all_entries():
    store = InMemoryStore()
    store.insert(_entry())
    store.insert(_entry(decision_date=date(2026, 6, 1)))
    assert len(store.all_entries()) == 2


# --- bias_for_fingerprint -----------------------------------------------


def test_bias_no_entries_returns_zeros():
    store = InMemoryStore()
    fp = _fp()
    bias = bias_for_fingerprint(store, fp, today=date(2026, 5, 1))
    assert bias.n_total == 0
    assert bias.n_effective == 0.0
    assert bias.win_rate == 0.0
    assert bias.last_seen is None


def test_bias_only_open_entries_returns_zero_effective():
    store = InMemoryStore()
    fp = _fp()
    store.insert(
        MemoryEntry(
            fingerprint=fp,
            decision_date=date(2026, 5, 1),
            stance=Stance.BUY,
            confidence=0.7,
            outcome=OutcomeLabel.OPEN,
        )
    )
    bias = bias_for_fingerprint(store, fp, today=date(2026, 5, 5))
    assert bias.n_total == 1
    assert bias.n_open == 1
    assert bias.n_effective == 0.0


def test_bias_recent_dominates_old():
    """Pin: half-life decay → recent entries dominate."""
    store = InMemoryStore()
    fp = _fp()
    # 4 wins 1 year ago, 1 loss yesterday — with HL=60d, the recent
    # loss should drag win-rate well below 80%.
    for _ in range(4):
        store.insert(
            _entry(
                fp=fp,
                decision_date=date(2025, 5, 1),
                outcome=OutcomeLabel.WIN,
                return_pct=0.05,
            )
        )
    store.insert(
        _entry(
            fp=fp,
            decision_date=date(2026, 4, 30),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    )
    bias = bias_for_fingerprint(store, fp, today=date(2026, 5, 1), half_life_days=60)
    # Without decay: 4/5 = 80%. With 365d ago halving 6×: w_old≈0.016 each.
    # Effective: 4×0.016=0.064 + 1.0=1.064; win_w=0.064 → 6%.
    assert bias.win_rate < 0.20


def test_bias_avg_return_weighted():
    store = InMemoryStore()
    fp = _fp()
    store.insert(
        _entry(
            fp=fp,
            decision_date=date(2026, 5, 1),
            outcome=OutcomeLabel.WIN,
            return_pct=0.10,
        )
    )
    store.insert(
        _entry(
            fp=fp,
            decision_date=date(2026, 5, 1),
            outcome=OutcomeLabel.LOSS,
            return_pct=-0.05,
        )
    )
    bias = bias_for_fingerprint(store, fp, today=date(2026, 5, 1))
    # Same date → equal weights → avg = (0.10 - 0.05)/2 = 0.025.
    assert abs(bias.avg_return - 0.025) < 1e-9


def test_bias_invalid_half_life_rejected():
    store = InMemoryStore()
    fp = _fp()
    with pytest.raises(ValueError):
        bias_for_fingerprint(store, fp, today=date(2026, 5, 1), half_life_days=0)


def test_bias_n_open_counted_separately():
    store = InMemoryStore()
    fp = _fp()
    store.insert(_entry(fp=fp, outcome=OutcomeLabel.WIN))
    store.insert(
        MemoryEntry(
            fingerprint=fp,
            decision_date=date(2026, 5, 1),
            stance=Stance.BUY,
            confidence=0.7,
            outcome=OutcomeLabel.OPEN,
        )
    )
    bias = bias_for_fingerprint(store, fp, today=date(2026, 5, 1))
    assert bias.n_open == 1
    assert bias.n_total == 2


def test_bias_last_seen_is_newest():
    store = InMemoryStore()
    fp = _fp()
    store.insert(_entry(fp=fp, decision_date=date(2026, 1, 1)))
    store.insert(_entry(fp=fp, decision_date=date(2026, 5, 1)))
    bias = bias_for_fingerprint(store, fp, today=date(2026, 5, 5))
    assert bias.last_seen == date(2026, 5, 1)


def test_bias_significant_threshold():
    bias = MemoryBias(
        fingerprint_digest="abc",
        n_total=5,
        n_effective=2.5,
        win_rate=0.6,
        avg_return=0.02,
        last_seen=date(2026, 5, 1),
        n_open=0,
    )
    assert not bias.is_significant(min_n_effective=3.0)
    assert bias.is_significant(min_n_effective=2.0)


# --- Render --------------------------------------------------------------


def test_render_no_entries():
    bias = MemoryBias(
        fingerprint_digest="abc",
        n_total=0,
        n_effective=0.0,
        win_rate=0.0,
        avg_return=0.0,
        last_seen=None,
        n_open=0,
    )
    out = render_bias(bias)
    assert "no prior entries" in out


def test_render_with_entries():
    bias = MemoryBias(
        fingerprint_digest="abc",
        n_total=5,
        n_effective=3.5,
        win_rate=0.6,
        avg_return=0.02,
        last_seen=date(2026, 5, 1),
        n_open=1,
    )
    out = render_bias(bias)
    assert "🧠" in out
    assert "win_rate" in out
    assert "60.00%" in out
