"""Tests for mention velocity / novelty."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from halal_trader.sentiment.velocity import (
    Mention,
    compute_velocity,
    filter_halal_mentions,
    format_velocity_for_prompt,
)


def _t(hours_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


def _mentions_for(sym: str, hours_offsets: list[float]) -> list[Mention]:
    return [Mention(symbol=sym, timestamp=_t(h)) for h in hours_offsets]


# ── Compute ──────────────────────────────────────────────────────


def test_velocity_surge_when_recent_dominant() -> None:
    mentions = _mentions_for("BTC", [0.5, 1.0, 2.0, 3.0, 4.0])  # all in recent window
    out = compute_velocity(mentions, recent_window_hours=6, older_window_hours=24)
    assert "BTC" in out
    r = out["BTC"]
    assert r.n_recent == 5
    assert r.n_older == 0
    assert r.label == "surge"
    assert r.velocity >= 2.0


def test_velocity_decay_when_recent_quiet() -> None:
    mentions = _mentions_for("ETH", [10.0, 12.0, 14.0, 18.0, 22.0, 23.0])  # all older
    out = compute_velocity(mentions, recent_window_hours=6, older_window_hours=24)
    r = out["ETH"]
    assert r.n_recent == 0
    assert r.n_older == 6
    assert r.velocity == 0.0
    assert r.label == "decay"


def test_velocity_neutral_balanced() -> None:
    mentions = _mentions_for("SOL", [0.5, 1.0, 8.0, 12.0])  # 2 recent, 2 older
    out = compute_velocity(mentions, recent_window_hours=6, older_window_hours=24)
    r = out["SOL"]
    assert r.label == "neutral"
    assert r.velocity == 1.0


def test_velocity_drops_old_outside_window() -> None:
    mentions = [
        Mention(symbol="X", timestamp=_t(0.5)),
        Mention(symbol="X", timestamp=_t(72)),  # outside the older window — dropped
    ]
    out = compute_velocity(mentions, older_window_hours=24)
    assert "X" in out
    assert out["X"].n_total == 1


def test_novelty_high_for_brand_new() -> None:
    mentions = _mentions_for("NEW", [0.5, 1.0])
    out = compute_velocity(mentions, recent_window_hours=6, older_window_hours=24)
    r = out["NEW"]
    assert r.novelty == 1.0


def test_velocity_capped_when_zero_baseline() -> None:
    # 50 recent mentions, 0 older — velocity caps at 10
    mentions = _mentions_for("HOT", [0.1] * 50)
    out = compute_velocity(mentions)
    r = out["HOT"]
    assert r.velocity == 10.0


def test_normalises_symbol_case() -> None:
    mentions = [
        Mention(symbol="btc", timestamp=_t(1)),
        Mention(symbol="BTC", timestamp=_t(2)),
    ]
    out = compute_velocity(mentions)
    assert "BTC" in out
    assert out["BTC"].n_total == 2


def test_naive_timestamp_treated_as_utc() -> None:
    naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    mentions = [Mention(symbol="X", timestamp=naive)]
    out = compute_velocity(mentions)
    assert "X" in out


# ── Halal filter ─────────────────────────────────────────────────


def test_filter_halal_mentions() -> None:
    mentions = [
        Mention(symbol="BTC", timestamp=_t(1)),
        Mention(symbol="DOGE", timestamp=_t(1)),
        Mention(symbol="ETH", timestamp=_t(2)),
    ]
    out = filter_halal_mentions(mentions, ["BTC", "ETH"])
    assert {m.symbol for m in out} == {"BTC", "ETH"}


# ── Prompt formatting ────────────────────────────────────────────


def test_format_returns_empty_when_no_surge() -> None:
    out = compute_velocity(_mentions_for("X", [10.0, 11.0]))  # all older
    text = format_velocity_for_prompt(out)
    assert text == ""


def test_format_includes_top_surges_only() -> None:
    out = {}
    out.update(compute_velocity(_mentions_for("A", [0.5] * 10)))  # surge
    out.update(compute_velocity(_mentions_for("B", [0.5] * 5)))  # surge
    out.update(compute_velocity(_mentions_for("Q", [10.0])))  # decay
    text = format_velocity_for_prompt(out, limit=5)
    assert "A" in text
    assert "B" in text
    assert "Q" not in text


def test_format_filters_low_recent_count() -> None:
    out = compute_velocity(_mentions_for("A", [0.5]))  # only 1 recent — below min
    text = format_velocity_for_prompt(out, min_recent=3)
    assert text == ""
