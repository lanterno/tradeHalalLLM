"""Tests for `core/position_story.py` (per-position story
aggregator).

Pins the P&L curve arithmetic, the indicator-delta partial-data
contract, the bullish/bearish news classification, the markdown
rendering shape, and the empty / partial-input degradation paths
(no price ticks → no current price; no entry indicators → no
deltas; legacy positions without rationale).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halal_trader.core.position_story import (
    IndicatorObservation,
    NewsEvent,
    PositionStory,
    PositionStoryInput,
    PriceObservation,
    build_story,
)


def _input(**overrides) -> PositionStoryInput:
    base = {
        "pair": "BTCUSDT",
        "side": "buy",
        "quantity": 0.5,
        "entry_price": 60_000.0,
        "entry_at": datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
        "stop_loss": 58_000.0,
        "take_profit": 65_000.0,
        "trailing_stop_pct": None,
        "llm_reasoning": "RSI bounce + volume confirmation.",
        "confidence": 0.7,
        "prompt_version": "crypto.strategy@a1b2",
        "entry_indicators": IndicatorObservation(
            at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC),
            rsi_14=35.0,
            macd_histogram=0.001,
            volume_ratio=1.4,
            atr_14=300.0,
            bb_position=0.2,
        ),
        "indicator_timeline": [
            IndicatorObservation(
                at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
                rsi_14=42.0,
                macd_histogram=0.002,
                volume_ratio=1.5,
                atr_14=305.0,
                bb_position=0.3,
            ),
            IndicatorObservation(
                at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
                rsi_14=55.0,
                macd_histogram=0.004,
                volume_ratio=1.6,
                atr_14=310.0,
                bb_position=0.45,
            ),
        ],
        "price_timeline": [
            PriceObservation(at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC), price=60_300.0),
            PriceObservation(at=datetime(2026, 5, 1, 12, 0, tzinfo=UTC), price=60_900.0),
            PriceObservation(at=datetime(2026, 5, 1, 13, 0, tzinfo=UTC), price=61_500.0),
        ],
        "news": [
            NewsEvent(
                at=datetime(2026, 5, 1, 10, 30, tzinfo=UTC),
                headline="ETF inflows spike",
                source="cryptopanic",
                score=0.5,
                url="https://example.com/a",
            ),
            NewsEvent(
                at=datetime(2026, 5, 1, 11, 15, tzinfo=UTC),
                headline="Regulator weighs new rule",
                source="cryptopanic",
                score=-0.3,
            ),
        ],
    }
    base.update(overrides)
    return PositionStoryInput(**base)


# ── P&L curve ─────────────────────────────────────────────


def test_pnl_curve_has_one_point_per_price_observation():
    story = build_story(_input())
    assert len(story.pnl_curve) == 3


def test_pnl_curve_first_point_uses_first_observation():
    story = build_story(_input())
    # entry 60_000 → price 60_300 = +0.5%, qty 0.5 → +$150
    assert story.pnl_curve[0].unrealized_pct == 0.005
    assert story.pnl_curve[0].unrealized_usd == 150.0


def test_pnl_curve_handles_negative_returns():
    inp = _input(
        price_timeline=[
            PriceObservation(at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC), price=58_500.0),
        ]
    )
    story = build_story(inp)
    # entry 60_000 → 58_500 = -2.5%
    assert story.pnl_curve[0].unrealized_pct == -0.025


def test_pnl_curve_empty_when_no_price_observations():
    """No ticks yet → curve empty + current_* None. Pin so the
    dashboard 'no data yet' state renders cleanly."""
    inp = _input(price_timeline=[])
    story = build_story(inp)
    assert story.pnl_curve == []
    assert story.current_price is None
    assert story.current_unrealized_pct is None


def test_pnl_curve_empty_when_entry_price_zero_or_negative():
    """Defensive: a zero entry price would divide by zero. Pin so
    a corrupt row doesn't NaN-out the dashboard."""
    inp = _input(entry_price=0.0)
    story = build_story(inp)
    assert story.pnl_curve == []


def test_current_unrealized_uses_last_price():
    """The story's `current_*` fields must come from the LAST
    price observation, not the first or an average."""
    story = build_story(_input())
    # entry 60_000 → 61_500 = +2.5% × 0.5 qty = +$750
    assert story.current_unrealized_pct == 0.025
    assert story.current_unrealized_usd == 750.0
    assert story.current_price == 61_500.0


# ── indicator deltas ──────────────────────────────────────


def test_deltas_compare_entry_vs_latest_observation():
    story = build_story(_input())
    # Default entry RSI 35 → latest 55 → +20
    rsi_delta = next(d for d in story.deltas if d.name == "rsi_14")
    assert rsi_delta.entry_value == 35.0
    assert rsi_delta.latest_value == 55.0
    assert rsi_delta.delta == 20.0


def test_deltas_skip_field_when_both_sides_none():
    """A field that was never measured shouldn't appear at all —
    fewer rows in the table beats a wall of n/a."""
    inp = _input(
        entry_indicators=IndicatorObservation(
            at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC), rsi_14=40.0
        ),
        indicator_timeline=[
            IndicatorObservation(at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC), rsi_14=50.0)
        ],
    )
    story = build_story(inp)
    names = {d.name for d in story.deltas}
    # Only RSI was measured on either side.
    assert names == {"rsi_14"}


def test_deltas_emit_partial_when_one_side_missing():
    """If we have entry but no latest, still emit the row with
    delta=None. The dashboard wants 'we measured this once' to be
    visible."""
    inp = _input(
        entry_indicators=IndicatorObservation(
            at=datetime(2026, 5, 1, 10, 0, tzinfo=UTC), rsi_14=40.0
        ),
        indicator_timeline=[
            IndicatorObservation(
                at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
                # rsi_14 missing on this snapshot
                macd_histogram=0.001,
            )
        ],
    )
    story = build_story(inp)
    rsi_d = next(d for d in story.deltas if d.name == "rsi_14")
    assert rsi_d.entry_value == 40.0
    assert rsi_d.latest_value is None
    assert rsi_d.delta is None


def test_deltas_empty_when_no_indicator_data_at_all():
    inp = _input(entry_indicators=None, indicator_timeline=[])
    story = build_story(inp)
    assert story.deltas == []


# ── news classification ──────────────────────────────────


def test_news_count_includes_all_events():
    story = build_story(_input())
    assert story.news_count == 2


def test_bullish_and_bearish_classify_by_score_sign():
    """Score > 0 → bullish; score < 0 → bearish; score == 0 → neither
    (pin so a neutral headline doesn't double-count)."""
    inp = _input(
        news=[
            NewsEvent(
                at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
                headline="positive",
                source="x",
                score=0.4,
            ),
            NewsEvent(
                at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
                headline="negative",
                source="x",
                score=-0.4,
            ),
            NewsEvent(
                at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC),
                headline="neutral",
                source="x",
                score=0.0,
            ),
        ]
    )
    story = build_story(inp)
    assert story.news_count == 3
    assert story.bullish_news == 1
    assert story.bearish_news == 1


def test_no_news_yields_zero_counts():
    story = build_story(_input(news=[]))
    assert story.news_count == 0
    assert story.bullish_news == 0
    assert story.bearish_news == 0


# ── markdown rendering ───────────────────────────────────


def test_markdown_includes_pair_and_side():
    story = build_story(_input())
    assert "BTCUSDT" in story.markdown
    assert "BUY" in story.markdown


def test_markdown_uses_green_emoji_when_in_profit():
    story = build_story(_input())
    assert "🟢" in story.markdown


def test_markdown_uses_red_emoji_when_underwater():
    inp = _input(
        price_timeline=[
            PriceObservation(at=datetime(2026, 5, 1, 11, 0, tzinfo=UTC), price=58_000.0)
        ]
    )
    story = build_story(inp)
    assert "🔴" in story.markdown


def test_markdown_includes_rationale_section_when_present():
    story = build_story(_input())
    assert "## Why we entered" in story.markdown
    assert "RSI bounce" in story.markdown


def test_markdown_truncates_long_rationale():
    long = "Very long rationale. " * 100
    story = build_story(_input(llm_reasoning=long))
    rationale_lines = [line for line in story.markdown.split("\n") if line.startswith("> ")]
    assert rationale_lines
    # 400-char cap + "…" appended.
    assert len(rationale_lines[0]) < 450


def test_markdown_omits_rationale_when_none():
    inp = _input(llm_reasoning=None)
    story = build_story(inp)
    assert "## Why we entered" not in story.markdown


def test_markdown_includes_indicator_drift_table():
    story = build_story(_input())
    assert "## Indicator drift since entry" in story.markdown
    assert "rsi_14" in story.markdown


def test_markdown_includes_news_section_with_counts():
    story = build_story(_input())
    assert "1 bullish" in story.markdown
    assert "1 bearish" in story.markdown


def test_markdown_lists_most_recent_news_first():
    inp = _input(
        news=[
            NewsEvent(
                at=datetime(2026, 5, 1, 10, 30, tzinfo=UTC),
                headline="OLD HEADLINE",
                source="x",
                score=0.1,
            ),
            NewsEvent(
                at=datetime(2026, 5, 1, 12, 30, tzinfo=UTC),
                headline="NEWER HEADLINE",
                source="x",
                score=0.2,
            ),
        ]
    )
    story = build_story(inp)
    # Find the indices of each headline in the markdown.
    md = story.markdown
    newer_idx = md.find("NEWER HEADLINE")
    older_idx = md.find("OLD HEADLINE")
    assert newer_idx != -1 and older_idx != -1
    assert newer_idx < older_idx


def test_markdown_includes_pnl_track_section():
    story = build_story(_input())
    assert "## P&L track" in story.markdown
    assert "Trough" in story.markdown
    assert "Peak" in story.markdown


def test_markdown_includes_risk_levels_when_present():
    story = build_story(_input())
    assert "Risk levels:" in story.markdown
    assert "SL" in story.markdown
    assert "TP" in story.markdown


def test_markdown_omits_risk_levels_section_when_none_set():
    inp = _input(stop_loss=None, take_profit=None, trailing_stop_pct=None)
    story = build_story(inp)
    assert "Risk levels:" not in story.markdown


def test_markdown_includes_trailing_stop_when_set():
    inp = _input(stop_loss=None, take_profit=None, trailing_stop_pct=0.005)
    story = build_story(inp)
    assert "trail" in story.markdown


def test_markdown_includes_prompt_version_when_set():
    """The prompt-version SHA is the audit-trail key — pin its
    presence in the operator-facing card."""
    story = build_story(_input())
    assert "crypto.strategy@a1b2" in story.markdown


# ── output structure ─────────────────────────────────────


def test_story_is_immutable():
    story = build_story(_input())
    assert isinstance(story, PositionStory)
    try:
        story.pair = "tampered"  # type: ignore[misc]
        raise AssertionError("frozen dataclass should reject mutation")
    except Exception:
        pass


def test_pnl_points_carry_full_breakdown():
    story = build_story(_input())
    p = story.pnl_curve[0]
    assert p.price == 60_300.0
    assert p.unrealized_pct == 0.005
    assert p.unrealized_usd == 150.0


def test_position_with_no_data_still_produces_valid_story():
    """Cold-start case: position just opened, no ticks, no
    indicator follow-ups, no news. Must still render cleanly."""
    inp = _input(
        indicator_timeline=[],
        price_timeline=[],
        news=[],
        entry_indicators=None,
        llm_reasoning=None,
        confidence=None,
        prompt_version=None,
    )
    story = build_story(inp)
    assert story.deltas == []
    assert story.pnl_curve == []
    assert story.news_count == 0
    assert story.markdown  # non-empty


def test_legacy_position_without_rationale_renders():
    """A pre-LLM trade row without `llm_reasoning` shouldn't crash
    the renderer."""
    inp = _input(llm_reasoning=None, confidence=None, prompt_version=None)
    story = build_story(inp)
    assert "## Why we entered" not in story.markdown
    # but the rest of the story still rendered
    assert "## Indicator drift" in story.markdown


# ── duration sanity ──────────────────────────────────────


def test_story_handles_long_lived_position():
    """A position open for a week with hourly ticks shouldn't blow
    up. Smoke test on a large input."""
    entry_at = datetime(2026, 5, 1, tzinfo=UTC)
    prices = [
        PriceObservation(
            at=entry_at + timedelta(hours=i),
            price=60_000.0 + i * 10,
        )
        for i in range(168)
    ]
    inp = _input(price_timeline=prices)
    story = build_story(inp)
    assert len(story.pnl_curve) == 168
    # Last point: entry + 167*10 = 61_670 → +2.78%
    assert story.current_unrealized_pct > 0.025
