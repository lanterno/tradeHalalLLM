"""Tests for community/chat_moderation.py — Round-5 Wave 17.F."""

from __future__ import annotations

from datetime import datetime

import pytest

from halal_trader.community.chat_moderation import (
    ChatMessage,
    HaramCategory,
    ModerationOutcome,
    ModerationResult,
    classify,
    classify_batch,
    default_lexicon,
    filter_passing,
    render_result,
)


def _msg(
    message_id: str = "M1",
    user_id: str = "alice",
    ticker_room: str = "AAPL",
    body: str = "Strong fundamentals; staying long.",
    posted_at: datetime = datetime(2026, 5, 10, 12, 0, 0),
) -> ChatMessage:
    return ChatMessage(
        message_id=message_id,
        user_id=user_id,
        ticker_room=ticker_room,
        body=body,
        posted_at=posted_at,
    )


# --- ChatMessage validation --------------------------------------------


def test_message_valid():
    m = _msg()
    assert m.user_id == "alice"


def test_message_empty_id_rejected():
    with pytest.raises(ValueError):
        _msg(message_id="")


def test_message_empty_body_rejected():
    with pytest.raises(ValueError):
        _msg(body=" ")


def test_message_long_body_rejected():
    with pytest.raises(ValueError):
        _msg(body="x" * 1500)


def test_message_immutable():
    m = _msg()
    with pytest.raises(AttributeError):
        m.body = "x"  # type: ignore[misc]


# --- default_lexicon ----------------------------------------------------


def test_default_lexicon_includes_categories():
    lex = default_lexicon()
    cats = {cat for (cat, _) in lex.values()}
    assert HaramCategory.GAMBLING in cats
    assert HaramCategory.RIBA in cats
    assert HaramCategory.HARAM_SECTOR in cats
    assert HaramCategory.SPECULATION_HYPE in cats


def test_default_lexicon_returns_fresh_copy():
    lex1 = default_lexicon()
    lex1["test"] = (HaramCategory.GAMBLING, ModerationOutcome.BLOCK)
    lex2 = default_lexicon()
    assert "test" not in lex2


# --- classify — clean path ---------------------------------------------


def test_classify_clean_passes():
    res = classify(_msg())
    assert res.outcome is ModerationOutcome.PASS
    assert not res.matched_phrases


# --- classify — gambling lexicon ---------------------------------------


def test_classify_gambling_blocks():
    res = classify(_msg(body="Casino vibes today"))
    assert res.outcome is ModerationOutcome.BLOCK
    assert HaramCategory.GAMBLING in res.matched_categories


def test_classify_lottery_blocks():
    res = classify(_msg(body="Lottery pick of the week"))
    assert res.outcome is ModerationOutcome.BLOCK


# --- classify — riba lexicon -------------------------------------------


def test_classify_riba_warns():
    res = classify(_msg(body="With leveraged margin you can amplify"))
    assert res.outcome is ModerationOutcome.WARN


def test_classify_guaranteed_return_blocks():
    res = classify(_msg(body="Trust me, guaranteed return next week"))
    assert res.outcome is ModerationOutcome.BLOCK


# --- classify — speculation hype ---------------------------------------


def test_classify_moonshot_warns():
    res = classify(_msg(body="This is a moonshot opportunity"))
    assert res.outcome is ModerationOutcome.WARN


# --- classify — case insensitive ---------------------------------------


def test_classify_case_insensitive():
    res = classify(_msg(body="CASINO PICK"))
    assert res.outcome is ModerationOutcome.BLOCK


# --- classify — hype score ---------------------------------------------


def test_hype_low_for_normal_text():
    res = classify(_msg(body="Normal message about earnings"))
    assert res.hype_score < 0.50


def test_hype_high_all_caps_warns():
    res = classify(_msg(body="THIS STOCK WILL EXPLODE TODAY!!!"))
    assert res.hype_score > 0.50
    assert res.outcome in (ModerationOutcome.WARN, ModerationOutcome.BLOCK)


def test_hype_with_rockets_warns():
    res = classify(_msg(body="🚀🚀🚀🚀 GOING UP 🚀🚀🚀🚀"))
    assert res.hype_score > 0.50


def test_hype_combined_with_lexicon_blocks():
    """Pin: hype > 0.80 + lexicon WARN → BLOCK."""
    res = classify(_msg(body="MOONSHOT YOLO!!!! 🚀🚀🚀🚀🚀 ALL CAPS!!!"))
    assert res.outcome is ModerationOutcome.BLOCK


# --- classify — repetition ---------------------------------------------


def test_repetition_warns():
    earlier = _msg(
        message_id="M0",
        body="Strong fundamentals.",
        posted_at=datetime(2026, 5, 10, 11, 59, 30),
    )
    current = _msg(
        message_id="M1",
        body="Strong fundamentals.",
        posted_at=datetime(2026, 5, 10, 12, 0, 0),
    )
    res = classify(current, history=[earlier])
    assert res.outcome is ModerationOutcome.WARN


def test_repetition_outside_window_passes():
    earlier = _msg(
        message_id="M0",
        body="Strong fundamentals.",
        posted_at=datetime(2026, 5, 10, 11, 0, 0),
    )
    current = _msg(
        message_id="M1",
        body="Strong fundamentals.",
        posted_at=datetime(2026, 5, 10, 12, 0, 0),
    )
    res = classify(current, history=[earlier], repeat_window_seconds=60)
    assert res.outcome is ModerationOutcome.PASS


def test_repetition_different_user_passes():
    other_user = _msg(
        message_id="M0",
        user_id="bob",
        body="Strong fundamentals.",
        posted_at=datetime(2026, 5, 10, 11, 59, 30),
    )
    current = _msg(
        message_id="M1",
        user_id="alice",
        body="Strong fundamentals.",
        posted_at=datetime(2026, 5, 10, 12, 0, 0),
    )
    res = classify(current, history=[other_user])
    assert res.outcome is ModerationOutcome.PASS


# --- classify — URL self-promotion -------------------------------------


def test_url_without_ticker_context_warns():
    res = classify(
        _msg(
            ticker_room="AAPL",
            body="Check out my blog https://example.com",
        )
    )
    assert res.outcome is ModerationOutcome.WARN


def test_url_with_ticker_context_passes():
    res = classify(
        _msg(
            ticker_room="AAPL",
            body="AAPL fundamentals https://example.com/aapl-research",
        )
    )
    assert res.outcome is ModerationOutcome.PASS


# --- classify_batch ----------------------------------------------------


def test_batch_detects_repetition_across_batch():
    msgs = [
        _msg(
            message_id="M1",
            body="Spam this",
            posted_at=datetime(2026, 5, 10, 12, 0, 0),
        ),
        _msg(
            message_id="M2",
            body="Spam this",
            posted_at=datetime(2026, 5, 10, 12, 0, 30),
        ),
    ]
    results = classify_batch(msgs)
    # First should pass, second should WARN due to repetition.
    assert results[0].outcome is ModerationOutcome.PASS
    assert results[1].outcome is ModerationOutcome.WARN


def test_batch_returns_one_result_per_message():
    msgs = [_msg(message_id=f"M{i}") for i in range(3)]
    results = classify_batch(msgs)
    assert len(results) == 3


# --- filter_passing ----------------------------------------------------


def test_filter_passing_drops_blocked():
    m_clean = _msg(message_id="M1", body="Clean message.")
    m_block = _msg(message_id="M2", body="Casino lottery yolo bet")
    results = classify_batch([m_clean, m_block])
    out = filter_passing([m_clean, m_block], results)
    ids = {m.message_id for m in out}
    assert "M1" in ids
    assert "M2" not in ids


def test_filter_passing_drops_warn():
    m_clean = _msg(message_id="M1", body="Clean message.")
    m_warn = _msg(message_id="M2", body="That moonshot is real")
    results = classify_batch([m_clean, m_warn])
    out = filter_passing([m_clean, m_warn], results)
    assert {m.message_id for m in out} == {"M1"}


# --- Custom lexicon -----------------------------------------------------


def test_custom_lexicon_overrides_default():
    custom = {
        "yikes": (HaramCategory.SPECULATION_HYPE, ModerationOutcome.WARN),
    }
    res = classify(_msg(body="yikes that move"), lexicon=custom)
    assert res.outcome is ModerationOutcome.WARN


def test_custom_lexicon_excludes_default():
    """Pin: replacing the lexicon disables default phrases."""
    custom = {"yikes": (HaramCategory.SPECULATION_HYPE, ModerationOutcome.WARN)}
    res = classify(_msg(body="Casino vibes"), lexicon=custom)
    # default 'casino' would BLOCK; with custom-only lexicon it's PASS.
    assert res.outcome is ModerationOutcome.PASS


# --- Render -------------------------------------------------------------


def test_render_pass_emoji():
    res = ModerationResult(
        message_id="M1",
        outcome=ModerationOutcome.PASS,
        matched_phrases=(),
        matched_categories=(),
        reasons=(),
        hype_score=0.1,
    )
    out = render_result(res)
    assert "✅" in out


def test_render_block_lists_reasons():
    res = ModerationResult(
        message_id="M1",
        outcome=ModerationOutcome.BLOCK,
        matched_phrases=("casino",),
        matched_categories=(HaramCategory.GAMBLING,),
        reasons=("matched 'casino' (gambling)",),
        hype_score=0.2,
    )
    out = render_result(res)
    assert "🛑" in out
    assert "casino" in out


def test_render_includes_hype_score():
    res = classify(_msg(body="Solid earnings call today"))
    out = render_result(res)
    assert "hype=" in out
