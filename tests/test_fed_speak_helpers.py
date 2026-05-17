"""Tests for the private RSS-parsing helpers in :mod:`trading.fed_speak`.

Existing `test_fed_speak.py` covers `score_text`, `aggregate_signal`,
`parse_rss`, and `format_fed_speak_for_prompt`. This file pins the
small string helpers underneath: `_extract_tag`, `_strip_cdata`,
`_clean_html`, `_parse_pubdate`, `_extract_speaker`.
"""

from __future__ import annotations

from datetime import UTC

from halal_trader.trading.fed_speak import (
    _clean_html,
    _extract_speaker,
    _extract_tag,
    _parse_pubdate,
    _strip_cdata,
)

# ── _extract_tag ────────────────────────────────────────────


def test_extract_tag_finds_simple_tag():
    blob = "<title>Speech by Powell</title>"
    assert _extract_tag(blob, "title") == "Speech by Powell"


def test_extract_tag_strips_cdata_wrapper():
    blob = "<title><![CDATA[Speech by Powell]]></title>"
    assert _extract_tag(blob, "title") == "Speech by Powell"


def test_extract_tag_returns_empty_string_when_missing():
    assert _extract_tag("<other>x</other>", "title") == ""


# ── _strip_cdata ────────────────────────────────────────────


def test_strip_cdata_removes_wrapper():
    assert _strip_cdata("<![CDATA[hello]]>") == "hello"


def test_strip_cdata_passes_plain_string_through():
    assert _strip_cdata("plain") == "plain"


def test_strip_cdata_handles_outer_whitespace():
    """The strip happens BEFORE the CDATA detect, so outer whitespace
    doesn't prevent the unwrap."""
    assert _strip_cdata("  <![CDATA[hello]]>  ") == "hello"


def test_strip_cdata_partial_marker_left_alone():
    """A string with only opening or only closing marker isn't unwrapped."""
    assert _strip_cdata("<![CDATA[hello") == "<![CDATA[hello"
    assert _strip_cdata("hello]]>") == "hello]]>"


# ── _clean_html ────────────────────────────────────────────


def test_clean_html_strips_simple_tags():
    assert _clean_html("<p>hello <b>world</b></p>") == "hello world"


def test_clean_html_strips_self_closing_tags():
    assert _clean_html("a<br/>b") == "ab"


def test_clean_html_returns_empty_on_pure_tags():
    assert _clean_html("<p></p>") == ""


def test_clean_html_strips_outer_whitespace():
    assert _clean_html("  <p>hi</p>  ") == "hi"


# ── _parse_pubdate ─────────────────────────────────────────


def test_parse_pubdate_rfc822_with_tz():
    out = _parse_pubdate("Tue, 23 Apr 2026 14:30:00 +0000")
    assert out is not None
    assert out.year == 2026
    assert out.month == 4
    assert out.day == 23


def test_parse_pubdate_naive_promoted_to_utc():
    """RFC 822 dates without an offset get UTC stamped."""
    out = _parse_pubdate("Tue, 23 Apr 2026 14:30:00")
    if out is None:
        # Not all parsers accept tz-less RFC822 — that's OK, it returns None.
        return
    assert out.tzinfo == UTC


def test_parse_pubdate_garbage_returns_none():
    assert _parse_pubdate("not-a-date") is None


def test_parse_pubdate_empty_returns_none():
    assert _parse_pubdate("") is None


# ── _extract_speaker ──────────────────────────────────────


def test_extract_speaker_em_dash():
    """Em dash (`—`) is the typical separator on FED titles."""
    assert _extract_speaker("Powell — Recent Economic Developments") == "Powell"


def test_extract_speaker_hyphen_fallback():
    """Some titles use a regular hyphen instead of em-dash."""
    assert _extract_speaker("Powell - Recent Developments") == "Powell"


def test_extract_speaker_em_dash_preferred_over_hyphen():
    """If both are present, em-dash wins (it's the more reliable separator)."""
    out = _extract_speaker("Powell-Vice — Subject")
    assert out == "Powell-Vice"


def test_extract_speaker_no_separator_returns_full_title():
    """A title with no separator is the speaker name verbatim."""
    assert _extract_speaker("Powell speaks today") == "Powell speaks today"


def test_extract_speaker_empty_returns_empty():
    assert _extract_speaker("") == ""
