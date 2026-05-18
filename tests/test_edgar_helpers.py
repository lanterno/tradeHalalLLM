"""Tests for the pure helpers in :mod:`trading.edgar_catalysts`.

The full source's HTTP path is covered in `test_edgar_catalysts.py`.
This file pins the two pure helpers underneath: `_parse_filing_date`
and `_summarize_items`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from halal_trader.trading.edgar_catalysts import (
    _parse_filing_date,
    _summarize_items,
)

# ── _parse_filing_date ──────────────────────────────────────


def test_parse_iso_date_attaches_utc():
    """A naive ISO date gets stamped UTC (EDGAR returns dates without tz)."""
    out = _parse_filing_date("2026-05-10")
    assert out == datetime(2026, 5, 10, tzinfo=UTC)


def test_parse_iso_datetime_attaches_utc():
    out = _parse_filing_date("2026-05-10T14:00:00")
    assert out == datetime(2026, 5, 10, 14, 0, tzinfo=UTC)


def test_parse_empty_returns_none():
    assert _parse_filing_date("") is None


def test_parse_garbage_returns_none():
    """Defensive — never raise on a malformed EDGAR date."""
    assert _parse_filing_date("not-a-date") is None
    assert _parse_filing_date("2026-13-99") is None


# ── _summarize_items ────────────────────────────────────────


def test_empty_list_returns_empty_string():
    assert _summarize_items([]) == ""


def test_known_item_resolves_to_human_label():
    out = _summarize_items(["2.02"])
    # The known label for 2.02 is "results of operations (earnings)".
    assert "earnings" in out


def test_item_prefix_stripped_before_lookup():
    """EDGAR sometimes returns "Item 2.02" rather than the bare key."""
    out = _summarize_items(["Item 2.02"])
    assert "earnings" in out


def test_unknown_item_falls_back_to_raw_number():
    """If we don't have a label for an item, render the cleaned number
    verbatim — better than dropping the signal entirely."""
    out = _summarize_items(["9.99"])
    assert "9.99" in out


def test_truncates_after_three_items_to_keep_prompt_cheap():
    """Filings can list many items; only the first 3 are summarised."""
    items = ["2.02", "5.02", "1.01", "8.01", "7.01"]
    out = _summarize_items(items)
    # First three labels present.
    assert "earnings" in out  # 2.02
    assert "executive departure" in out  # 5.02
    assert "material agreement entered" in out  # 1.01
    # Fourth + onward dropped.
    assert "other events" not in out  # 8.01


def test_separator_is_semicolon_space():
    """Stable separator so the prompt template can split / count."""
    out = _summarize_items(["2.02", "5.02"])
    assert "; " in out
    parts = out.split("; ")
    assert len(parts) == 2
