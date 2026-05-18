"""Tests for the private helpers in :mod:`web.prometheus`.

`render_metrics` + `collect_default_snapshots` are integration-tested
in `test_prometheus.py`. This file pins the small format helpers
underneath: `_escape` (Prometheus label escaping) and `_format_value`
(NaN-safe float rendering).
"""

from __future__ import annotations

from halal_trader.web.prometheus import _escape, _format_value

# ── _escape ────────────────────────────────────────────────


def test_escape_passes_plain_text_through():
    assert _escape("hello") == "hello"


def test_escape_doubles_backslash():
    """Backslash gets doubled — Prometheus text format spec."""
    assert _escape("path\\to\\file") == "path\\\\to\\\\file"


def test_escape_doubles_quote():
    """Double-quote in a label value must be escaped."""
    assert _escape('he said "hi"') == 'he said \\"hi\\"'


def test_escape_replaces_newline():
    """Newlines break the line-oriented exposition format."""
    assert _escape("line1\nline2") == "line1\\nline2"


def test_escape_handles_all_three_in_one_pass():
    """Mixed escapes — order matters: backslashes must be doubled
    *before* the quote/newline replacements (otherwise we'd double-
    escape the new backslash). Verify the function does it correctly."""
    out = _escape('a\\b"c\nd')
    # Backslash → \\, quote → \", newline → \n
    assert out == 'a\\\\b\\"c\\nd'


def test_escape_empty_string():
    assert _escape("") == ""


# ── _format_value ──────────────────────────────────────────


def test_format_value_int_like_float():
    assert _format_value(42.0) == "42"


def test_format_value_decimal():
    assert _format_value(3.14) == "3.14"


def test_format_value_zero():
    assert _format_value(0.0) == "0"


def test_format_value_negative():
    assert _format_value(-1.5) == "-1.5"


def test_format_value_nan_renders_as_nan_string():
    """Prometheus accepts the literal `NaN` token in float fields."""
    assert _format_value(float("nan")) == "NaN"


def test_format_value_strips_trailing_zeros():
    """0.10000 → 0.1 (g format)."""
    assert _format_value(0.10000) == "0.1"


def test_format_value_handles_very_small_number():
    """Very small numbers use `g` format (scientific or stripped)."""
    out = _format_value(1e-7)
    # Just sanity-check it's parseable and non-zero — the exact
    # representation depends on the `g` formatter.
    assert float(out) == 1e-7


def test_format_value_handles_large_number():
    out = _format_value(1_000_000.0)
    assert float(out) == 1_000_000.0
