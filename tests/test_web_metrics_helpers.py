"""Tests for pure helpers in :mod:`web.metrics`.

The integration-level `cycle_metrics` / `llm_metrics` cases live in
``test_web_metrics.py``. This file pins the small private helpers
underneath — `_percentile`, `_within`, `_tail`, `_iter_records` —
where the integration path doesn't directly exercise edge cases like
the asctime (non-ISO-T) timestamp form or the missing-tzinfo branch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from halal_trader.web.metrics import _iter_records, _percentile, _tail, _within

# ── _percentile ────────────────────────────────────────────


def test_percentile_empty_returns_none():
    assert _percentile([], 0.5) is None


def test_percentile_single_element():
    assert _percentile([42.0], 0.5) == 42.0
    assert _percentile([42.0], 0.99) == 42.0
    assert _percentile([42.0], 0.0) == 42.0


def test_percentile_p50_picks_middle():
    """Index-based — for [1,2,3,4,5] p50 = round(0.5 * 4) = 2 → 3."""
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_percentile_p100_picks_max():
    assert _percentile([1.0, 5.0, 9.0], 1.0) == 9.0


def test_percentile_p0_picks_min():
    assert _percentile([5.0, 1.0, 9.0], 0.0) == 1.0


def test_percentile_clamps_above_one():
    """Defensive: the formula uses min() so > 1.0 still picks the last element."""
    assert _percentile([1.0, 2.0, 3.0], 1.5) == 3.0


def test_percentile_sorts_input():
    """Input order doesn't matter — helper sorts internally."""
    assert _percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0


# ── _within ────────────────────────────────────────────────


def test_within_iso_with_z_suffix():
    """The ``Z`` suffix (Zulu time) is normalised to +00:00."""
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    rec = {"timestamp": "2026-04-25T13:00:00Z"}
    assert _within(rec, since) is True


def test_within_iso_without_tz_assumes_utc():
    """Naive ISO timestamps get UTC stamped on — the bot logs in UTC,
    so we don't want a missing tz to silently shift the comparison."""
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    rec = {"timestamp": "2026-04-25T13:00:00"}
    assert _within(rec, since) is True


def test_within_asctime_form_no_T():
    """`python-json-logger`'s asctime emits ``2026-04-25 13:00:00`` (space,
    not T). Helper must accept this — covered indirectly elsewhere but
    pinned here so a regression in the parsing branch is caught."""
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    rec = {"timestamp": "2026-04-25 13:00:00"}
    assert _within(rec, since) is True


def test_within_before_window_returns_false():
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    rec = {"timestamp": "2026-04-25T11:59:00Z"}
    assert _within(rec, since) is False


def test_within_at_boundary_inclusive():
    """`>= since` — exactly at the cutoff is in-window, not out."""
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    rec = {"timestamp": "2026-04-25T12:00:00Z"}
    assert _within(rec, since) is True


def test_within_missing_timestamp_returns_false():
    """No timestamp key → silently exclude rather than crash."""
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    assert _within({}, since) is False


def test_within_non_string_timestamp_returns_false():
    """A numeric / None timestamp (malformed log row) is silently excluded."""
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    assert _within({"timestamp": 12345}, since) is False
    assert _within({"timestamp": None}, since) is False


def test_within_unparseable_string_returns_false():
    since = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    assert _within({"timestamp": "not-a-date"}, since) is False


# ── _tail ──────────────────────────────────────────────────


def test_tail_missing_file_yields_nothing(tmp_path: Path):
    """Defensive: dashboard polls metrics on cold start before any log
    rotation has happened; a missing file must not crash."""
    out = list(_tail(tmp_path / "no-such.log", max_lines=100))
    assert out == []


def test_tail_returns_all_when_under_cap(tmp_path: Path):
    p = tmp_path / "log"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert list(_tail(p, max_lines=100)) == ["a", "b", "c"]


def test_tail_truncates_to_last_n_lines(tmp_path: Path):
    """Memory-bound for huge logs — only the last ``max_lines`` lines
    are yielded. Order is preserved (oldest of the kept window first)."""
    p = tmp_path / "log"
    p.write_text("\n".join(f"line{i}" for i in range(100)) + "\n", encoding="utf-8")
    out = list(_tail(p, max_lines=5))
    assert out == ["line95", "line96", "line97", "line98", "line99"]


def test_tail_strips_trailing_newline_only(tmp_path: Path):
    """A line with embedded whitespace shouldn't lose internal spaces —
    only the newline at the end gets stripped."""
    p = tmp_path / "log"
    p.write_text("hello world  \n", encoding="utf-8")
    assert list(_tail(p, max_lines=10)) == ["hello world  "]


def test_tail_handles_non_utf8_bytes_via_replace(tmp_path: Path):
    """Bot logs are UTF-8 but a random truncated rotation might land a
    half-byte at the start. ``errors='replace'`` keeps the read going
    rather than raising mid-poll."""
    p = tmp_path / "log"
    p.write_bytes(b"\xff invalid start\nclean line\n")
    out = list(_tail(p, max_lines=10))
    assert len(out) == 2
    assert out[1] == "clean line"


# ── _iter_records ──────────────────────────────────────────


def test_iter_records_skips_unparseable_lines(tmp_path: Path):
    """Mixed valid + garbage lines: only the JSON ones are yielded as
    dicts. The integration test covers this end-to-end; this helper
    test pins the contract directly so a future refactor that surfaces
    the JSON error breaks here first."""
    p = tmp_path / "log"
    p.write_text(
        '{"event":"a"}\nnot-json\n{"event":"b"}\n',
        encoding="utf-8",
    )
    out = list(_iter_records(p, max_lines=10))
    assert [r["event"] for r in out] == ["a", "b"]


def test_iter_records_empty_file_yields_nothing(tmp_path: Path):
    p = tmp_path / "log"
    p.write_text("", encoding="utf-8")
    assert list(_iter_records(p, max_lines=10)) == []


def test_iter_records_missing_file_yields_nothing(tmp_path: Path):
    assert list(_iter_records(tmp_path / "no-such.log", max_lines=10)) == []


# ── timedelta interplay ───────────────────────────────────


def test_within_window_timedelta_basic():
    """Concrete check that the standard 1-hour window logic works."""
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    since = now - timedelta(hours=1)
    in_window = {"timestamp": (now - timedelta(minutes=30)).isoformat()}
    out_window = {"timestamp": (now - timedelta(hours=2)).isoformat()}
    assert _within(in_window, since) is True
    assert _within(out_window, since) is False
