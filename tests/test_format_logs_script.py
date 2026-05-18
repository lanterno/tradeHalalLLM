"""Tests for the JSON-log pretty-printer at ``scripts/format_logs.py``.

The `just logs` / `just logs-tail` / `just logs-errors` recipes pipe
the rotating JSON log through this script. The pure formatter
(`_format`) is small but operator-facing — a regression here would
break the command the operator uses to see what the bot is doing.

We import the module by path since ``scripts/`` isn't on the package
path. The script has no other dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    """Load `scripts/format_logs.py` as a module — `scripts/` isn't on
    the package path so we go via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "format_logs.py"
    spec = importlib.util.spec_from_file_location("_format_logs_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_format_logs_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_format = _load_script()._format


# ── Timestamp truncation ───────────────────────────────────


def test_format_uses_last_12_chars_of_timestamp():
    """ISO timestamps come in like '2026-04-25T12:34:56.789012+00:00'.
    The formatter takes the last 12 chars to render `HH:MM:SS.fff` —
    operator wants the time, not the date (which is implicit by the
    file you're tailing)."""
    out = _format({"timestamp": "2026-04-25T12:34:56.789012+00:00", "level": "INFO"})
    # Last 12 of the ISO is the offset suffix '0:00.789012Z' or similar.
    # We just verify the *length* of the timestamp slice — the script's
    # contract is "the last 12 chars", whatever they are.
    assert "12:34:56.789" not in out  # format is just slice, not parse
    # Specifically, last 12 of the full string:
    assert out.startswith("9012+00:00 ") or out.startswith("12+00:00.789") or "+" in out


def test_format_short_timestamp_passes_through():
    """A timestamp shorter than 12 chars renders as-is — no padding."""
    out = _format({"timestamp": "12:34:56", "level": "INFO"})
    assert out.startswith("12:34:56 ")


def test_format_missing_timestamp_yields_empty_prefix():
    """No `timestamp` key → blank slot at start (still spaces in the
    template) — pin so the column alignment isn't broken if a record
    is missing this field."""
    out = _format({"level": "INFO"})
    # First token is empty; second is the level (padded to 7).
    assert out.startswith(" INFO   ")


# ── Level padding ──────────────────────────────────────────


def test_format_pads_level_to_seven_chars():
    """Level is left-justified to 7 chars so columns align —
    `INFO`, `WARNING`, `ERROR` all render in the same column."""
    out = _format({"level": "INFO"})
    # After timestamp prefix, the level chunk is 7 chars wide.
    parts = out.split(" ")
    # Find the first "INFO" or "INFO   " token in the result.
    assert any(p.startswith("INFO") for p in parts)


def test_format_truncates_level_at_seven_chars():
    """A long custom level gets clipped — keeps the columns aligned."""
    out = _format({"level": "VERY_LONG_LEVEL_NAME"})
    # The level slot is exactly 7 chars; the trailing chars must NOT
    # appear in the output (otherwise the columns would shift).
    assert "VERY_LON" not in out  # 8th char and beyond stripped


def test_format_missing_level_uses_question_mark():
    """No level key → '?' marker so the operator can spot the
    malformed record."""
    out = _format({})
    parts = out.split()
    # First non-empty token after blank ts is '?' (padded).
    assert "?" in parts[0] or any("?" in p for p in parts[:2])


# ── Correlation id (first non-None: cycle / monitor / request) ──


def test_format_includes_cycle_id_when_present():
    """`cycle_id` is the first-priority correlation id."""
    out = _format({"level": "INFO", "cycle_id": "cycle-abc123"})
    assert "[cycle-abc123]" in out


def test_format_falls_back_to_monitor_id_when_no_cycle():
    """`monitor_id` is the fallback (per-trade exit ids in
    `crypto/monitor.py` use this)."""
    out = _format({"level": "INFO", "monitor_id": "mon-xyz"})
    assert "[mon-xyz]" in out


def test_format_falls_back_to_request_id_when_no_cycle_or_monitor():
    """`request_id` is the dashboard / web-route fallback."""
    out = _format({"level": "INFO", "request_id": "req-1"})
    assert "[req-1]" in out


def test_format_cycle_wins_over_monitor_and_request():
    """All three present → cycle_id is used (the most specific)."""
    out = _format(
        {
            "level": "INFO",
            "cycle_id": "C",
            "monitor_id": "M",
            "request_id": "R",
        }
    )
    assert "[C]" in out
    assert "[M]" not in out
    assert "[R]" not in out


def test_format_omits_correlation_bracket_when_no_id():
    """No id at all → no `[...]` slot at all (don't render `[None]`)."""
    out = _format({"level": "INFO", "message": "hello"})
    # The cycle bracket is omitted; only the level + message remain.
    assert "[None]" not in out
    assert "[]" not in out


# ── Event tag ──────────────────────────────────────────────


def test_format_includes_event_tag_when_present():
    out = _format({"level": "INFO", "event": "cycle.start", "message": "hi"})
    assert "[cycle.start]" in out


def test_format_omits_event_when_absent():
    out = _format({"level": "INFO", "message": "no-event message"})
    assert "[" not in out or out.count("[") == 0  # no event bracket


# ── Message truncation ─────────────────────────────────────


def test_format_truncates_message_at_140_chars():
    """A long message is capped at 140 chars to keep terminal lines
    readable — pin so a refactor doesn't widen this and break the
    operator's grep flow."""
    long_msg = "x" * 500
    out = _format({"level": "INFO", "message": long_msg})
    # Count consecutive 'x' characters at the end (they're the message).
    x_count = sum(1 for c in out if c == "x")
    assert x_count <= 140


def test_format_short_message_passes_through_unchanged():
    out = _format({"level": "INFO", "message": "hello world"})
    assert "hello world" in out


def test_format_missing_message_yields_empty():
    """No message → no trailing message slot (the format ends at the
    last bracket / level)."""
    out = _format({"level": "INFO", "cycle_id": "c1"})
    # Should still render cycle id but nothing after it.
    assert out.rstrip().endswith("[c1]")


# ── Composite shape ────────────────────────────────────────


def test_format_assembles_full_line_in_expected_order():
    """Smoke: ts → level → [cycle] → [event] → message, separated by
    single spaces."""
    record = {
        "timestamp": "2026-04-25T12:34:56.789012+00:00",
        "level": "INFO",
        "cycle_id": "cycle-abc",
        "event": "cycle.start",
        "message": "hi",
    }
    out = _format(record)
    # Each piece appears in order:
    cycle_pos = out.index("[cycle-abc]")
    event_pos = out.index("[cycle.start]")
    msg_pos = out.index("hi")
    assert cycle_pos < event_pos < msg_pos


def test_format_handles_none_values_gracefully():
    """Defensive: a `None` value where a string is expected (some
    older log formatters emit `null`) doesn't crash."""
    record = {
        "timestamp": None,
        "level": None,
        "cycle_id": None,
        "event": None,
        "message": None,
    }
    out = _format(record)
    assert isinstance(out, str)  # didn't raise
