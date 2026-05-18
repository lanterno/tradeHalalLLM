"""Tests for the pure helpers in :mod:`core.llm.rag_db`.

The DB-backed store + pgvector retrieval are integration; this file
covers the dataclass adapter (`_row_to_dc`) and timestamp parser
(`_parse_timestamp`) that hydrate query results from the SQLModel
table into the public dataclass shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

from halal_trader.core.llm.rag_db import _parse_timestamp, _row_to_dc
from halal_trader.db.models import RationaleRow


def _row(**overrides) -> RationaleRow:
    base = dict(
        trade_id="trade-1",
        symbol="BTCUSDT",
        text="bullish breakout on rising volume",
        embedding=[0.1, 0.2, 0.3],
        outcome_pnl_pct=0.025,
        outcome_win=True,
        setup_type="breakout",
        timestamp=datetime(2026, 5, 1, 14, 30, tzinfo=UTC),
    )
    base.update(overrides)
    return RationaleRow(**base)


# ── _row_to_dc ──────────────────────────────────────────────


def test_row_to_dc_copies_all_simple_fields():
    out = _row_to_dc(_row())
    assert out.trade_id == "trade-1"
    assert out.symbol == "BTCUSDT"
    assert out.text == "bullish breakout on rising volume"
    assert out.outcome_pnl_pct == 0.025
    assert out.outcome_win is True
    assert out.setup_type == "breakout"


def test_row_to_dc_isoformats_timestamp():
    out = _row_to_dc(_row())
    assert isinstance(out.timestamp, str)
    assert "2026-05-01T14:30" in out.timestamp


def test_row_to_dc_empty_string_when_no_timestamp():
    """Some rows might have a None timestamp (legacy backfill); the
    dataclass uses an empty string rather than crashing."""
    out = _row_to_dc(_row(timestamp=None))
    assert out.timestamp == ""


def test_row_to_dc_handles_none_embedding_with_empty_vector():
    """A row with no embedding (background-fill in progress) → empty
    vector rather than crashing the public adapter."""
    out = _row_to_dc(_row(embedding=None))
    assert out.vector == []


def test_row_to_dc_copies_embedding_to_list():
    """The hydration converts the pgvector array to a plain list so
    downstream code (cosine, JSON serialise) doesn't see a numpy/array
    type."""
    out = _row_to_dc(_row(embedding=[1.0, 2.0, 3.0]))
    assert out.vector == [1.0, 2.0, 3.0]
    assert isinstance(out.vector, list)


def test_row_to_dc_starts_with_empty_tags():
    """The DB adapter doesn't carry tags forward — they're populated by
    a different table; defaults to an empty list so prompt code doesn't
    crash on `for tag in row.tags`."""
    out = _row_to_dc(_row())
    assert out.tags == []


# ── _parse_timestamp ────────────────────────────────────────


def test_parse_timestamp_iso_returns_aware():
    out = _parse_timestamp("2026-05-01T14:30:00+00:00")
    assert out == datetime(2026, 5, 1, 14, 30, tzinfo=UTC)


def test_parse_timestamp_naive_iso_promoted_to_utc():
    """Naive timestamps get UTC stamped (the DB column may have lost
    tz info on a roundtrip; we treat it as UTC by convention)."""
    out = _parse_timestamp("2026-05-01T14:30:00")
    assert out.tzinfo == UTC


def test_parse_timestamp_garbage_falls_back_to_now_utc():
    """Defensive: malformed timestamp → now (so callers always get a
    valid datetime to compare against)."""
    out = _parse_timestamp("not-a-date")
    assert isinstance(out, datetime)
    assert out.tzinfo == UTC


def test_parse_timestamp_empty_returns_now_utc():
    out = _parse_timestamp("")
    assert isinstance(out, datetime)
    assert out.tzinfo == UTC
