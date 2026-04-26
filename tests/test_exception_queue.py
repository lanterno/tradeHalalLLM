"""Tests for the Sharia exception queue."""

from __future__ import annotations

from pathlib import Path

import pytest

from halal_trader.halal.exception_queue import (
    ExceptionQueue,
    render_summary,
)


def _q(tmp_path: Path) -> ExceptionQueue:
    return ExceptionQueue(path=tmp_path / "exceptions.json")


# ── Add ──────────────────────────────────────────────────────────


def test_add_creates_pending_entry(tmp_path: Path) -> None:
    q = _q(tmp_path)
    e = q.add(instrument="DOGE", kind="new_token", reasoning="meme coin, no Sharia ruling yet")
    assert e.status == "pending"
    assert e.instrument == "DOGE"
    assert e.entry_id == "DOGE:new_token"


def test_add_idempotent_on_pending(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="X", kind="k", reasoning="first")
    q.add(instrument="X", kind="k", reasoning="updated")
    assert len(q.entries) == 1
    assert q.entries["X:k"].reasoning == "updated"


def test_add_after_decision_creates_new_record(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="X", kind="k", reasoning="a")
    q.decide("X:k", status="rejected", decided_by="ops")
    # Re-adding overwrites with a fresh pending entry
    q.add(instrument="X", kind="k", reasoning="re-screened")
    e = q.entries["X:k"]
    assert e.status == "pending"
    assert e.reasoning == "re-screened"
    # decided_at gets cleared because it's a fresh ExceptionEntry
    assert e.decided_at is None


# ── Decide ───────────────────────────────────────────────────────


def test_decide_approve(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="X", kind="k", reasoning="x")
    assert q.decide("X:k", status="approved", decided_by="ops") is True
    assert q.entries["X:k"].status == "approved"
    assert q.is_approved("X", "k") is True
    assert q.is_approved("X", "other_kind") is False


def test_decide_unknown_entry_returns_false(tmp_path: Path) -> None:
    q = _q(tmp_path)
    assert q.decide("nope", status="approved") is False


def test_decide_invalid_status_raises(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="X", kind="k", reasoning="x")
    with pytest.raises(ValueError):
        q.decide("X:k", status="bogus")  # type: ignore[arg-type]


# ── Filtering ────────────────────────────────────────────────────


def test_pending_filters(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="A", kind="k", reasoning="a")
    q.add(instrument="B", kind="k", reasoning="b")
    q.decide("A:k", status="approved")
    pending = q.pending()
    assert len(pending) == 1
    assert pending[0].instrument == "B"


def test_by_status(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="A", kind="k", reasoning="a")
    q.add(instrument="B", kind="k", reasoning="b")
    q.decide("A:k", status="rejected")
    assert len(q.by_status("rejected")) == 1
    assert len(q.by_status("pending")) == 1


# ── Persistence ──────────────────────────────────────────────────


def test_persists_across_instances(tmp_path: Path) -> None:
    p = tmp_path / "exc.json"
    q1 = ExceptionQueue(path=p)
    q1.add(instrument="X", kind="k", reasoning="x")
    q1.decide("X:k", status="approved", decided_by="ops")
    q2 = ExceptionQueue(path=p)
    assert len(q2.entries) == 1
    assert q2.entries["X:k"].status == "approved"


def test_resilient_to_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "exc.json"
    p.write_text("{not json")
    q = ExceptionQueue(path=p)
    assert q.entries == {}


# ── Render ───────────────────────────────────────────────────────


def test_render_empty() -> None:
    assert "empty" in render_summary([])


def test_render_lists_each_entry(tmp_path: Path) -> None:
    q = _q(tmp_path)
    q.add(instrument="A", kind="k", reasoning="a a a a")
    out = render_summary(q.all())
    assert "A" in out
    assert "[pending]" in out
