"""Tests for the Sharia exception queue."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.halal.exception_queue import (
    ExceptionQueue,
    render_summary,
)

# ── Add ──────────────────────────────────────────────────────────


async def test_add_creates_pending_entry(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    e = await q.add(
        instrument="DOGE", kind="new_token", reasoning="meme coin, no Sharia ruling yet"
    )
    assert e.status == "pending"
    assert e.instrument == "DOGE"
    assert e.entry_id == "DOGE:new_token"


async def test_add_idempotent_on_pending(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="X", kind="k", reasoning="first")
    await q.add(instrument="X", kind="k", reasoning="updated")
    rows = await q.all()
    assert len(rows) == 1
    assert rows[0].reasoning == "updated"


async def test_add_after_decision_creates_new_record(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="X", kind="k", reasoning="a")
    await q.decide("X:k", status="rejected", decided_by="ops")
    # Re-adding overwrites with a fresh pending entry
    await q.add(instrument="X", kind="k", reasoning="re-screened")
    rows = await q.all()
    assert len(rows) == 1
    assert rows[0].status == "pending"
    assert rows[0].reasoning == "re-screened"
    assert rows[0].decided_at is None


# ── Decide ───────────────────────────────────────────────────────


async def test_decide_approve(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="X", kind="k", reasoning="x")
    assert await q.decide("X:k", status="approved", decided_by="ops") is True
    assert await q.is_approved("X", "k") is True
    assert await q.is_approved("X", "other_kind") is False


async def test_decide_unknown_entry_returns_false(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    assert await q.decide("nope", status="approved") is False


async def test_decide_invalid_status_raises(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="X", kind="k", reasoning="x")
    with pytest.raises(ValueError):
        await q.decide("X:k", status="bogus")  # type: ignore[arg-type]


# ── Filtering ────────────────────────────────────────────────────


async def test_pending_filters(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="A", kind="k", reasoning="a")
    await q.add(instrument="B", kind="k", reasoning="b")
    await q.decide("A:k", status="approved")
    pending = await q.pending()
    assert len(pending) == 1
    assert pending[0].instrument == "B"


async def test_by_status(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="A", kind="k", reasoning="a")
    await q.add(instrument="B", kind="k", reasoning="b")
    await q.decide("A:k", status="rejected")
    assert len(await q.by_status("rejected")) == 1
    assert len(await q.by_status("pending")) == 1


# ── Persistence ──────────────────────────────────────────────────


async def test_persists_across_instances(engine: AsyncEngine) -> None:
    q1 = ExceptionQueue(engine=engine)
    await q1.add(instrument="X", kind="k", reasoning="x")
    await q1.decide("X:k", status="approved", decided_by="ops")
    q2 = ExceptionQueue(engine=engine)
    rows = await q2.all()
    assert len(rows) == 1
    assert rows[0].status == "approved"


# ── Render ───────────────────────────────────────────────────────


def test_render_empty() -> None:
    assert "empty" in render_summary([])


async def test_render_lists_each_entry(engine: AsyncEngine) -> None:
    q = ExceptionQueue(engine=engine)
    await q.add(instrument="A", kind="k", reasoning="a a a a")
    out = render_summary(await q.all())
    assert "A" in out
    assert "[pending]" in out
