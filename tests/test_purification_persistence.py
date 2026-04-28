"""Persisted purification ledger — repository round-trip + totals."""

from __future__ import annotations

from halal_trader.db.repository import Repository


async def test_record_and_read_outstanding(engine):
    repo = Repository(engine)
    eid = await repo.record_purification(
        symbol="aapl",
        dividend_usd=100.0,
        haram_pct=0.05,
        purification_usd=5.0,
        notes="Q1 dividend",
    )
    assert eid > 0

    outstanding = await repo.get_outstanding_purification()
    assert len(outstanding) == 1
    assert outstanding[0]["symbol"] == "AAPL"
    assert outstanding[0]["purification_usd"] == 5.0
    assert outstanding[0]["paid_at"] is None


async def test_mark_paid_moves_entry_out_of_outstanding(engine):
    repo = Repository(engine)
    eid = await repo.record_purification(
        symbol="A", dividend_usd=100, haram_pct=0.05, purification_usd=5
    )
    ok = await repo.mark_purification_paid(eid)
    assert ok is True

    outstanding = await repo.get_outstanding_purification()
    assert outstanding == []

    totals = await repo.get_purification_totals()
    assert totals["outstanding_usd"] == 0.0
    assert totals["paid_usd"] == 5.0


async def test_mark_paid_unknown_id_returns_false(engine):
    repo = Repository(engine)
    ok = await repo.mark_purification_paid(9999)
    assert ok is False


async def test_totals_sum_across_many_entries(engine):
    repo = Repository(engine)
    ids = []
    for sym, div in [("A", 100), ("B", 200), ("C", 50)]:
        ids.append(
            await repo.record_purification(
                symbol=sym,
                dividend_usd=div,
                haram_pct=0.10,
                purification_usd=div * 0.10,
            )
        )
    await repo.mark_purification_paid(ids[1])
    totals = await repo.get_purification_totals()
    assert totals["outstanding_usd"] == 15.0
    assert totals["paid_usd"] == 20.0


async def test_outstanding_sorted_newest_first(engine):
    repo = Repository(engine)
    for sym in ("A", "B", "C"):
        await repo.record_purification(
            symbol=sym, dividend_usd=10, haram_pct=0.05, purification_usd=0.5
        )
    rows = await repo.get_outstanding_purification()
    assert rows[0]["symbol"] == "C"
    assert rows[-1]["symbol"] == "A"
