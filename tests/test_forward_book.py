"""Paper-forward books: genesis, point-in-time screen, drift, monthly rebalance, no rewrites."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.research.forward_book import advance_book, create_book, report


@pytest.fixture(autouse=True)
def _no_event_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evening run's event-store refresh reaches SEC and Alpaca; these tests
    are about bars, screens and books (events/daily.py has its own)."""
    from halal_trader.events import daily

    async def nothing(*args: object, **kwargs: object) -> daily.EventRefresh:
        return daily.EventRefresh()

    monkeypatch.setattr(daily, "refresh_events", nothing)

    from halal_trader.research import daily as research_daily

    async def no_purification(*args: object, **kwargs: object) -> dict[str, int]:
        return {}

    monkeypatch.setattr(research_daily, "_purify", no_purification)
    monkeypatch.setattr(research_daily, "_zakat", no_purification)

    async def no_problems(*args: object, **kwargs: object) -> list[str]:
        return []

    monkeypatch.setattr(research_daily, "_backup_health", no_problems)


def _sessions(n: int) -> list[date]:
    out, d = [], date(2025, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _sessions(320)
# Daily growth per symbol: UP trends hard (top momentum), FLAT barely moves,
# SPUS/HLAL/SPY are benchmarks and never eligible.
DRIFT = {"UP": 0.004, "MID": 0.002, "FLAT": 0.0001, "SPY": 0.001, "SPUS": 0.001, "HLAL": 0.001}


def _close(symbol: str, i: int) -> float:
    # A small alternating wiggle so volatility is defined and differs by name.
    wiggle = 0.001 * (1 if i % 2 else -1) * (1 + list(DRIFT).index(symbol))
    return 100.0 * (1.0 + DRIFT[symbol]) ** i * (1.0 + wiggle)


async def _seed(engine: AsyncEngine, upto: int) -> None:
    rows = [{"s": s, "d": DAYS[i], "c": _close(s, i)} for s in DRIFT for i in range(upto + 1)]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, 'all', :c, :c, :c, :c, 1e6, now()) "
                "ON CONFLICT DO NOTHING"
            ),
            rows,
        )


async def _screen(engine: AsyncEngine, as_of: date, halal: list[str]) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, '', :v, '[]', '{}', "
                "'test', now())"
            ),
            [
                {"a": as_of, "s": s, "v": "halal" if s in halal else "not_halal"}
                for s in ("UP", "MID", "FLAT")
            ],
        )


def _mid_month_index() -> int:
    """A session late in the series whose next session is in the same month."""
    for i in range(300, 318):
        if DAYS[i + 1].month == DAYS[i].month and DAYS[i - 1].month == DAYS[i].month:
            return i
    raise AssertionError("no mid-month session")


async def _days(engine: AsyncEngine) -> list:  # type: ignore[type-arg]
    async with engine.connect() as conn:
        return (await conn.execute(text("SELECT * FROM forward_book_days ORDER BY day"))).all()


async def test_genesis_then_first_close_buys_the_top_names(engine: AsyncEngine) -> None:
    k = _mid_month_index()
    await _seed(engine, k + 1)
    await _screen(engine, DAYS[0], ["UP", "MID", "FLAT"])
    await create_book(engine, "s1", top_n=2, cost_bps=10)

    genesis = await advance_book(engine, "s1", through=DAYS[k])
    first = await advance_book(engine, "s1", through=DAYS[k + 1])

    assert [(g.day, g.nav, g.weights, g.rebalance_next) for g in genesis] == [
        (DAYS[k], 1.0, {}, True)
    ]
    (row,) = first
    assert row.day == DAYS[k + 1]
    assert row.turnover == pytest.approx(1.0)
    assert row.nav == pytest.approx(1.0 - 0.001)  # empty book: only the entry cost
    assert row.weights == {"UP": 0.5, "MID": 0.5}
    assert row.rebalance_next is False


async def test_eligibility_is_the_screen_as_of_the_decision_day(engine: AsyncEngine) -> None:
    """A screen dated after the decision cannot reach back and change it."""
    k = _mid_month_index()
    await _seed(engine, k + 1)
    await _screen(engine, DAYS[0], ["FLAT"])
    await _screen(engine, DAYS[k + 1], ["UP", "MID", "FLAT"])  # newer than the decision
    await create_book(engine, "s1", top_n=2)

    await advance_book(engine, "s1", through=DAYS[k])
    (row,) = await advance_book(engine, "s1", through=DAYS[k + 1])

    assert row.weights == {"FLAT": 1.0}


async def test_no_screen_holds_cash(engine: AsyncEngine) -> None:
    k = _mid_month_index()
    await _seed(engine, k + 1)
    await create_book(engine, "s1")

    await advance_book(engine, "s1", through=DAYS[k])
    (row,) = await advance_book(engine, "s1", through=DAYS[k + 1])

    assert (row.weights, row.nav, row.turnover) == ({}, 1.0, 0.0)


async def test_weights_drift_mid_month_and_rebalance_on_a_new_month(engine: AsyncEngine) -> None:
    k = _mid_month_index()
    end = next(i for i in range(k + 2, 319) if DAYS[i].month != DAYS[i - 1].month)
    await _seed(engine, end)
    await _screen(engine, DAYS[0], ["UP", "MID", "FLAT"])
    await create_book(engine, "s1", top_n=2, cost_bps=0)
    await advance_book(engine, "s1", through=DAYS[k])

    rows = await advance_book(engine, "s1", through=DAYS[end])

    by_day = {r.day: r for r in rows}
    middle = by_day[DAYS[k + 2]]
    assert middle.turnover == 0.0
    assert middle.weights["UP"] > 0.5  # the faster riser now weighs more
    growth = 0.5 * (_close("UP", k + 2) / _close("UP", k + 1)) + 0.5 * (
        _close("MID", k + 2) / _close("MID", k + 1)
    )
    assert middle.day_return == pytest.approx(growth - 1.0)
    first_of_month = by_day[DAYS[end]]
    assert first_of_month.turnover > 0.0  # back to equal weight
    assert first_of_month.weights == {"UP": 0.5, "MID": 0.5}
    assert all(r.turnover == 0.0 for r in rows if r.day not in (DAYS[k + 1], DAYS[end]))


async def test_rows_are_append_only(engine: AsyncEngine) -> None:
    k = _mid_month_index()
    await _seed(engine, k + 2)
    await _screen(engine, DAYS[0], ["UP", "MID", "FLAT"])
    await create_book(engine, "s1", top_n=2)
    await advance_book(engine, "s1", through=DAYS[k + 1])  # genesis only (empty book)
    await advance_book(engine, "s1", through=DAYS[k + 2])
    before = await _days(engine)

    again = await advance_book(engine, "s1", through=DAYS[k + 2])

    assert again == []
    assert await _days(engine) == before


async def test_report_compares_with_benchmarks_over_the_same_sessions(
    engine: AsyncEngine,
) -> None:
    k = _mid_month_index()
    await _seed(engine, k + 5)
    await _screen(engine, DAYS[0], ["UP", "MID", "FLAT"])
    await create_book(engine, "s1", top_n=2)
    await advance_book(engine, "s1", through=DAYS[k])
    await advance_book(engine, "s1", through=DAYS[k + 5])

    r = await report(engine, "s1")

    assert (r.started, r.days) == (DAYS[k], 5)
    assert r.stats is not None and r.benchmarks["SPUS"] is not None
    assert set(r.holdings) == {"UP", "MID"}


async def test_unknown_strategy_is_refused(engine: AsyncEngine) -> None:
    with pytest.raises(ValueError):
        await create_book(engine, "x", strategy="guesswork")


# ── the evening run ───────────────────────────────────────────────────


def _settings():  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    return SimpleNamespace(
        alpaca=SimpleNamespace(api_key="k", secret_key="s"),
        edgar=SimpleNamespace(user_agent="test test@example.invalid"),
        core=SimpleNamespace(paper=True),
    )


async def test_run_without_bars_reports_instead_of_failing(engine: AsyncEngine) -> None:
    from halal_trader.research.daily import run_research

    run = await run_research(engine, _settings(), today=DAYS[10])

    assert run.errors and "backfill" in run.errors[0]


async def test_a_failed_bar_update_still_advances_books_and_skips_a_fresh_screen(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halal_trader.data import store
    from halal_trader.research.daily import run_research

    async def broken(*args: object, **kwargs: object) -> dict[str, int]:
        raise RuntimeError("alpaca down")

    monkeypatch.setattr(store, "update_bars", broken)
    k = _mid_month_index()
    await _seed(engine, k + 1)
    await _screen(engine, DAYS[k], ["UP", "MID", "FLAT"])  # fresh: no re-screen
    await create_book(engine, "s1", top_n=2)
    await advance_book(engine, "s1", through=DAYS[k])

    run = await run_research(engine, _settings(), today=DAYS[k + 1])

    assert run.books == {"s1": 1}
    assert run.screened is None
    assert len(run.errors) == 1 and "alpaca down" in run.errors[0]


async def _cap_screen(engine: AsyncEngine, as_of: date, caps: dict[str, float]) -> None:
    """A screen with prices and share counts: price 100, shares = cap / 100."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, '', :v, '[]', "
                "CAST(:m AS JSONB), 'test', now()) ON CONFLICT (as_of, symbol) DO UPDATE "
                "SET verdict = EXCLUDED.verdict, metrics = EXCLUDED.metrics"
            ),
            [
                {
                    "a": as_of,
                    "s": s,
                    "v": "halal" if s in caps else "not_halal",
                    "m": '{"price": 100, "shares_outstanding": %f}' % (caps.get(s, 1e6) / 100),
                }
                for s in ("UP", "MID", "FLAT")
            ],
        )


async def test_the_core_book_holds_cap_weights_trades_monthly_and_sells_a_failed_name_at_once(
    engine: AsyncEngine,
) -> None:
    k = _mid_month_index()
    await _seed(engine, k + 2)
    await _cap_screen(engine, DAYS[k - 5], {"UP": 3e9, "MID": 1e9})
    await create_book(engine, "core", strategy="core-strict-cap", top_n=100, cost_bps=5.0)
    await advance_book(engine, "core", through=DAYS[k - 1])  # genesis
    first = await advance_book(engine, "core", through=DAYS[k])  # the forced first rebalance
    w = first[-1].weights
    assert set(w) == {"UP", "MID"} and w["UP"] / w["MID"] == pytest.approx(3.0, rel=0.05)

    quiet = await advance_book(engine, "core", through=DAYS[k + 1])  # mid-month, no change
    assert quiet[-1].turnover == 0.0

    await _cap_screen(engine, DAYS[k + 1], {"UP": 3e9})  # MID now fails the screen
    sold = await advance_book(engine, "core", through=DAYS[k + 2])
    assert set(sold[-1].weights) == {"UP"} and sold[-1].turnover > 0
