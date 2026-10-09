"""The Core page's payload (portfolio/core_view.build): holdings against the
screen and the book, the gate's day strip, runs with their fills, benchmarks."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sectors import sector_of
from halal_trader.market_hours import MARKET_TZ
from halal_trader.portfolio.core_view import _earliest_pass, _fill_histogram, _weighted, build

SETTINGS = SimpleNamespace(core=SimpleNamespace(paper=True))
# Thursday noon in New York: today's 15:40 run is still to come.
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=MARKET_TZ)
MON, TUE, WED = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)
SOFTWARE = "SERVICES-PREPACKAGED SOFTWARE"
SEMIS = "SEMICONDUCTORS & RELATED DEVICES"


def _fill(bps: float | None, notional: float, arrival: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(vs_close_bps=bps, vs_arrival_bps=arrival, filled_notional=notional)


def test_the_histogram_bins_filled_value_and_folds_the_tails_into_the_outer_bins() -> None:
    bins = _fill_histogram(
        [_fill(3, 100), _fill(-12, 50), _fill(95, 10), _fill(-400, 5), _fill(None, 99), _fill(7, 0)]
    )
    value = {b["from_bps"]: b["value"] for b in bins}
    assert [b["from_bps"] for b in bins] == [-40, -30, -20, -10, 0, 10, 20, 30]
    assert value[0] == 100 and value[-20] == 50
    assert value[30] == 10 and value[-40] == 5  # the tails
    assert sum(value.values()) == 165  # no reference, or nothing filled: not counted


def test_costs_are_weighted_by_filled_value() -> None:
    fills = [
        _fill(None, 300, arrival=10),
        _fill(None, 100, arrival=-10),
        _fill(None, 0, arrival=99),
    ]
    assert _weighted(fills, "vs_arrival_bps") == pytest.approx(5.0)
    assert _weighted([_fill(None, 100)], "vs_arrival_bps") is None


def test_the_earliest_pass_counts_one_session_a_day_a_day_late() -> None:
    """Each session counts once its equity arrives, the next trading day."""
    thu, fri = date(2026, 10, 8), date(2026, 10, 9)
    # Thu, Fri and Mon still needed: Monday's equity arrives Tuesday.
    assert _earliest_pass(thu, days=17, min_days=20, today=thu) == date(2026, 10, 13)
    assert _earliest_pass(fri, days=17, min_days=20, today=thu) == date(2026, 10, 14)
    # A Saturday start skips the weekend.
    assert _earliest_pass(date(2026, 10, 10), days=19, min_days=20, today=fri) == date(2026, 10, 13)
    assert _earliest_pass(thu, days=20, min_days=20, today=thu) == thu


async def _seed(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO forward_books (name, strategy, params, created_at) "
                "VALUES ('core', 'core-strict-cap', '{}', now())"
            )
        )
        for day, nav, equity in ((MON, 1.0, 1000.0), (TUE, 1.01, 1010.0), (WED, 1.02, 1020.0)):
            await conn.execute(
                text(
                    "INSERT INTO forward_book_days (book, day, nav, day_return, turnover, weights, "
                    "rebalance_next, recorded_at) VALUES ('core', :d, :n, 0, 0, "
                    '\'{"MSFT": 0.6, "NVDA": 0.4}\', false, now())'
                ),
                {"d": day, "n": nav},
            )
            await conn.execute(
                text(
                    "INSERT INTO broker_equity (account, day, equity, profit_loss, "
                    "profit_loss_pct, synced_at) VALUES ('core', :d, :e, 0, 0, now())"
                ),
                {"d": day, "e": equity},
            )
        # Monday's monthly run bought MSFT (filled) and NVDA (never filled);
        # Tuesday has no run; Wednesday's ran with nothing to trade.
        for day, monthly, orders in ((MON, True, 2), (WED, False, 0)):
            await conn.execute(
                text(
                    "INSERT INTO core_runs (account, run_on, monthly, executed, equity, cash, "
                    "orders, recorded_at) VALUES ('core', :d, :m, true, 1000, 100, :o, now())"
                ),
                {"d": day, "m": monthly, "o": orders},
            )
        at = datetime(2026, 10, 5, 15, 40, 30, tzinfo=MARKET_TZ)
        for symbol, qty, price, order_id in (("MSFT", 1.5, 400, "o1"), ("NVDA", 1, 200, "o2")):
            await conn.execute(
                text(
                    "INSERT INTO core_orders (account, submitted_at, symbol, side, qty, est_price, "
                    "notional, reason, status, broker_order_id) VALUES ('core', :t, :s, 'buy', "
                    ":q, :p, :n, 'rebalance', 'submitted', :id)"
                ),
                {"t": at, "s": symbol, "q": qty, "p": price, "n": qty * price, "id": order_id},
            )
        fill_at = datetime(2026, 10, 5, 15, 41, tzinfo=MARKET_TZ)
        for fid, symbol, qty, price, order_id in (
            ("f1", "MSFT", 1.5, 401, "o1"),
            ("f2", "CRDO", 2, 100, None),  # held from before, now failing the screen
        ):
            await conn.execute(
                text(
                    "INSERT INTO broker_activities (id, account, activity_type, transaction_time, "
                    "symbol, side, qty, price, order_id, raw) VALUES (:id, 'core', 'FILL', :t, "
                    ":s, 'buy', :q, :p, :o, '{}')"
                ),
                {"id": fid, "t": fill_at, "s": symbol, "q": qty, "p": price, "o": order_id},
            )
        bars = [
            ("MSFT", MON, "raw", 402.0),
            ("MSFT", WED, "raw", 404.0),
            ("CRDO", WED, "raw", 110.0),
            ("SPUS", TUE, "raw", 40.0),
            ("SPUS", WED, "raw", 41.0),
            ("SPUS", MON, "all", 40.0),
            ("SPUS", WED, "all", 42.0),
        ]
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :c, :c, :c, :c, 1, now())"
            ),
            [{"s": s, "d": d, "a": a, "c": c} for s, d, a, c in bars],
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, :s, :c, :sic, :v, "
                "CAST(:r AS JSONB), '{}', 't', now())"
            ),
            [
                {"a": WED, "s": "MSFT", "c": 1, "sic": SOFTWARE, "v": "halal", "r": "[]"},
                {"a": WED, "s": "NVDA", "c": 2, "sic": SEMIS, "v": "halal", "r": "[]"},
                {
                    "a": WED,
                    "s": "CRDO",
                    "c": 3,
                    "sic": SEMIS,
                    "v": "not_halal",
                    "r": '["debt 40% of market cap > 30%"]',
                },
            ],
        )


async def test_held_names_that_fail_the_screen_are_marked_to_sell_with_the_reason(engine) -> None:
    await _seed(engine)
    body = await build(engine, SETTINGS, now=NOW)

    by_symbol = {h["symbol"]: h for h in body["holdings"]}
    assert by_symbol["CRDO"]["to_sell"] and by_symbol["CRDO"]["verdict"] == "not_halal"
    assert body["to_sell"] == [
        {"symbol": "CRDO", "name": None, "reason": "debt 40% of market cap > 30%"}
    ]
    assert not by_symbol["MSFT"]["to_sell"]
    assert not by_symbol["NVDA"]["to_sell"]  # a target never bought is not a sale
    assert by_symbol["MSFT"]["value"] == 606.0 and by_symbol["MSFT"]["sector"] == sector_of(
        SOFTWARE
    )
    # MSFT: 606 of 1020 against a 60% target, banded at a quarter of it.
    assert by_symbol["MSFT"]["band"] == pytest.approx(0.15)
    assert by_symbol["MSFT"]["in_band"] is True
    assert body["positions"] == 2 and body["invested"] == 826.0

    sectors = {s["sector"]: s for s in body["sectors"]}
    assert sectors[sector_of(SEMIS)]["names"] == ["CRDO"]
    assert sectors[sector_of(SOFTWARE)]["value"] == 606.0
    assert body["sectors"][0]["sector"] == sector_of(SOFTWARE)  # the biggest first


async def test_the_gate_strip_marks_each_day_of_its_window(engine) -> None:
    await _seed(engine)
    body = await build(engine, SETTINGS, now=NOW)
    window = {date.fromisoformat(d["day"]): d["status"] for d in body["readiness"]["window"]}

    assert len(window) == 20
    assert window[MON] == "run" and window[WED] == "run"
    assert window[TUE] == "missed"
    assert window[date(2026, 10, 8)] == "pending"  # today, before 15:40
    assert window[date(2026, 10, 2)] == "before"  # the account did not exist yet
    assert body["readiness"]["earliest"] >= body["readiness"]["earliest_days"]


async def test_each_run_carries_its_orders_and_how_they_filled(engine) -> None:
    await _seed(engine)
    body = await build(engine, SETTINGS, now=NOW)

    runs = {r["run_on"]: r for r in body["runs"]}
    monday = runs["2026-10-05"]
    assert [o["symbol"] for o in monday["order_rows"]] == ["MSFT", "NVDA"]
    assert monday["notional"] == 800.0 and monday["filled"] == 1
    msft = monday["order_rows"][0]
    assert msft["fill_status"] == "filled" and msft["fill_price"] == 401.0
    assert msft["vs_arrival_bps"] == 25.0  # bought 401 against the plan's 400
    assert msft["vs_close_bps"] == pytest.approx(-24.9, abs=0.1)  # below the 402 close
    assert monday["order_rows"][1]["fill_status"] == "unfilled"
    assert runs["2026-10-07"]["order_rows"] == [] and runs["2026-10-07"]["notional"] is None

    hist = body["execution"]["histogram"]
    assert sum(b["value"] for b in hist) == pytest.approx(601.5)
    assert next(b for b in hist if b["from_bps"] == -30)["value"] == pytest.approx(601.5)


async def test_benchmarks_ride_the_series_and_today(engine) -> None:
    await _seed(engine)
    body = await build(engine, SETTINGS, now=NOW)

    assert body["series"][-1]["spus"] == 105.0  # 42 against Monday's 40, adjusted
    assert body["since"]["spus_pct"] == pytest.approx(0.05)
    assert body["since"]["change_pct"] == pytest.approx(0.02)
    assert body["since"]["book_pct"] == pytest.approx(0.02)
    spus = next(b for b in body["today_vs"] if b["symbol"] == "SPUS")
    assert spus["change_pct"] == pytest.approx(0.025)
    assert spus["diff_pts"] is None  # the ledger has no intraday move to set beside it


async def test_today_against_the_benchmarks_uses_the_snapshots_move(engine) -> None:
    await _seed(engine)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO account_snapshots (account, taken_at, equity, cash, last_equity, "
                "positions) VALUES ('core', :t, 1030.0, 50, 1020.0, "
                '\'[{"symbol": "MSFT", "qty": 1.5, "market_value": 610, "change_today": 0.01}]\')'
            ),
            {"t": NOW},
        )
    body = await build(engine, SETTINGS, now=NOW)

    assert body["equity_source"] == "live" and body["change"] == 10.0
    spus = next(b for b in body["today_vs"] if b["symbol"] == "SPUS")
    assert spus["diff_pts"] == pytest.approx(10 / 1020 - 0.025, abs=1e-4)
    assert body["cash"] == 50.0
    assert [h["symbol"] for h in body["holdings"] if h["shares"]] == ["MSFT"]
