"""The dashboard's home page: sectors, live snapshots and the assembled view."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.sectors import OTHER, sector_of
from halal_trader.portfolio import home, snapshots
from halal_trader.web import app as web_app


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        ("Services-Prepackaged Software", "Software & internet"),
        ("Electronic Computers", "Computer hardware"),
        ("Semiconductors & Related Devices", "Semiconductors"),
        ("Pharmaceutical Preparations", "Healthcare"),
        ("Retail-Variety Stores", "Retail"),  # "stores" is not "ores"
        ("Gold and Silver Ores", "Materials & mining"),
        ("Aircraft Engines & Engine Parts", "Industrials"),
        ("Air Transportation, Scheduled", "Transport"),
        ("Real Estate Investment Trusts", "Real estate"),
        ("not an SEC registrant (or ticker not mapped)", OTHER),
        (None, OTHER),
    ],
)
def test_sic_descriptions_fold_into_plain_sectors(sic: str | None, sector: str) -> None:
    assert sector_of(sic) == sector


def test_holiday_names_and_trading_days() -> None:
    assert home.holiday_name(date(2026, 11, 26)) == "Thanksgiving"
    assert home.holiday_name(date(2027, 3, 26)) == "Good Friday"
    assert home.trading_days_in_month(date(2026, 10, 5)) == (22, 19)
    assert home.trading_days_in_month(date(2026, 11, 2))[0] == 20  # Thanksgiving off
    assert home.first_trading_day_of_next_month(date(2026, 10, 5)) == date(2026, 11, 2)


def test_company_names_lose_the_share_description() -> None:
    assert home.clean_name("Apple Inc. Common Stock") == "Apple Inc."
    assert home.clean_name("Alphabet Inc. Class A Common Stock") == "Alphabet Inc. Class A"
    assert home.clean_name(None) is None


def test_the_schedule_is_ordered_and_names_the_next_monthly_rebalance() -> None:
    now = datetime.fromisoformat("2026-10-05T19:30:00-04:00")  # Monday evening
    items = home.upcoming(
        now,
        core_enabled=True,
        monthly_done=True,
        last_screen=date(2026, 10, 1),
        next_hawl=date(2027, 2, 8),
    )
    labels = [i["label"] for i in items]
    assert labels[0] == "Evening research run"
    assert labels.index("Core daily check") < labels.index("Next monthly rebalance")
    assert [i["at"] for i in items] == sorted(i["at"] for i in items)
    assert any(i["at"].startswith("2026-11-02T15:40") for i in items)
    assert any(i["at"].startswith("2026-10-09T17:15") for i in items)  # Friday's digest


def test_an_account_without_history_counts_at_its_current_value() -> None:
    rows = [("paper", date(2026, 10, 1), 100.0), ("paper", date(2026, 10, 2), 110.0)]
    series = home.total_series(rows, ["paper", "core"], date(2026, 9, 1), {"core": 50.0})
    assert [p["equity"] for p in series] == [150.0, 160.0]


def test_snapshot_rows_parse_alpaca_shapes() -> None:
    row = snapshots.position_row(
        {
            "symbol": "AAPL",
            "qty": "1.5",
            "market_value": "300",
            "current_price": "200",
            "lastday_price": "198",
            "change_today": "0.0101",
            "unrealized_pl": "3",
        }
    )
    assert row["market_value"] == 300.0 and row["change_today"] == pytest.approx(0.0101)
    assert snapshots.quote_row("SPUS", {"latestTrade": {"p": 60.5}, "prevDailyBar": {"c": 60}}) == {
        "symbol": "SPUS",
        "price": 60.5,
        "prev_close": 60.0,
    }
    assert snapshots.quote_row("SPUS", {}) is None
    after_hours = {"latestTrade": {"p": 61.2}, "dailyBar": {"c": 60.8}, "prevDailyBar": {"c": 60}}
    assert snapshots.quote_row("SPUS", after_hours, in_session=False)["price"] == 60.8
    assert snapshots.quote_row("SPUS", after_hours, in_session=True)["price"] == 61.2


class FakeBroker:
    async def account_payload(self) -> dict:
        return {"equity": "105500", "cash": "1050", "last_equity": "105000"}

    async def positions_payload(self) -> list[dict]:
        return [
            {
                "symbol": "AAPL",
                "qty": "60",
                "market_value": "16000",
                "current_price": "266.6",
                "lastday_price": "264",
                "change_today": "0.0098",
            },
            {
                "symbol": "MSFT",
                "qty": "30",
                "market_value": "13000",
                "current_price": "433.3",
                "lastday_price": "435",
                "change_today": "-0.0039",
            },
        ]

    async def get_stock_snapshot(self, symbols: str) -> dict:
        return {
            s: {"latestTrade": {"p": 100.0}, "prevDailyBar": {"c": 99.0}}
            for s in symbols.split(",")
        }


async def _seed_screen(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES ('2026-10-01', :s, :sic, :v, '[]', "
                "'{}', 'test', now())"
            ),
            [
                {"s": "AAPL", "sic": "Electronic Computers", "v": "halal"},
                {"s": "MSFT", "sic": "Services-Prepackaged Software", "v": "not_halal"},
            ],
        )


async def test_the_home_view_reads_the_live_snapshot(engine: AsyncEngine) -> None:
    await snapshots.snapshot_account(engine, "core", FakeBroker())
    await snapshots.snapshot_quotes(engine, FakeBroker())
    await _seed_screen(engine)
    settings = SimpleNamespace(
        core=SimpleNamespace(enabled=True, paper=True),
        zakat=SimpleNamespace(hawl_hijri=""),
    )
    now = datetime.now(UTC)

    view = await home.build(engine, settings, now=now)

    core = view["accounts"][0]
    assert core["account"] == "core" and core["source"] == "live"
    assert core["change"] == 500.0 and core["invested"] == 104450.0
    assert view["total"]["equity"] == 105500.0
    assert {b["symbol"] for b in view["market"]["benchmarks"]} == {"SPUS", "HLAL", "SPY"}
    assert all(b["live"] for b in view["market"]["benchmarks"])
    held = view["portfolio"]
    assert [h["symbol"] for h in held["holdings"]] == ["AAPL", "MSFT"]
    assert held["holdings"][0]["change_today"] == pytest.approx(0.0098)
    assert {s["sector"] for s in held["sectors"]} == {"Computer hardware", "Software & internet"}
    assert held["screen"]["failing"] == ["MSFT"]  # held, and the newest screen fails it


async def test_set_aside_counts_broker_accounts_and_zakat_is_an_estimate(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, shares, "
                "dividend, impure_ratio, amount, method, accrued_at) VALUES "
                "('core', 'd1', 'AAA', '2026-09-01', 1, 10, 0.1, 1.0, 'm', now()), "
                "('paper', 'd2', 'AAA', '2026-09-01', 1, 10, 0.1, 2.0, 'm', now()), "
                "('book:core', 'd3', 'AAA', '2026-09-01', 1, 10, 0.1, 40.0, 'm', now())"
            )
        )
    settings = SimpleNamespace(
        core=SimpleNamespace(enabled=True, paper=True),
        zakat=SimpleNamespace(hawl_hijri="09-01"),
    )
    view = await home.build(engine, settings, now=datetime(2026, 10, 6, 15, tzinfo=UTC))
    # The forward book's $40 is per a notional $10,000 nobody holds.
    assert view["set_aside"]["purification_unpaid"] == 3.0
    assert view["set_aside"]["zakat"]["estimate"] is True


async def test_a_live_core_is_shown_from_its_own_account(engine: AsyncEngine) -> None:
    await snapshots.snapshot_account(engine, "core", FakeBroker())  # the paper history
    settings = SimpleNamespace(
        core=SimpleNamespace(enabled=True, paper=False),
        zakat=SimpleNamespace(hawl_hijri=""),
    )
    view = await home.build(engine, settings, now=datetime.now(UTC))
    assert view["accounts"] == []  # paper's figures are not the live account's
    await snapshots.snapshot_account(engine, "core-live", FakeBroker())
    view = await home.build(engine, settings, now=datetime.now(UTC))
    assert [a["account"] for a in view["accounts"]] == ["core-live"]
    assert view["accounts"][0]["paper"] is False


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    from halal_trader import config

    monkeypatch.setattr(config, "_settings", None)
    app = web_app.create_app()
    with TestClient(app) as c:
        yield c
    config._settings = None


def test_the_home_endpoint_answers_on_an_empty_database(client) -> None:
    body = client.get("/api/home").json()
    assert body["accounts"] == [] and body["portfolio"]["count"] == 0
    assert body["market"]["trading_days_month"] > 0
    assert body["market"]["holidays"] == sorted(body["market"]["holidays"])
    assert body["upcoming"]
