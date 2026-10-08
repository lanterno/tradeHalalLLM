"""One list of broker accounts, and one unpaid-purification total over them."""

from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.purification import unpaid
from halal_trader.portfolio.core_account import DAY_TRADER, broker_accounts


def _settings(*, core_keys: bool, core_paper: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        alpaca=SimpleNamespace(api_key="dk", secret_key="ds", paper_trade=True),
        core=SimpleNamespace(
            alpaca_api_key="ck" if core_keys else "",
            alpaca_secret_key="cs" if core_keys else "",
            paper=core_paper,
            enabled=core_keys,
        ),
    )


def test_the_day_trader_always_and_the_core_once_its_keys_are_set() -> None:
    assert [a.name for a in broker_accounts(_settings(core_keys=False))] == [DAY_TRADER]
    paper = broker_accounts(_settings(core_keys=True))
    assert [(a.name, a.label, a.api_key) for a in paper] == [
        ("paper", "day-trader", "dk"),
        ("core", "core", "ck"),
    ]
    live = broker_accounts(_settings(core_keys=True, core_paper=False))
    assert (live[1].name, live[1].paper) == ("core-live", False)


async def test_unpaid_purification_counts_broker_accounts_not_books(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for account, amount, paid in (
            ("paper", 1.0, False),
            ("core", 2.0, False),
            ("core-live", 4.0, False),
            ("core", 8.0, True),
            ("book:s1", 16.0, False),
        ):
            await conn.execute(
                text(
                    "INSERT INTO purification_accruals (account, dividend_id, symbol, ex_date, "
                    "shares, dividend, impure_ratio, amount, method, accrued_at, paid_at) VALUES "
                    "(:a, :d, 'MSFT', '2026-09-01', 1, 1, 0.01, :x, 't', now(), "
                    "CASE WHEN :p THEN now() END)"
                ),
                {"a": account, "d": f"{account}-{amount}", "x": amount, "p": paid},
            )

    assert await unpaid(engine) == 7.0
    assert await unpaid(engine, ["core"]) == 2.0
