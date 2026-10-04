"""`halal-trader zakat`: each year's zakat by both of Dar al-Ifta's methods."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from halal_trader.logging import console


@click.group("zakat")
def zakat() -> None:
    """Zakat on the portfolio: trade-goods and income methods, the higher chosen."""


@zakat.command("assess")
@click.option("--account", default="paper", show_default=True, help='"paper" or "book:<name>".')
@click.option(
    "--as-of",
    type=click.DateTime(["%Y-%m-%d"]),
    default=None,
    help="Assess the hawl falling on or before this date (default: today).",
)
@click.option("--record", is_flag=True, help="Store the result as that year's assessment.")
def assess_cmd(account: str, as_of: Any, record: bool) -> None:
    """Show zakat for the latest hawl by both methods, and which is higher."""

    async def _run() -> Any:
        from halal_trader.compliance import zakat as z
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.market_hours import today_eastern

        settings = get_settings()
        if not settings.zakat.hawl_hijri:
            raise click.ClickException(
                "set ZAKAT_HAWL_HIJRI (MM-DD, the Hijri day your zakat year ends) in .env"
            )
        day = as_of.date() if as_of else today_eastern()
        start, end = z.hawl_period(z.parse_hawl(settings.zakat.hawl_hijri), day)
        engine = await init_db(settings.database_url)
        try:
            a = await z.assess(engine, account, period_start=start, hawl_date=end)
            if record:
                await z.record(engine, a)
            return a
        finally:
            await engine.dispose()

    from halal_trader.compliance.purification import BOOK_NOTIONAL
    from halal_trader.compliance.zakat import SOURCE, hijri_label

    a = asyncio.run(_run())
    unit = f" (per ${BOOK_NOTIONAL:,.0f} following the book)" if account.startswith("book:") else ""
    console.print(
        f"[bold]{account}[/bold]{unit}: zakat year {a.period_start} -> {a.hawl_date} "
        f"({hijri_label(a.hawl_date)})"
    )
    mark_tg = "  <- higher" if a.chosen == "trade goods" else ""
    mark_in = "  <- higher" if a.chosen == "income" else ""
    console.print(
        f"  trade goods: 2.5% of market value ${a.market_value:,.2f} = "
        f"[bold]${a.trade_goods:,.2f}[/bold]{mark_tg}"
    )
    console.print(
        f"  income:      2.5% of dividends ${a.dividends:,.2f} less purified ${a.purified:,.2f} "
        f"= [bold]${a.income:,.2f}[/bold]{mark_in}"
    )
    console.print(
        f"  [bold]due: ${a.amount:,.2f}[/bold] ({a.chosen}){' — recorded' if record else ''}"
    )
    console.print(
        f"  {SOURCE}. Not included: the nisab test and cash held, which concern your whole "
        "wealth; add this to the rest of it."
    )
