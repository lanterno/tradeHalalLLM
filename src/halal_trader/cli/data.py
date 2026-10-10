"""`halal-trader data`: the research market-data store (data/)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, TypeVar

import click

from halal_trader.cli._run import fail, run_db
from halal_trader.logging import console

T = TypeVar("T")


def _with_store(work: Callable[[Any, Any], Awaitable[T]]) -> T:
    """Run ``work(engine, client)`` with a DB engine and a market-data client."""

    async def _run(engine: Any, settings: Any) -> T:
        from halal_trader.data.alpaca_market import AlpacaMarketData

        client = AlpacaMarketData.from_settings(settings)
        try:
            return await work(engine, client)
        finally:
            await client.aclose()

    return run_db(_run)


@click.group("data")
def data() -> None:
    """Research market data: Alpaca assets and 10 years of daily SIP bars."""


@data.command("assets")
def assets_cmd() -> None:
    """Sync Alpaca's active US-equity asset list."""
    from halal_trader.data.store import sync_assets

    n = _with_store(sync_assets)
    console.print(f"{n} active assets synced")


@data.command("backfill")
@click.option("--top", default=1500, show_default=True, help="Most liquid names to keep.")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2016-01-01",
    show_default=True,
    help="First session to fetch.",
)
def backfill_cmd(top: int, since: Any) -> None:
    """Pick the liquid universe, then fetch raw + adjusted daily bars for it.

    Resumable: symbols already stored continue from their last session.
    """
    from halal_trader.data.store import BENCHMARKS, liquid_universe, sync_assets, update_bars
    from halal_trader.market_hours import today_eastern

    async def work(engine: Any, client: Any) -> tuple[int, dict[str, int]]:
        await sync_assets(engine, client)
        members = await liquid_universe(engine, client, top_n=top, as_of=today_eastern())
        symbols = sorted({m.symbol for m in members} | set(BENCHMARKS))
        stored = await update_bars(engine, client, symbols, since=since.date())
        return len(symbols), stored

    n, stored = _with_store(work)
    console.print(f"{n} symbols; rows stored: {stored}")


@data.command("update")
def update_cmd() -> None:
    """Top up every stored symbol to the latest closed session."""

    from halal_trader.data.store import update_bars

    async def work(engine: Any, client: Any) -> tuple[int, dict[str, int]]:
        from halal_trader.data.store import stored_symbols

        symbols = await stored_symbols(engine)
        if not symbols:
            return 0, {}
        return len(symbols), await update_bars(engine, client, symbols, since=date(2016, 1, 1))

    n, stored = _with_store(work)
    console.print(
        f"{n} symbols topped up; rows stored: {stored}"
        if n
        else "No bars stored yet: run `data backfill`."
    )


@data.command("pit-universe")
@click.option("--top", default=1000, show_default=True, help="Names in the universe each month.")
@click.option(
    "--since",
    type=click.DateTime(["%Y-%m-%d"]),
    default="2016-01-01",
    show_default=True,
    help="First month of the universe; monthly bars start a year earlier.",
)
def pit_universe_cmd(top: int, since: Any) -> None:
    """Point-in-time universe: monthly bars for every listed and delisted stock,
    the top names by trailing dollar volume each month, daily bars for all of them.

    Resumable: daily bars continue from each symbol's last stored session.
    """
    from halal_trader.data.store import BENCHMARKS, update_bars
    from halal_trader.data.universe import sync_monthly_bars, universe_history
    from halal_trader.market_hours import today_eastern

    start = since.date()

    async def work(engine: Any, client: Any) -> tuple[int, int, int, dict[str, int]]:
        symbols, rows = await sync_monthly_bars(
            engine, client, since=start.replace(year=start.year - 1)
        )
        history = await universe_history(engine, start=start, end=today_eastern(), top_n=top)
        members = sorted({s for names in history.values() for s in names} | set(BENCHMARKS))
        stored = await update_bars(engine, client, members, since=start)
        return symbols, rows, len(members), stored

    symbols, rows, members, stored = _with_store(work)
    console.print(f"monthly bars: {rows} rows for {symbols} symbols (listed and delisted)")
    console.print(f"{members} names were ever in the top {top}; daily rows stored: {stored}")


# events/units.PARTS, spelled out so `--help` does not import the planner (a test keeps them equal).
MINUTE_PARTS = (
    "spy",
    "gate_g1",
    "gate_calib",
    "gate_sue",
    "gate_reactor",
    "train",
    "validation",
)


@data.command("minutes")
@click.option(
    "--plan",
    "plan_name",
    type=click.Choice(["h1"]),
    required=True,
    help="Which unit plan: h1 is the news engine's Phase 0/1 plan (spec §H).",
)
@click.option(
    "--part",
    "parts",
    multiple=True,
    type=click.Choice(MINUTE_PARTS),
    help="Only these parts (repeatable); every part by default.",
)
@click.option("--dry-run", is_flag=True, help="Print the counts and the estimate; fetch nothing.")
@click.option(
    "--rate", default=100, show_default=True, type=click.IntRange(min=1), help="Requests a minute."
)
@click.option(
    "--force", is_flag=True, help="Fetch during market hours or the research job's window too."
)
def minutes_cmd(
    plan_name: str, parts: tuple[str, ...], dry_run: bool, rate: int, force: bool
) -> None:
    """Fetch the minute bars of a unit plan, part by part (resumable).

    Each (symbol, session) unit is marked done once fetched, empty ones
    included, so a rerun asks only for what is missing. A real run stops
    at the next session batch when US market hours or the evening research
    job begin (unless --force); run it again later to resume.
    """

    async def _plan(engine: Any) -> tuple[Any, Counter[str]]:
        from halal_trader.events import units

        counts: Counter[str] = Counter()
        plan = await units.h1_plan(engine, parts=parts or None, counts=counts)
        return plan, counts

    if dry_run:

        async def _dry(engine: Any, settings: Any) -> tuple[Any, Counter[str], set[str]]:
            from halal_trader.data.minutes import done_units

            plan, counts = await _plan(engine)
            return plan, counts, await done_units(engine)

        plan, counts, done = run_db(_dry)
        for line in _dry_run_lines(plan, counts, done, rate):
            console.print(line, highlight=False)
        return

    from datetime import UTC, datetime

    from halal_trader.events import units

    if not force and (why := units.busy(datetime.now(UTC))) is not None:
        fail(f"not fetching during {why}; run later, or pass --force")

    async def _fetch(engine: Any, settings: Any) -> Any:
        from halal_trader.data.alpaca_market import AlpacaMarketData

        plan, _ = await _plan(engine)
        console.print(f"plan {plan_name}: {len(plan.all())} units in {len(plan.parts)} part(s)")
        market = AlpacaMarketData.from_settings(settings, min_interval_s=60.0 / rate)
        try:
            return await units.fetch(
                engine,
                market,
                plan,
                stop=None if force else units.busy,
                on_part=lambda part, n, bars: console.print(
                    f"  {part}: {n} unit(s) fetched, {bars} bar(s) stored"
                ),
            )
        finally:
            await market.aclose()

    report = run_db(_fetch)
    if report.stopped is not None:
        fail(f"stopped at {report.stopped}; run the command again later to resume")
    console.print(
        f"done: {sum(report.fetched.values())} unit(s) fetched, "
        f"{sum(report.stored.values())} bar(s) stored"
    )


def _dry_run_lines(plan: Any, counts: Counter[str], done: set[str], rate: int) -> list[str]:
    from halal_trader.events import units

    rows = units.estimate(plan, done)
    lines = [
        f"plan h1 (seed {units.SEED}), parts in fetch order; requests at "
        f"{units.SYMBOLS_PER_REQUEST} symbols and {units.PAGE_BARS:,} bars a page, "
        f"{units.BARS_PER_UNIT} bars a unit",
        f"{'part':<13}{'units':>8}{'new':>8}{'done':>8}{'fetch':>8}{'sessions':>9}"
        f"{'requests':>9}  {'first':<11}{'last':<11}sha",
    ]
    for r in rows:
        lines.append(
            f"{r.part:<13}{r.units:>8}{r.new:>8}{r.done:>8}{r.to_fetch:>8}{r.sessions:>9}"
            f"{r.requests:>9}  {r.first or '-'!s:<11}{r.last or '-'!s:<11}{r.sha[:12]}"
        )
    unique = len(plan.all())
    fetch = sum(r.to_fetch for r in rows)
    requests = sum(r.requests for r in rows)
    lines.append(
        f"unique units {unique}; to fetch {fetch}; about {requests} request(s), "
        f"{units.hours_at(requests, rate):.1f} h at {rate} a minute"
    )
    if counts:
        lines.append("selection: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for part, n in sorted(plan.outside_gate_ranges().items()):
        lines.append(f"warning: {part} has {n} unit(s) outside its gate's dates")
    return lines


@data.command("fundamentals")
@click.option("--since", default=2014, show_default=True, help="First calendar year.")
def fundamentals_cmd(since: int) -> None:
    """Sync annual gross profit and total assets for every SEC filer (the quality input)."""

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.compliance.sec import SecClient
        from halal_trader.data.fundamentals import sync_annual, usable_year
        from halal_trader.market_hours import today_eastern

        sec = SecClient(settings.edgar.user_agent)
        try:
            last = usable_year(today_eastern())
            return await sync_annual(sec, engine, range(since, last + 1))
        finally:
            await sec.aclose()

    console.print(f"{run_db(_run)} filer-years stored")
