"""`halal-trader events`: the event store's labels and what the scores knew."""

from __future__ import annotations

from typing import Any

import click

from halal_trader.cli._run import run_db
from halal_trader.logging import console


@click.group("events")
def events() -> None:
    """News and filings as dated, scored, labelled events (S2 research)."""


@events.command("report")
@click.option("--threshold", default=0.85, show_default=True, help="The reactor's trigger.")
def report_cmd(threshold: float) -> None:
    """Label matured events, then show each scorer's IC against abnormal returns."""

    async def _run(engine: Any, settings: Any) -> tuple[int, list[Any], dict[str, int]]:
        from sqlalchemy import text

        from halal_trader.events.labels import label_events, report

        written = await label_events(engine)
        async with engine.connect() as conn:
            counts = {
                "events": (await conn.execute(text("SELECT count(*) FROM events"))).scalar() or 0,
                "scored": (await conn.execute(text("SELECT count(*) FROM event_scores"))).scalar()
                or 0,
            }
        return written, await report(engine, threshold=threshold), counts

    written, rows, counts = run_db(_run)
    console.print(f"{counts['events']} events, {counts['scored']} scores; {written} new label(s)")
    if not rows:
        console.print("[yellow]no labelled scores yet: horizons need sessions to elapse[/yellow]")
        return
    for r in rows:
        above = f"{r.above_abn:+.2%}" if r.above_abn is not None else "  n/a"
        below = f"{r.below_abn:+.2%}" if r.below_abn is not None else "  n/a"
        console.print(
            f"  {r.scorer:40} {r.horizon:>2}d  n={r.events:<5} IC {r.ic:+.3f}  "
            f">= {threshold:g}: {r.above} events, mean abn {above}; below: {below}"
        )


@events.command("backfill")
@click.argument("what", type=click.Choice(["news", "filings", "insiders", "eps", "all"]))
@click.option(
    "--rate",
    default=100,
    show_default=True,
    help="Alpaca requests per minute for news (the live bot shares the key's 200/min).",
)
def backfill_cmd(what: str, rate: int) -> None:
    """Fill the event store's history (resumable; finished units are skipped)."""

    async def _run(engine: Any, settings: Any) -> dict[str, int]:
        from halal_trader.compliance.sec import SecClient
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.events import history

        sec = SecClient(settings.edgar.user_agent)
        out: dict[str, int] = {}
        try:
            companies = await history.covered_companies(engine)
            if what in ("filings", "all"):
                out["filings"] = await history.backfill_filings(engine, sec, companies)
            if what in ("insiders", "all"):
                out["insiders"] = await history.backfill_insiders(engine, sec, companies)
            if what in ("eps", "all"):
                out["eps"] = await history.backfill_eps(engine, sec, companies)
            if what in ("news", "all"):
                market = AlpacaMarketData.from_settings(
                    settings,
                    min_interval_s=60.0 / max(rate, 1),
                )
                try:
                    symbols = await history.covered_symbols(engine)
                    out["news"] = await history.backfill_news(engine, market, symbols=symbols)
                finally:
                    await market.aclose()
        finally:
            await sec.aclose()
        return out

    for name, n in run_db(_run).items():
        console.print(f"{name}: {n} new")


@events.command("extract")
def extract_cmd() -> None:
    """Read earnings results and guidance vs consensus out of stored headlines."""

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.events.earnings_parse import extract_all

        return await extract_all(engine)

    console.print(f"{run_db(_run)} earnings fact(s) extracted")


@events.command("quality")
def quality_cmd() -> None:
    """Coverage by year and source, earnings-timestamp agreement, duplicate rate."""

    async def _run(engine: Any, settings: Any) -> tuple[list[Any], Any, float | None]:
        from halal_trader.events.quality import coverage, duplicate_share, timing

        return await coverage(engine), await timing(engine), await duplicate_share(engine)

    rows, t, dup = run_db(_run)
    console.print("year  source      kind           events  companies / universe")
    for c in rows:
        share = f"{c.companies / c.universe:.0%}" if c.universe else "  -"
        console.print(
            f"{c.year}  {c.source:10}  {c.kind:13} {c.events:8}  {c.companies:5} / "
            f"{c.universe:<5} {share}"
        )
    if t.pairs:
        console.print(
            f"earnings timing: {t.pairs} releases seen as both headline and 8-K 2.02; "
            f"median headline - 8-K {t.median_minutes:+.0f} min, "
            f"{t.within_hour:.0%} within an hour, headline first in {t.headline_first:.0%}"
        )
    else:
        console.print("earnings timing: no pairs yet (run `events extract`)")
    if dup is not None:
        console.print(f"duplicate headlines (same symbol, day, text): {dup:.1%}")


@events.command("study")
@click.argument("signal", type=click.Choice(["sue"]))
@click.option("--start", type=int, default=2016, show_default=True, help="First year.")
@click.option("--end", type=int, default=2019, show_default=True, help="Last year.")
@click.option("--by", type=click.Choice(["all", "bucket", "year"]), default="bucket")
def study_cmd(signal: str, start: int, end: int, by: str) -> None:
    """Event study of a free signal: net abnormal return by signal decile and horizon."""

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.history import covered_companies
        from halal_trader.events.study import Observation, evaluate, summarise
        from halal_trader.events.sue import sue_observations

        raw = await sue_observations(engine, await covered_companies(engine))
        obs = [
            Observation(o.symbol, o.announced_at, o.sue)
            for o in raw
            if start <= o.announced_at.year <= end
        ]
        return summarise(await evaluate(engine, obs), by=None if by == "all" else by)

    result = run_db(_run)
    console.print(f"{signal} {start}-{end}: net abnormal return vs SPY by decile (t-stat)")
    groups = sorted({r.group for r in result.rows})
    for group in groups:
        console.print(f"[bold]{group}[/bold] (n={result.n.get(group, 0)})")
        for h in sorted({r.horizon for r in result.rows if r.group == group}):
            cells = [r for r in result.rows if r.group == group and r.horizon == h]
            line = "  ".join(f"D{r.decile} {r.mean:+.2%}({r.t:+.1f})" for r in cells)
            top = next((r for r in cells if r.decile == 10), None)
            bot = next((r for r in cells if r.decile == 1), None)
            spread = f"{top.mean - bot.mean:+.2%}" if top and bot else "n/a"
            console.print(
                f"  {h:>2}d IC {result.ic.get((group, h), 0):+.3f}  D10-D1 {spread}  | {line}"
            )


def _research_meter(engine: Any, settings: Any) -> None:
    """Meter this CLI process's LLM calls into the research pool (core/llm/spend.py)."""
    from halal_trader.core.llm import spend

    spend.install(spend.research_meter(engine, settings))


@events.command("cutoff-probe")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default="2024-01-01")
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default="2026-09-30")
@click.option("--per-month", default=10, show_default=True)
def cutoff_probe_cmd(start: Any, end: Any, per_month: int) -> None:
    """Find the LLM's training cutoff: its recall of reported EPS, by filing month."""

    async def _run(engine: Any, settings: Any) -> list[Any]:
        from halal_trader.core.llm import create_classifier_llm
        from halal_trader.events.llm_cutoff import probes, run

        _research_meter(engine, settings)
        items = await probes(engine, start=start.date(), end=end.date(), per_month=per_month)
        return await run(create_classifier_llm(settings), items)

    for m in run_db(_run):
        bar = "#" * m.correct
        console.print(
            f"  {m.month:%Y-%m}  asked {m.asked:2}  answered {m.answered:2}  "
            f"correct {m.correct:2}  {bar}"
        )


@events.command("llm-score")
@click.option("--max-pairs", default=200_000, show_default=True, help="Stop after this many.")
def llm_score_cmd(max_pairs: int) -> None:
    """Score post-cutoff company headlines with the LLM, in batches (research budget)."""

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.core.llm import create_classifier_llm
        from halal_trader.events.llm_score import score_all

        _research_meter(engine, settings)
        return await score_all(
            create_classifier_llm(settings),
            engine,
            model=settings.llm.model,
            max_pairs=max_pairs,
        )

    console.print(f"{run_db(_run)} headline/symbol pairs scored")


@events.command("llm-eval")
def llm_eval_cmd() -> None:
    """LLM vs lexicon scores of post-cutoff news: IC and deciles of net abnormal return."""

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.llm_eval import compare

        return await compare(engine)

    c = run_db(_run)
    console.print(f"{c.days} (symbol, day) readings of post-cutoff news")
    for label, result in (
        ("llm", c.llm),
        ("lexicon", c.lexicon),
        ("llm where lexicon neutral", c.llm_where_lexicon_neutral),
    ):
        console.print(f"[bold]{label}[/bold] (n={result.n.get('all', 0)})")
        for h in sorted({h for (_, h) in result.ic}):
            cells = {r.decile: r for r in result.rows if r.horizon == h}
            line = f"  {h:>2}d IC {result.ic[('all', h)]:+.3f}"
            if 10 in cells and 1 in cells:
                top, bot = cells[10], cells[1]
                line += (
                    f"  top decile {top.mean:+.2%} (t {top.t:+.1f})"
                    f"  bottom decile {bot.mean:+.2%} (t {bot.t:+.1f})"
                )
            console.print(line)


@events.command("intraday")
@click.option("--rate", default=80, show_default=True, help="Alpaca requests per minute.")
def intraday_cmd(rate: int) -> None:
    """The pre-registered "fast in" test: minute-bar entries 60 s after in-session headlines."""

    async def _run(engine: Any, settings: Any) -> list[Any]:
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.events.intraday import first_in_session, run, selection, summarise

        market = AlpacaMarketData.from_settings(settings, min_interval_s=60.0 / rate)
        try:
            chosen = selection(await first_in_session(engine))
            console.print(f"{len(chosen)} in-session headlines to study")
            return summarise(await run(engine, market, chosen))
        finally:
            await market.aclose()

    for b in run_db(_run):
        console.print(f"  {b.label:20} {b.horizon:9} n={b.n:<5} mean {b.mean:+.2%}  t {b.t:+.1f}")
