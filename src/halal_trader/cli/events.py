"""`halal-trader events`: the event store's labels and what the scores knew."""

from __future__ import annotations

from typing import Any

import click

from halal_trader.cli._run import fail, run_db
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


@events.group("renames")
def renames_group() -> None:
    """Renamed tickers: companies whose older news is filed under another ticker."""


@renames_group.command("seed")
def renames_seed_cmd() -> None:
    """List symbols whose first news comes over a year after their first halal screen."""
    from halal_trader.events.renames import TICKER_RENAMES, old_tickers

    async def _run(engine: Any, settings: Any) -> list[tuple[str, Any, Any]]:
        from halal_trader.events.renames import seed_candidates

        return await seed_candidates(engine)

    candidates = run_db(_run)
    console.print(f"{len(candidates)} candidate(s): first halal screen, first news, old tickers")
    for symbol, first_halal, first_news in candidates:
        olds = old_tickers(symbol)
        mapped = (
            ", ".join(f"{o} (to {TICKER_RENAMES[o][1]})" for o in olds)
            if olds
            else "[yellow]unmapped[/yellow]"
        )
        console.print(f"  {symbol:6} {first_halal}  {str(first_news or 'no news'):10}  {mapped}")
    flagged = {c[0] for c in candidates}
    others = sorted({current for current, _ in TICKER_RENAMES.values()} - flagged)
    if others:
        console.print(f"also mapped (not flagged by the rule): {', '.join(others)}")


@renames_group.command("backfill")
@click.option(
    "--rate",
    default=100,
    show_default=True,
    help="Alpaca requests per minute (the live bot shares the key's 200/min).",
)
def renames_backfill_cmd(rate: int) -> None:
    """Store each old ticker's news under its current symbol (resumable)."""
    from halal_trader.events.renames import RenamedNewsError

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.events.renames import backfill_renamed_news

        # The client paces every request, each page included.
        market = AlpacaMarketData.from_settings(settings, min_interval_s=60.0 / max(rate, 1))
        try:
            return await backfill_renamed_news(engine, market)
        finally:
            await market.aclose()

    try:
        written = run_db(_run)
    except RenamedNewsError as e:
        fail(str(e))
    console.print(f"renamed-ticker news: {written} new event(s)")


@events.group("aliases")
def aliases_group() -> None:
    """The names the story builder's entity check accepts for each symbol."""


@aliases_group.command("build")
@click.option(
    "--force",
    is_flag=True,
    help="Build even though months of renamed-ticker news are missing.",
)
def aliases_build_cmd(force: bool) -> None:
    """Learn every symbol's aliases and store them (replaces this builder version's)."""
    from halal_trader.events.renames import RenamedNewsError

    async def _run(engine: Any, settings: Any) -> tuple[int, str]:
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.events.aliases import alias_sha, build_aliases

        market = AlpacaMarketData.from_settings(settings)
        try:
            rows = await build_aliases(engine, market, force=force)
        finally:
            await market.aclose()
        return rows, await alias_sha(engine)

    try:
        rows, sha = run_db(_run)
    except RenamedNewsError as e:
        fail(f"{e} (or pass --force)")
    console.print(f"{rows} alias row(s) stored; alias_sha {sha}")


@events.group("stories")
def stories_group() -> None:
    """The news engine's stories: one symbol's news and 8-Ks of one reaction session."""


@stories_group.command("build")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), required=True)
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), required=True)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "Build even though renamed-ticker news is missing, no alias is stored, "
        "or news is not parsed by the current earnings extractor."
    ),
)
def stories_build_cmd(start: Any, end: Any, force: bool) -> None:
    """Build every story with its reaction session in [start, end] (replaces those rows)."""
    from halal_trader.events.stories import StoriesNotReady

    async def _run(engine: Any, settings: Any) -> tuple[int, dict[str, int], dict[str, str]]:
        from collections import Counter

        from halal_trader.events.stories import build_range, pins

        counters: Counter[str] = Counter()
        written = await build_range(
            engine, start=start.date(), end=end.date(), force=force, counters=counters
        )
        return written, dict(counters), await pins(engine)

    try:
        written, counters, pinned = run_db(_run)
    except StoriesNotReady as e:
        fail(f"{e} (or pass --force)")
    console.print(f"{written} story(ies) stored for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    console.print("items read: " + ", ".join(f"{k} {v}" for k, v in sorted(counters.items())))
    console.print("pins: " + ", ".join(f"{k} {v}" for k, v in pinned.items()))


@stories_group.command("counts")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default="2016-10-03")
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default="2024-12-31")
def stories_counts_cmd(start: Any, end: Any) -> None:
    """Stories by close type and NSN, per year and window: all, PRIMARY, Technology."""

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.stories import count_stories

        return await count_stories(engine, start=start.date(), end=end.date())

    from halal_trader.events.stories import counts_table

    for line in counts_table(run_db(_run), start=start.date(), end=end.date()):
        console.print(line, markup=False, highlight=False)


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


@events.command("peers")
@click.option("--sector", type=click.Choice(["technology", "all"]), default="technology")
@click.option("--start", type=int, default=2016, show_default=True, help="First year.")
@click.option("--end", type=int, default=2021, show_default=True, help="Last year.")
@click.option("--by", type=click.Choice(["all", "year"]), default="all")
def peers_cmd(sector: str, start: int, end: int, by: str) -> None:
    """Peer read-through: industry peers' net abnormal return by the leader's surprise."""

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.earnings_signal import releases
        from halal_trader.events.peers import peer_outcomes
        from halal_trader.events.study import summarise
        from halal_trader.halal.sector_limits import TECHNOLOGY

        leaders = [r for r in await releases(engine) if start <= r.published_at.year <= end]
        outcomes = await peer_outcomes(
            engine, leaders, sector=TECHNOLOGY if sector == "technology" else None
        )
        return summarise(outcomes, by=None if by == "all" else by)

    result = run_db(_run)
    console.print(
        f"peers of {sector} earnings releases {start}-{end}: peers' mean net abnormal "
        "return by the leader's sales-surprise decile (t-stat)"
    )
    for group in sorted({r.group for r in result.rows}):
        console.print(f"[bold]{group}[/bold] (releases={result.n.get(group, 0)})")
        for h in sorted({r.horizon for r in result.rows if r.group == group}):
            cells = [r for r in result.rows if r.group == group and r.horizon == h]
            line = "  ".join(f"D{r.decile} {r.mean:+.2%}({r.t:+.1f})" for r in cells)
            console.print(f"  {h:>2}d IC {result.ic.get((group, h), 0):+.3f}  | {line}")


@events.command("reversal")
@click.option("--sector", type=click.Choice(["all", "technology"]), default="all")
@click.option("--start", type=int, default=2016, show_default=True, help="First year.")
@click.option("--end", type=int, default=2021, show_default=True, help="Last year.")
def reversal_cmd(sector: str, start: int, end: int) -> None:
    """Buying after bad-news days: net abnormal return by the day's lexicon decile."""

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.reversal import HORIZONS, news_days
        from halal_trader.events.study import evaluate, summarise

        symbols = None
        if sector == "technology":
            from halal_trader.events.tech_expert import tech_symbols

            symbols = await tech_symbols(engine)
        obs = await news_days(engine, start, end, symbols)
        return summarise(await evaluate(engine, obs, HORIZONS)), len(obs)

    result, n = run_db(_run)
    console.print(
        f"news days {start}-{end} ({sector}, {n} symbol-days): net abnormal return vs SPY "
        "by the day's lexicon decile (D1 = worst news)"
    )
    for h in sorted({r.horizon for r in result.rows}):
        cells = [r for r in result.rows if r.horizon == h]
        line = "  ".join(f"D{r.decile} {r.mean:+.2%}({r.t:+.1f})" for r in cells)
        console.print(f"  {h:>2}d IC {result.ic.get(('all', h), 0):+.3f}  | {line}")


@events.command("study")
@click.argument("signal", type=click.Choice(["sue", "sales", "beat-raise"]))
@click.option("--start", type=int, default=2016, show_default=True, help="First year.")
@click.option("--end", type=int, default=2019, show_default=True, help="Last year.")
@click.option("--by", type=click.Choice(["all", "bucket", "year"]), default="bucket")
def study_cmd(signal: str, start: int, end: int, by: str) -> None:
    """Event study of a free signal: net abnormal return by signal decile and horizon.

    sue: SEC EPS surprise; sales, beat-raise: earnings releases as catalysts,
    entered at their first tradable price (events/earnings_signal.py).
    """

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.study import Observation, evaluate, summarise

        if signal == "sue":
            from halal_trader.events.history import covered_companies
            from halal_trader.events.sue import sue_observations

            raw = await sue_observations(engine, await covered_companies(engine))
            obs = [
                Observation(o.symbol, o.announced_at, o.sue)
                for o in raw
                if start <= o.announced_at.year <= end
            ]
        else:
            from halal_trader.events.earnings_signal import releases, signal_of

            obs = [
                Observation(r.symbol, r.published_at, value)
                for r in await releases(engine)
                if start <= r.published_at.year <= end
                and (value := signal_of(r, signal)) is not None
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


@events.command("tech-score")
@click.argument("variant", type=click.Choice(["expert", "context"]))
@click.option("--max-pairs", default=40_000, show_default=True, help="Stop after this many.")
def tech_score_cmd(variant: str, max_pairs: int) -> None:
    """Score post-cutoff tech headlines as a tech-industry expert (research budget).

    Only events the generic scorer already scored, so the two compare on the
    same headlines (events/tech_expert.py; read with `events tech-eval`).
    """

    async def _run(engine: Any, settings: Any) -> int:
        from halal_trader.core.llm import create_classifier_llm
        from halal_trader.events.llm_score import score_all
        from halal_trader.events.tech_expert import tech_symbols, variants

        _research_meter(engine, settings)
        return await score_all(
            create_classifier_llm(settings),
            engine,
            model=settings.llm.model,
            max_pairs=max_pairs,
            variant=variants(await tech_symbols(engine))[variant],
        )

    console.print(f"{run_db(_run)} tech headline/symbol pairs scored ({variant})")


@events.command("tech-eval")
@click.option(
    "--scorer",
    "extra",
    multiple=True,
    help="LABEL=SCORER_ID: compare the generic score with this one instead of the variants.",
)
@click.option("--since", type=click.DateTime(["%Y-%m-%d"]), default=None)
def tech_eval_cmd(extra: tuple[str, ...], since: Any) -> None:
    """Generic vs tech-expert vs expert-with-context scores on the same tech headlines."""

    async def _run(engine: Any, settings: Any) -> Any:
        from datetime import UTC

        from halal_trader.events.tech_expert import compare

        scorers = dict(e.split("=", 1) for e in extra) or None
        return await compare(
            engine,
            settings.llm.model,
            scorers=scorers,
            since=since.replace(tzinfo=UTC) if since else None,
        )

    c = run_db(_run)
    console.print(f"{c.days} (symbol, day) readings every scorer read")
    for label, result in c.results.items():
        n = result.n.get("all", 0)
        console.print(f"[bold]{label}[/bold] (n={n})")
        for h in sorted({h for (_, h) in result.ic}):
            ic = result.ic[("all", h)]
            cells = {r.decile: r for r in result.rows if r.horizon == h}
            line = f"  {h:>2}d IC {ic:+.3f} (t {ic * n**0.5:+.1f})"
            if 10 in cells and 1 in cells:
                line += f"  top decile {cells[10].mean:+.2%}  bottom decile {cells[1].mean:+.2%}"
            console.print(line)


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


@events.command("exit-test")
def exit_test_cmd() -> None:
    """Negative news as an exit: the gross move after a 60 s exit, train and holdout halves."""

    async def _run(engine: Any, settings: Any) -> dict[str, list[Any]]:
        from halal_trader.data.alpaca_market import AlpacaMarketData
        from halal_trader.events.intraday import exit_test, first_in_session, run, selection

        market = AlpacaMarketData.from_settings(settings, min_interval_s=60.0 / 80)
        try:
            chosen = selection(await first_in_session(engine))
            return exit_test(await run(engine, market, chosen, cost_sides=0))
        finally:
            await market.aclose()

    for half, buckets in run_db(_run).items():
        console.print(f"[bold]{half}[/bold] (gross abnormal move from 60 s after the headline)")
        for b in buckets:
            console.print(
                f"  {b.label:20} {b.horizon:9} n={b.n:<5} mean {b.mean:+.2%}  t {b.t:+.1f}"
            )


@events.command("atlas")
@click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default="2016-10-03")
@click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default="2021-12-23")
@click.option(
    "--workers", default=1, show_default=True, help="Simulator workers (serial on macOS)."
)
def atlas_cmd(start: Any, end: Any, workers: int) -> None:
    """The path atlas: price paths by story type on train (descriptive; after H1's verdict).

    Writes data/research/news_atlas-stories-v1.json and prints the tables.
    """
    from halal_trader.events.atlas import AtlasLocked, check_range, tables

    try:
        check_range(start.date(), end.date())
    except ValueError as e:
        fail(str(e))

    async def _run(engine: Any, settings: Any) -> tuple[Any, Any]:
        from halal_trader.events.atlas import output_path, run_atlas, write_atlas

        result = await run_atlas(engine, start=start.date(), end=end.date(), workers=workers)
        return result, write_atlas(result, output_path(settings))

    try:
        result, path = run_db(_run)
    except AtlasLocked as e:
        fail(str(e))
    for line in tables(result.cells):
        console.print(line, markup=False, highlight=False)
    counts = result.meta.get("counts", {})
    console.print(
        f"{len(result.rows)} stories, {counts.get('measured', 0)} measured; "
        f"{counts.get('persisted_mismatch', 0)} differ from their news_stories row",
        markup=False,
        highlight=False,
    )
    console.print(f"written to {path}", markup=False, highlight=False)
