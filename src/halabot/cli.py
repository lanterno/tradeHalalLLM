"""halabot CLI — run the understanding engine (read-only shadow).

A SEPARATE entrypoint from the legacy ``halal-trader`` bot, so the engine runs
isolated: if it misbehaves the live bot is unaffected, and it never executes
(Phase 3 is shadow-only). Legacy imports (config, MCP, DB) are lazy so the CLI
stays importable without a full environment.

    halabot shadow --once     # one poll, print beliefs + proposals
    halabot shadow            # run continuously (Ctrl-C to stop)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:  # annotations only (PEP 563) — no runtime import, lazy CLI preserved
    from halabot.cognition.bars import Bar

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """halabot — market-understanding engine (read-only)."""


@cli.command("shadow")
@click.option("--once", is_flag=True, default=False, help="One poll then exit (else run forever).")
@click.option(
    "--interval", default=None, type=float, help="Poll/heartbeat seconds (default: from settings)."
)
@click.option("--timeframe", default="1Hour", show_default=True, help="Bar timeframe.")
@click.option("--days", default=5, show_default=True, help="Bar lookback window (days).")
@click.option(
    "--rescreen-compliance",
    is_flag=True,
    default=False,
    help="Add the Zoya re-screening source (freshness + lapse detection, INV-7). "
    "Off by default to spare Zoya quota; the startup seed keeps verdicts fresh.",
)
def shadow(
    once: bool, interval: float | None, timeframe: str, days: int, rescreen_compliance: bool
) -> None:
    """Run the read-only engine on live Alpaca data, logging shadow proposals."""
    from halabot.platform.observability import setup_logging

    setup_logging(logging.INFO)
    asyncio.run(
        _run_shadow(
            once=once,
            interval=interval,
            timeframe=timeframe,
            days=days,
            rescreen_compliance=rescreen_compliance,
        )
    )


async def _run_shadow(
    *,
    once: bool,
    interval: float | None,
    timeframe: str,
    days: int,
    rescreen_compliance: bool = False,
) -> None:
    # Lazy imports — legacy config/MCP/DB only loaded when actually running.
    from halabot.app import build_engine
    from halabot.perception.base import SourceSupervisor
    from halabot.perception.sources.alpaca_bars import AlpacaBarSource
    from halabot.platform.clock import SystemClock
    from halabot.platform.config import get_settings as get_hb_settings
    from halabot.platform.events import EventType, new_event
    from halabot.platform.supervisor import Supervisor, heartbeat_loop
    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db
    from halal_trader.db.repository import Repository
    from halal_trader.mcp.client import AlpacaMCPClient

    settings = get_settings()  # legacy: MCP, halal universe, finnhub key
    hb = get_hb_settings()  # engine config: bands, heartbeat, thresholds
    clock = SystemClock()
    interval = interval if interval is not None else hb.engine.heartbeat_interval_s
    # Engine configs (cold-start bands, decay, risk) derive from HalabotSettings —
    # the fitted calibrator (L4/L8) replaces the cold-start bands once enough
    # closed outcomes map raw → P(win). The two systems share one DATABASE_URL.
    # Coalesce belief writes in the continuous loop (per-asset ts-ordering,
    # Appendix F); inline + synchronous for --once so the summary is immediate.
    # Bootstrap warms beliefs from the event log before the live stream starts.
    engine = await build_engine(
        database_url=settings.database_url, settings=hb, coalesce=not once, bootstrap=True
    )
    ht_engine = await init_db(settings.database_url)  # legacy DB, for the halal universe
    repo = Repository(ht_engine)
    mcp = AlpacaMCPClient()
    await mcp.connect()

    async def universe() -> list[str]:
        return await repo.get_halal_symbols()

    # The benchmark (SPY) is fed for relative strength but NEVER traded: its bars
    # go to the buffer (bar_universe), but it's excluded from the compliance seed
    # and the news feed, so the halal gate blocks any benchmark buy.
    bench = hb.cognition.benchmark_symbol if hb.cognition.relstrength_enabled else None

    async def bar_universe() -> list[str]:
        syms = await universe()
        return syms + [bench] if bench and bench not in syms else syms

    from halabot.perception.dedup import PgDedupStore
    from halabot.perception.sources.finnhub_news import FinnhubNewsSource

    # Persisted dedup so a restart doesn't re-emit the last day of headlines.
    dedup = PgDedupStore(engine.db_engine)

    bar_source = AlpacaBarSource(
        mcp, bar_universe, clock, timeframe=timeframe, days=days, interval_s=interval
    )
    sources: list[Any] = [bar_source]
    finnhub_key = getattr(getattr(settings, "finnhub", None), "api_key", "") or ""
    news_source = None
    if finnhub_key:
        news_source = FinnhubNewsSource(
            finnhub_key, universe, clock, interval_s=min(60.0, interval), dedup_store=dedup
        )
        sources.append(news_source)

    # Macro release calendar → observation.macro → catalysts_pending
    # (Task B slice 1; spec L1 row macro/fred). Fresh legacy fetcher
    # instance — never shared with the running stock bot.
    fred_key = getattr(getattr(settings, "fred", None), "api_key", "") or ""
    fred_fetcher = None
    if fred_key:
        from halabot.perception.sources.macro_catalysts import MacroCatalystSource
        from halal_trader.trading.fred_catalysts import FREDReleaseCalendarSource

        fred_fetcher = FREDReleaseCalendarSource(api_key=fred_key)
        sources.append(
            MacroCatalystSource(fred_fetcher, universe, clock, dedup_store=dedup)
        )

    zoya_client = None
    if rescreen_compliance:
        from halabot.perception.sources.zoya_compliance import ZoyaComplianceSource
        from halal_trader.halal.zoya import ZoyaClient

        zoya_key = getattr(getattr(settings, "zoya", None), "api_key", "") or ""
        if zoya_key:
            zoya_client = ZoyaClient(
                zoya_key, use_sandbox=getattr(settings.zoya, "use_sandbox", True)
            )
            sources.append(ZoyaComplianceSource(zoya_client, universe, clock))
            click.echo("compliance re-screening ENABLED (Zoya)")
        else:
            click.echo("--rescreen-compliance set but ZOYA_API_KEY missing; skipping")

    supervisor = SourceSupervisor()
    heartbeat = Supervisor()

    try:
        syms = await universe()
        click.echo(f"halal universe: {len(syms)} symbols — {', '.join(sorted(syms)[:12])}")

        # Seed compliance: the universe IS the halal-screened set (get_halal_symbols
        # returns compliance='halal'), so stamp each belief halal so the policy's
        # halal gate doesn't block every buy. A live re-screening source replaces
        # this later; for the shadow, membership in the universe = halal.
        for sym in syms:
            await engine.bus.publish(
                new_event(
                    clock, EventType.COMPLIANCE_VERDICT, source="halal-universe", asset=sym,
                    payload={"status": "halal", "detail": "halal universe member",
                             "screening_id": None, "transient_error": False},
                )
            )

        click.echo(f"sources: {', '.join(s.name for s in sources)}")
        # click.echo lands in a block-buffered stdout under launchd and can sit
        # unflushed for weeks (observed 2026-07-03: log mtime June 12); the
        # logging handler is the operationally visible channel.
        logger.info("shadow sources wired: %s", ", ".join(s.name for s in sources))
        if once:
            total = 0
            for s in sources:
                total += await s.poll_once(engine.bus.publish)
            click.echo(f"emitted {total} observations")
        else:
            supervisor.start(sources, engine.bus.publish)
            # The heartbeat drives time-decay (R-08) so conviction fades on the
            # passage of time even with no new data; supervised so a transient
            # publish failure restarts it rather than silently stopping decay.
            heartbeat.spawn(
                "heartbeat", lambda: heartbeat_loop(engine.bus, clock, interval)
            )
            click.echo(f"shadow running (poll/heartbeat every {interval:.0f}s) — Ctrl-C to stop")
            try:
                stop = asyncio.Event()
                await stop.wait()
            except (KeyboardInterrupt, asyncio.CancelledError):
                click.echo("stopping…")

        await _print_summary(engine)
    finally:
        await heartbeat.shutdown()
        await supervisor.stop()
        if news_source is not None:
            await news_source.aclose()
        if fred_fetcher is not None:
            await fred_fetcher.aclose()
        if zoya_client is not None:
            await zoya_client.close()
        await mcp.disconnect()
        await engine.stop()
        await ht_engine.dispose()


@cli.command("ab-report")
@click.option("--days", default=1, show_default=True, help="Look-back window (days).")
def ab_report_cmd(days: int) -> None:
    """Compare shadow proposals vs the live cycle's trades (Phase-3 gate)."""
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_run_ab_report(days=days))


@cli.command("backtest")
@click.option("--symbols", default="", help="Comma-separated; default = the halal universe.")
@click.option("--days", default=10, show_default=True, help="History window to fetch.")
@click.option("--timeframe", default="1Hour", show_default=True)
@click.option("--continuous", is_flag=True, default=False, help="24/7 decay (else RTH).")
@click.option(
    "--sweep-bands",
    default="",
    help="Comma-separated entry bands to compare (fetch once, replay each), e.g. 0.15,0.25,0.35.",
)
@click.option(
    "--cost-bps", default=5.0, show_default=True,
    help="One-way transaction cost (slippage+commission) in basis points.",
)
@click.option(
    "--exit-ladder", is_flag=True, default=False,
    help="Enable the Appendix-H slow-out exits (trend-break + trailing stop) in the book.",
)
@click.option(
    "--ladder-ab", is_flag=True, default=False,
    help="Controlled A/B: replay the SAME fetched bars with the exit ladder off vs on.",
)
@click.option(
    "--market-gate-ab", is_flag=True, default=False,
    help="Controlled A/B: replay the SAME bars with the market-regime gate off vs on.",
)
@click.option(
    "--trailing-pct", default=0.05, show_default=True,
    help="Trailing-stop ratchet distance for --exit-ladder (fraction of the high-water mark).",
)
@click.option(
    "--cache-write", default="", help="After fetching, write the bars to this JSON file.",
)
@click.option(
    "--cache-read", default="",
    help="Replay bars from this JSON cache instead of fetching (reproducible, offline).",
)
@click.option(
    "--oos-splits", default=1, show_default=True,
    help="Partition the bars into N disjoint time windows and report each (out-of-sample).",
)
@click.option(
    "--forecaster", type=click.Choice(["", "ols", "chronos"]), default="",
    help="Append a forecaster interpreter to the stack (ols = cheap slope; chronos = [ml]).",
)
@click.option(
    "--forecaster-ab", is_flag=True, default=False,
    help="Controlled A/B: replay the SAME bars with no forecaster vs the Chronos forecaster.",
)
def backtest(
    symbols: str, days: int, timeframe: str, continuous: bool, sweep_bands: str,
    cost_bps: float, exit_ladder: bool, ladder_ab: bool, market_gate_ab: bool,
    trailing_pct: float, cache_write: str, cache_read: str, oos_splits: int,
    forecaster: str, forecaster_ab: bool,
) -> None:
    """Replay historical bars through the engine and report hypothetical P&L."""
    from halabot.platform.observability import setup_logging

    setup_logging(logging.WARNING)  # quiet — the result line is the output
    asyncio.run(
        _run_backtest(
            symbols=symbols, days=days, timeframe=timeframe, continuous=continuous,
            sweep_bands=sweep_bands, cost_bps=cost_bps,
            exit_ladder=exit_ladder, ladder_ab=ladder_ab, market_gate_ab=market_gate_ab,
            trailing_pct=trailing_pct, cache_write=cache_write, cache_read=cache_read,
            oos_splits=oos_splits, forecaster=forecaster, forecaster_ab=forecaster_ab,
        )
    )


async def _fetch_backtest_bars(
    *, symbols: str, timeframe: str, days: int, bench: str | None
) -> dict[str, list[Bar]]:
    """Fetch OHLC bars via Alpaca MCP for the halal universe (+ benchmark).
    Returns {symbol: [Bar, ...]}. Owns its MCP + DB lifecycle (connect/dispose)."""
    from halabot.cognition.bars import Bar
    from halabot.perception.sources.alpaca_bars import AlpacaBarSource
    from halabot.platform.clock import SystemClock, parse_iso
    from halabot.platform.events import Event
    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db
    from halal_trader.db.repository import Repository
    from halal_trader.mcp.client import AlpacaMCPClient

    settings = get_settings()
    clock = SystemClock()
    mcp = AlpacaMCPClient()
    await mcp.connect()
    ht_engine = await init_db(settings.database_url)
    try:
        repo = Repository(ht_engine)
        chosen = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        syms = chosen or await repo.get_halal_symbols()
        fetch_syms = syms + ([bench] if bench and bench not in syms else [])

        async def universe() -> list[str]:
            return fetch_syms

        src = AlpacaBarSource(
            mcp, universe, clock, timeframe=timeframe, days=days, interval_s=999.0
        )
        collected: list[Event] = []

        async def collect(e: Event) -> None:
            collected.append(e)

        await src.poll_once(collect)
        bars_by_symbol: dict[str, list[Bar]] = {}
        for e in collected:
            if e.asset is None:
                continue
            p = e.payload
            ts = parse_iso(p.get("bar_ts")) or e.ts
            bars_by_symbol.setdefault(e.asset, []).append(
                Bar(o=float(p["o"]), h=float(p["h"]), low=float(p["low"]),
                    c=float(p["c"]), v=float(p.get("v", 0.0)), ts=ts)
            )
        return bars_by_symbol
    finally:
        await mcp.disconnect()
        await ht_engine.dispose()


def _save_bars_cache(path: str, bars_by_symbol: dict[str, list[Bar]]) -> None:
    """Serialize fetched bars to JSON so a backtest is reproducible offline
    (no Alpaca, no re-fetch drift — the same bars replay identically)."""
    import json

    data = {
        sym: [[b.o, b.h, b.low, b.c, b.v, b.ts.isoformat()] for b in bars]
        for sym, bars in bars_by_symbol.items()
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _load_bars_cache(path: str) -> dict[str, list[Bar]]:
    import json
    from datetime import datetime

    from halabot.cognition.bars import Bar

    with open(path) as f:
        data = json.load(f)
    return {
        sym: [
            Bar(o=r[0], h=r[1], low=r[2], c=r[3], v=r[4], ts=datetime.fromisoformat(r[5]))
            for r in rows
        ]
        for sym, rows in data.items()
    }


def _oos_windows(
    bars_by_symbol: dict[str, list[Bar]], n: int
) -> Iterator[tuple[str, dict[str, list[Bar]]]]:
    """Partition the global time axis into ``n`` DISJOINT contiguous windows for
    out-of-sample validation (each window is a fresh, independent replay). Yields
    (label, sliced_bars_by_symbol)."""
    all_ts = sorted(b.ts for bars in bars_by_symbol.values() for b in bars)
    if not all_ts or n < 2:
        yield ("", bars_by_symbol)
        return
    t0, t1 = all_ts[0], all_ts[-1]
    span = (t1 - t0) / n
    for i in range(n):
        lo = t0 + span * i
        hi = t1 if i == n - 1 else t0 + span * (i + 1)
        last = i == n - 1
        sliced = {
            sym: [b for b in bars if lo <= b.ts <= hi if last or b.ts < hi]
            for sym, bars in bars_by_symbol.items()
        }
        sliced = {sym: bars for sym, bars in sliced.items() if bars}
        yield (f"{lo:%Y-%m-%d %H:%M}..{hi:%Y-%m-%d %H:%M}", sliced)


async def _run_backtest(
    *, symbols: str, days: int, timeframe: str, continuous: bool, sweep_bands: str = "",
    cost_bps: float = 5.0, exit_ladder: bool = False, ladder_ab: bool = False,
    market_gate_ab: bool = False, trailing_pct: float = 0.05,
    cache_read: str = "", cache_write: str = "", oos_splits: int = 1,
    forecaster: str = "", forecaster_ab: bool = False,
) -> None:
    from halabot.analysis.backtest import Backtester
    from halabot.belief.updater import UpdaterConfig
    from halabot.platform.config import get_settings as get_hb_settings
    from halabot.policy.sizing import PolicyConfig
    from halabot.risk.engine import RiskConfig

    hb = get_hb_settings()
    # The benchmark (SPY) is fed for relative strength + the market gate, never traded.
    bench = hb.cognition.benchmark_symbol if hb.cognition.relstrength_enabled else None
    # Default the backtest forecaster to the live config (so the headline backtest
    # mirrors live behavior); an explicit --forecaster overrides.
    eff_forecaster = forecaster or (
        hb.cognition.forecaster_model if hb.cognition.forecaster_enabled else ""
    )

    def _make(
        entry_band: float, exit_band: float, *,
        ladder: bool = exit_ladder, market_gate: bool = hb.policy.market_gate_enabled,
        fcast: str = eff_forecaster,
    ) -> Backtester:
        return Backtester(
            policy_config=PolicyConfig(
                conviction_entry_band=entry_band,
                conviction_exit_band=exit_band,
                max_weight_per_asset=hb.policy.max_weight_per_asset,
                max_gross_exposure=hb.policy.max_gross_exposure,
                target_rebalance_threshold=hb.policy.target_rebalance_threshold,
                max_open_positions=hb.engine.max_open_positions,
                relstrength_gate=hb.policy.relstrength_gate,
            ),
            updater_config=UpdaterConfig(
                long_threshold=hb.belief.long_threshold,
                evidence_decay_halflife_min=hb.belief.evidence_decay_halflife_min,
                llm_thesis_enabled=False,
            ),
            risk_config=RiskConfig(
                max_portfolio_heat_pct=hb.risk.max_portfolio_heat_pct,
                max_drawdown_pct=hb.risk.max_drawdown_pct,
                daily_loss_limit=hb.risk.daily_loss_limit,
            ),
            trading_hours=not continuous,
            win_threshold_pct=hb.conviction.win_threshold_pct,
            cost_bps=cost_bps,
            exit_ladder=ladder,
            trailing_pct=trailing_pct,
            market_gate=market_gate,
            forecaster=fcast,
        )

    async def _run_and_report(bbs: dict[str, list[Bar]], *, label: str = "") -> None:
        if label:
            nbars = sum(len(v) for v in bbs.values())
            click.echo(f"\n########## window {label} ({nbars} bars) ##########")
        bands = [float(b) for b in sweep_bands.split(",") if b.strip()]
        eb = hb.policy.conviction_entry_band
        xb = hb.policy.conviction_exit_band
        if ladder_ab:
            # Controlled A/B: same bars, ladder OFF vs ON. (Comparing across separate
            # invocations is invalid — each re-fetches live bars; use --cache-read.)
            click.echo(f"=== exit-ladder A/B (same bars, trailing={trailing_pct:.0%}) ===")
            off = await _make(eb, xb, ladder=False).run(bbs, benchmark=bench)
            on = await _make(eb, xb, ladder=True).run(bbs, benchmark=bench)
            click.echo(f"  OFF: {off.summary()}")
            click.echo(f"  ON : {on.summary()}")
        elif market_gate_ab:
            # Controlled A/B: same bars, market-regime gate OFF vs ON.
            click.echo("=== market-regime gate A/B (same bars, benchmark vs SMA) ===")
            off = await _make(eb, xb, market_gate=False).run(bbs, benchmark=bench)
            on = await _make(eb, xb, market_gate=True).run(bbs, benchmark=bench)
            click.echo(f"  OFF: {off.summary()}")
            click.echo(f"  ON : {on.summary()}")
        elif forecaster_ab:
            # Controlled A/B: same bars, NO forecaster vs the Chronos forecaster (B1).
            click.echo("=== forecaster A/B (same bars, none vs chronos) ===")
            off = await _make(eb, xb, fcast="").run(bbs, benchmark=bench)
            on = await _make(eb, xb, fcast="chronos").run(bbs, benchmark=bench)
            click.echo(f"  none   : {off.summary()}")
            click.echo(f"  chronos: {on.summary()}")
        elif bands:
            click.echo("=== entry-band sweep ===")
            for band in bands:
                exit_band = min(band - 1e-6, hb.policy.conviction_exit_band)
                res = await _make(band, max(0.0, exit_band)).run(bbs, benchmark=bench)
                click.echo(f"  entry={band:.2f}: {res.summary()}")
        else:
            res = await _make(
                hb.policy.conviction_entry_band, hb.policy.conviction_exit_band
            ).run(bbs, benchmark=bench)
            click.echo(f"=== backtest result ===\n  {res.summary()}")
            click.echo("=== by entry regime ===")
            click.echo(res.regime_summary())
            click.echo("=== by entry price-structure ===")
            click.echo(res.structure_summary())
            click.echo("=== by entry market-regime (benchmark vs SMA) ===")
            click.echo(res.market_summary())
            click.echo("=== by entry evidence source (multi-label) ===")
            click.echo(res.source_summary())
            click.echo("=== by entry raw conviction (does conviction predict wins?) ===")
            click.echo(res.conviction_summary())

    if cache_read:
        bars_by_symbol = _load_bars_cache(cache_read)
        nbars = sum(len(v) for v in bars_by_symbol.values())
        click.echo(
            f"backtest: loaded {len(bars_by_symbol)} symbols / {nbars} bars from {cache_read}"
        )
    else:
        bars_by_symbol = await _fetch_backtest_bars(
            symbols=symbols, timeframe=timeframe, days=days, bench=bench
        )
        nbars = sum(len(v) for v in bars_by_symbol.values())
        click.echo(f"backtest: {len(bars_by_symbol)} symbols, {nbars} bars ({timeframe}, {days}d)")
        if cache_write:
            _save_bars_cache(cache_write, bars_by_symbol)
            click.echo(f"backtest: wrote bar cache → {cache_write}")

    if oos_splits and oos_splits > 1:
        click.echo(f"=== out-of-sample: {oos_splits} disjoint windows ===")
        for lbl, sliced in _oos_windows(bars_by_symbol, oos_splits):
            await _run_and_report(sliced, label=lbl)
    else:
        await _run_and_report(bars_by_symbol)


@cli.command("attribution")
@click.option("--min-n", default=1, show_default=True, help="Min closed trades per bucket.")
def attribution_cmd(min_n: int) -> None:
    """Per-regime / per-source win-rate + avg-return over closed outcomes."""
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_run_attribution(min_n=min_n))


async def _run_attribution(*, min_n: int) -> None:
    from halabot.analysis.attribution import attribution
    from halabot.platform.config import get_settings as get_hb_settings
    from halabot.platform.db import bootstrap_schema, make_engine
    from halal_trader.config import get_settings

    engine = make_engine(get_settings().database_url)
    await bootstrap_schema(engine)
    try:
        attr = await attribution(
            engine, min_n=min_n,
            win_threshold_pct=get_hb_settings().conviction.win_threshold_pct,
        )
    finally:
        await engine.dispose()
    click.echo(
        f"=== outcome attribution ({attr.total} closed + {attr.open_count} open, "
        "bias-corrected) ==="
    )
    click.echo("by regime:")
    for b in attr.by_regime:
        click.echo(f"  {b.line()}")
    click.echo("by evidence source:")
    for b in attr.by_source:
        click.echo(f"  {b.line()}")


@cli.command("dashboard")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8083, show_default=True, help="HTTP port (legacy uses 8082).")
def dashboard(host: str, port: int) -> None:
    """Serve the read-first understanding API (beliefs / decisions / risk / controls)."""
    from halabot.platform.observability import setup_logging

    setup_logging(logging.INFO)
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed — install the dashboard extra: uv sync --extra dashboard"
        ) from None

    from halabot.api.app import create_api
    from halabot.platform.db import bootstrap_schema, make_engine
    from halal_trader.config import get_settings

    engine = make_engine(get_settings().database_url)
    app = create_api(engine)

    @app.on_event("startup")  # type: ignore[untyped-decorator]
    async def _bootstrap() -> None:  # ensure hb_* tables exist before serving
        await bootstrap_schema(engine)

    click.echo(f"halabot API on http://{host}:{port}  (GET /beliefs, /decisions, /risk, /health)")
    uvicorn.run(app, host=host, port=port, log_level="info")


async def _run_ab_report(*, days: int) -> None:
    from datetime import UTC, datetime, timedelta

    from halabot.analysis.ab_report import ab_report
    from halabot.platform.db import bootstrap_schema, make_engine
    from halal_trader.config import get_settings

    settings = get_settings()
    engine = make_engine(settings.database_url)
    await bootstrap_schema(engine)
    until = datetime.now(UTC)
    since = until - timedelta(days=days)
    try:
        rep = await ab_report(engine, since=since, until=until)
    finally:
        await engine.dispose()

    click.echo(f"=== shadow vs live ({days}d window) ===")
    click.echo(f"  shadow proposals: {rep.shadow_total}   live trades: {rep.live_total}")
    if rep.churn_reduction_pct is not None:
        click.echo(f"  churn reduction: {rep.churn_reduction_pct:+.0%} (shadow vs live count)")
    if rep.shadow_closed:
        avg = rep.shadow_avg_return_pct or 0.0
        win = rep.shadow_win_rate or 0.0
        click.echo(
            f"  shadow hypothetical P&L: {rep.shadow_closed} closed, "
            f"avg {avg:+.2%}, win {win:.0%}, book-weighted {rep.shadow_weighted_return:+.4f}"
        )
    if rep.symbols_only_live:
        click.echo(f"  live-only symbols (churn avoided): {sorted(rep.symbols_only_live)}")
    click.echo(f"  shadow by symbol: {dict(sorted(rep.shadow_by_symbol.items()))}")
    click.echo(f"  live by symbol:   {dict(sorted(rep.live_by_symbol.items()))}")
    if rep.live_closed:
        click.echo(
            f"  live realized P&L: {rep.live_closed} closed, avg {rep.live_avg_return_pct:+.2%}"
        )
    if rep.promotion is not None:
        g = rep.promotion
        verdict = "PROMOTE ✅" if g.promote else "HOLD ⛔"
        es = f"{g.effect_size:+.2f}" if g.effect_size is not None else "n/a"
        p = f"{g.p_two_sided:.3f}" if g.p_two_sided is not None else "n/a"
        click.echo(
            f"  Phase-3 gate: {verdict}  (shadow n={g.n_shadow}, live n={g.n_live}, "
            f"effect d={es}, p={p})"
        )
        for r in g.reasons:
            click.echo(f"    - {r}")


async def _print_summary(engine: object) -> None:
    beliefs = await engine.store.all_active()  # type: ignore[attr-defined]
    click.echo(f"\n=== beliefs ({len(beliefs)}) ===")
    for b in sorted(beliefs, key=lambda x: -x.conviction):
        click.echo(
            f"  {b.asset:6s} {b.direction.value:9s} regime={b.regime.value:13s} "
            f"conv={b.conviction:.3f} (raw {b.conviction_raw:.3f}) "
            f"halal={b.halal.status if b.halal else '?'}"
        )
    runner = engine.shadow  # type: ignore[attr-defined]
    click.echo(f"\n=== shadow proposals this run: {runner.proposals_count} ===")
    for p in runner.last_proposals:
        click.echo(f"  {p.side} {p.asset} Δw={p.weight_delta:+.3f} → {p.target_weight:.3f}")


if __name__ == "__main__":
    cli()
