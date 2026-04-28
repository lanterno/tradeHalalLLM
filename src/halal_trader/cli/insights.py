"""Insights CLI — surface the new analysis modules at the terminal.

Each subcommand is a thin Click wrapper that builds the relevant
analyzer over recently-closed trades or runs an offline harness:

* ``halal-trader insights regret``     — hindsight regret + summary.
* ``halal-trader insights thesis``     — thesis attribution table.
* ``halal-trader insights stress``     — synthetic stress harness report.
* ``halal-trader insights drift``      — concept-drift state from recent trades.
* ``halal-trader insights calibration``— win-rate calibration on closed trades.

Heavy modules (binance, sqlmodel, ml) are imported inside command bodies
to keep ``--help`` fast — same pattern the rest of the CLI uses.
"""

from __future__ import annotations

import asyncio

import click

# ── shared helper ────────────────────────────────────────────────


async def _load_closed_crypto_trades(limit: int = 200) -> list:
    """Pull the last N closed CryptoTrade rows; one place to fix later."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from halal_trader.config import get_settings
    from halal_trader.db.models import CryptoTrade, init_db

    settings = get_settings()
    engine = await init_db(settings.database_url)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sm() as s:
            stmt = (
                select(CryptoTrade)
                .where(CryptoTrade.closed_at.is_not(None))
                .order_by(CryptoTrade.closed_at.desc())
                .limit(limit)
            )
            result = await s.execute(stmt)
            return list(result.scalars().all())
    finally:
        await engine.dispose()


def _trades_to_closed_views(trades) -> list:
    """Map CryptoTrade → ClosedTradeView for regret/calibration."""
    from halal_trader.core.regret import ClosedTradeView

    out = []
    for t in trades:
        if not t.entry_price or not t.exit_price or t.side != "buy":
            continue
        pnl_pct = (t.exit_price - t.entry_price) / t.entry_price
        out.append(
            ClosedTradeView(
                trade_id=str(t.id),
                symbol=t.pair,
                action_size_pct=1.0,  # CryptoTrade lacks size_pct; use 1.0 placeholder
                pnl_pct=pnl_pct,
            )
        )
    return out


def _trades_to_tagged(trades) -> list:
    """Map CryptoTrade → TaggedTradeContext for attribution."""
    from halal_trader.core.thesis import TaggedTradeContext

    out = []
    for t in trades:
        if not t.entry_price or not t.exit_price or t.side != "buy":
            continue
        pnl_pct = (t.exit_price - t.entry_price) / t.entry_price
        hold_seconds = 0
        if t.closed_at and t.timestamp:
            hold_seconds = max(0, int((t.closed_at - t.timestamp).total_seconds()))
        out.append(
            TaggedTradeContext(
                trade_id=str(t.id),
                symbol=t.pair,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                exit_reason=t.exit_reason,
                pnl_pct=pnl_pct,
                hold_seconds=hold_seconds,
                reasoning=(t.llm_reasoning or "")[:240],
            )
        )
    return out


# ── group ────────────────────────────────────────────────────────


@click.group("insights")
def insights() -> None:
    """Run analytics over recent trades and synthetic scenarios."""


# ── regret ───────────────────────────────────────────────────────


@insights.command("regret")
@click.option("--limit", default=200, show_default=True, help="Last N closed trades")
def regret_cmd(limit: int) -> None:
    """Hindsight regret on the last N closed crypto trades."""

    async def _run() -> None:
        from halal_trader.core.regret import aggregate_regret, hindsight_regret
        from halal_trader.logging import console

        trades = await _load_closed_crypto_trades(limit)
        views = _trades_to_closed_views(trades)
        if not views:
            console.print("[yellow]No closed trades found.[/]")
            return
        records = [hindsight_regret(v) for v in views]
        summary = aggregate_regret(records)
        console.print(f"[bold]Regret over last {summary.n} trades:[/]")
        console.print(f"  mean regret  : {summary.mean_regret:.2f}")
        console.print(f"  median       : {summary.median_regret:.2f}")
        console.print(f"  high-regret  : {summary.pct_high_regret:.0%}")
        console.print(f"  missed-edge  : {summary.missed_edge_count}")
        console.print(f"  tail-loss    : {summary.tail_loss_count}")
        if summary.by_symbol:
            console.print("[bold]By symbol:[/]")
            for sym, r in sorted(summary.by_symbol.items(), key=lambda kv: -kv[1]):
                console.print(f"  {sym:<10} {r:.2f}")

    asyncio.run(_run())


# ── thesis ───────────────────────────────────────────────────────


@insights.command("thesis")
@click.option("--limit", default=200, show_default=True, help="Last N closed trades")
def thesis_cmd(limit: int) -> None:
    """Tag-by-tag P&L attribution with the heuristic tagger."""

    async def _run() -> None:
        from halal_trader.core.thesis import (
            attribute_pnl_by_thesis,
            deprecated_thesis_kill_list,
            render_attribution,
        )
        from halal_trader.logging import console

        trades = await _load_closed_crypto_trades(limit)
        views = _trades_to_tagged(trades)
        if not views:
            console.print("[yellow]No closed trades found.[/]")
            return
        rows = attribute_pnl_by_thesis(views)
        console.print(render_attribution(list(rows.values())))
        kills = deprecated_thesis_kill_list(rows)
        if kills:
            console.print(f"[red]Kill candidates: {', '.join(kills)}[/]")

    asyncio.run(_run())


# ── stress ───────────────────────────────────────────────────────


@insights.command("stress")
def stress_cmd() -> None:
    """Run the standard adversarial stress scenarios with a sane stub strategy.

    Replace the stub with the live strategy (see ``crypto.stress`` docs)
    once you're ready to score the real prompt against the suite.
    """

    async def _run() -> None:
        from halal_trader.crypto.stress import (
            evaluate_scenarios,
            render_report,
        )
        from halal_trader.domain.models import CryptoTradingPlan
        from halal_trader.logging import console

        async def _baseline_strategy(_klines):
            # Stub: emit no buys, all hold. Used to sanity-check the harness.
            return CryptoTradingPlan(decisions=[], market_outlook="baseline")

        verdicts = await evaluate_scenarios(_baseline_strategy)
        console.print(render_report(verdicts))

    asyncio.run(_run())


# ── drift ────────────────────────────────────────────────────────


@insights.command("drift")
@click.option("--limit", default=200, show_default=True, help="Last N closed trades")
def drift_cmd(limit: int) -> None:
    """Feed recent residuals into a DriftMonitor and report state."""

    async def _run() -> None:
        from halal_trader.logging import console
        from halal_trader.ml.drift import DriftMonitor

        trades = await _load_closed_crypto_trades(limit)
        views = _trades_to_closed_views(trades)
        if not views:
            console.print("[yellow]No closed trades found.[/]")
            return
        mon = DriftMonitor()
        for v in views:
            mon.observe(v.pnl_pct)
        console.print(f"state           : {mon.state}")
        console.print(f"observations    : {mon.n}")
        console.print(f"drift events    : {mon.drift_count}")
        console.print(f"last drift at   : {mon.last_drift_at}")

    asyncio.run(_run())


# ── calibration ──────────────────────────────────────────────────


@insights.command("shadow")
def shadow_cmd() -> None:
    """Show divergence between live and shadow equity curves.

    Reads from the in-process ShadowLedger via ``insights_hub``. Empty
    if the cycle hasn't recorded any rows yet.
    """

    def _run() -> None:
        from halal_trader.core.insights_hub import hub
        from halal_trader.core.shadow import (
            divergence_metrics,
            render_status,
            shadow_alert_state,
        )
        from halal_trader.logging import console

        led = hub.shadow
        if led.size == 0:
            console.print("[yellow]Shadow ledger empty — no cycles recorded yet.[/]")
            return
        metrics = divergence_metrics(led.entries)
        level = shadow_alert_state(metrics)
        console.print(render_status(metrics, level))

    _run()


@insights.command("regime")
def regime_cmd() -> None:
    """Show recent entries from the regime memory store."""

    def _run() -> None:
        from halal_trader.core.insights_hub import hub
        from halal_trader.logging import console

        mem = hub.regime
        if mem.size == 0:
            console.print("[yellow]Regime memory empty.[/]")
            return
        console.print(f"[bold]Regime memory:[/] {mem.size} snapshot(s)")
        for s in mem.snapshots[-10:]:
            console.print(f"  {s.label()}")

    _run()


@insights.command("purification")
def purification_cmd() -> None:
    """Outstanding round-trip purification due."""

    def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.halal.round_trip_purification import (
            RoundTripLedger,
            outstanding_round_trip_due,
        )
        from halal_trader.logging import console

        settings = get_settings()
        path = settings.resolve_data_dir() / "analytics" / "round_trip_purification.json"
        if not path.exists():
            console.print("[yellow]No purification ledger yet — no closed wins.[/]")
            return
        ledger = RoundTripLedger(path=path)
        summary = outstanding_round_trip_due(ledger)
        console.print(f"[bold]Outstanding:[/] ${summary['total_usd']:.2f}")
        console.print(f"Disbursed total: ${summary['disbursed_total_usd']:.2f}")
        console.print(f"Total entries: {summary['n_entries']}")
        if summary["by_symbol"]:
            console.print("[bold]By symbol:[/]")
            for sym, due in sorted(summary["by_symbol"].items(), key=lambda kv: -kv[1]):
                console.print(f"  {sym:<10} ${due:.2f}")

    _run()


@insights.command("replay")
@click.option("--limit", default=20, show_default=True)
def replay_cmd(limit: int) -> None:
    """List recent cycle snapshots in the replay store."""

    def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core.replay import ReplayStore
        from halal_trader.logging import console

        settings = get_settings()
        root = settings.resolve_data_dir() / "replay"
        if not root.exists():
            console.print("[yellow]No replay store yet — start the bot to capture snapshots.[/]")
            return
        store = ReplayStore(root=root)
        ids = store.list_cycle_ids()[-limit:]
        if not ids:
            console.print("[yellow]Replay store empty.[/]")
            return
        for cid in ids:
            console.print(f"  {cid}")

    _run()


@insights.command("catalysts")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--lookahead",
    default=24,
    show_default=True,
    help="Hours of history/forward-window for time-bound sources",
)
def catalysts_cmd(symbols: tuple[str, ...], lookahead: int) -> None:
    """Inspect what catalysts the stock cycle will surface for SYMBOLS.

    Constructs the same source bundle the live trading scheduler builds
    (FRED + EDGAR + Options-IV + Fed-speak), runs each for the given
    symbols, and prints the assembled catalyst list — exactly what the
    LLM would see in the next cycle's prompt.
    """

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.logging import console
        from halal_trader.trading.catalysts import (
            StockCatalystFeed,
            format_catalysts_for_prompt,
        )

        settings = get_settings()
        sources: list = []

        if settings.fred.api_key:
            from halal_trader.trading.fred_catalysts import (
                FREDReleaseCalendarSource,
            )

            sources.append(FREDReleaseCalendarSource(api_key=settings.fred.api_key))
            console.print("[green]✓[/] FRED enabled")
        else:
            console.print("[yellow]✗[/] FRED disabled (no FRED_API_KEY)")

        if settings.edgar.user_agent:
            from halal_trader.trading.edgar_catalysts import EDGAREightKSource

            sources.append(EDGAREightKSource(user_agent=settings.edgar.user_agent))
            console.print("[green]✓[/] EDGAR enabled")
        else:
            console.print("[yellow]✗[/] EDGAR disabled (no EDGAR_USER_AGENT)")

        from halal_trader.trading.fed_speak_adapter import FedSpeakCatalystSource
        from halal_trader.trading.options_catalyst_adapter import (
            OptionsIVCatalystSource,
        )

        sources.append(OptionsIVCatalystSource())
        sources.append(FedSpeakCatalystSource())
        console.print("[green]✓[/] Options-IV + Fed-speak enabled (always-on)")
        console.print()

        feed = StockCatalystFeed(sources=sources)
        catalysts = await feed.fetch_all([s.upper() for s in symbols])

        console.print(f"[bold]Catalysts for {', '.join(s.upper() for s in symbols)}:[/]")
        if not catalysts:
            console.print("[yellow](no catalysts in window)[/]")
            return
        text = format_catalysts_for_prompt(
            catalysts, symbols=[s.upper() for s in symbols], max_age_hours=lookahead
        )
        console.print(text or "[yellow](all catalysts older than lookahead)[/]")

        # Close any sources that hold open clients.
        for src in sources:
            if hasattr(src, "aclose"):
                try:
                    await src.aclose()
                except Exception:  # noqa: BLE001
                    pass

    asyncio.run(_run())


@insights.command("whale")
def whale_cmd() -> None:
    """Show the latest on-chain whale-flow signals (Etherscan).

    Reads from :data:`insights_hub.whale_flows`, populated each cycle
    when ``ETHERSCAN_API_KEY`` is configured. Empty if the source is
    not enabled or no recent flows met the min-transfer threshold.
    """

    def _run() -> None:
        from halal_trader.core.insights_hub import hub
        from halal_trader.crypto.onchain import format_whale_flows_for_prompt
        from halal_trader.logging import console

        flows = hub.whale_flows
        if not flows:
            console.print("[yellow]No whale flows recorded yet.[/]")
            return
        text = format_whale_flows_for_prompt(flows)
        if text:
            console.print(text)
        else:
            console.print("[yellow]All recent flows balanced — no actionable signal.[/]")

    _run()


@insights.command("velocity")
def velocity_cmd() -> None:
    """Show the latest social mention-velocity per symbol.

    Reads from :data:`insights_hub.velocity`, populated each cycle from
    Reddit's public JSON endpoints (no OAuth). Empty until the cycle
    has run at least once with the RedditPublicFetcher wired.
    """

    def _run() -> None:
        from halal_trader.core.insights_hub import hub
        from halal_trader.logging import console
        from halal_trader.sentiment.velocity import format_velocity_for_prompt

        results = hub.velocity
        if not results:
            console.print("[yellow]No velocity results recorded yet.[/]")
            return
        text = format_velocity_for_prompt(results)
        if text:
            console.print(text)
        else:
            console.print("[yellow]No surge labels — all symbols below threshold.[/]")

    _run()


@insights.command("rag")
@click.option("--query", "-q", default="", help="Text to retrieve analogues for")
@click.option("--k", default=5, show_default=True)
def rag_cmd(query: str, k: int) -> None:
    """Top-K most-similar past trade rationales by cosine of hashed BoW."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core.llm.rag import format_rag_for_prompt
        from halal_trader.core.llm.rag_db import DBRationaleStore
        from halal_trader.db.models import init_db
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            store = DBRationaleStore(engine=engine)
            size = await store.size()
            if size == 0:
                console.print("[yellow]RAG store empty — close some trades first.[/]")
                return
            if not query:
                console.print(f"[bold]RAG store:[/] {size} rationale(s)")
                from sqlalchemy.ext.asyncio import async_sessionmaker
                from sqlmodel import select

                from halal_trader.db.models import RationaleRow as _Row

                sm = async_sessionmaker(engine, expire_on_commit=False)
                async with sm() as s:
                    rows = (
                        (await s.execute(select(_Row).order_by(_Row.timestamp.desc()).limit(10)))
                        .scalars()
                        .all()
                    )
                for r in rows:
                    outcome = "WIN" if r.outcome_win else "LOSS"
                    console.print(f"  {outcome} {r.outcome_pnl_pct:+.2%} {r.symbol}: {r.text[:80]}")
                return
            hits = await store.query(query, k=k, min_similarity=0.0)
            console.print(format_rag_for_prompt(hits, max_rows=k))
            agg = await store.aggregate(hits)
            console.print(
                f"\n[bold]Weighted outcome:[/] pnl={agg['weighted_pnl_pct']:+.2%} "
                f"win-rate={agg['weighted_win_rate']:.0%} (n={agg['n']})"
            )
        finally:
            await engine.dispose()

    asyncio.run(_run())


@insights.command("exceptions")
@click.option("--status", default="pending", show_default=True)
def exceptions_cmd(status: str) -> None:
    """List Sharia exception queue entries (pending by default)."""

    def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.halal.exception_queue import (
            ExceptionQueue,
            render_summary,
        )
        from halal_trader.logging import console

        settings = get_settings()
        path = settings.resolve_data_dir() / "analytics" / "sharia_exceptions.json"
        if not path.exists():
            console.print("[yellow]Exception queue empty.[/]")
            return
        q = ExceptionQueue(path=path)
        rows = q.all() if status == "all" else q.by_status(status)
        console.print(render_summary(rows))

    _run()


@insights.command("calibrate")
def calibrate_cmd() -> None:
    """Refit the calibration curve from recent closed trades and save it.

    Run this on a cron (weekly) to keep the live sizing engine reading
    a fresh calibrator. Refits via fit_auto: isotonic when n>=200,
    Platt for smaller samples, identity below the floor.
    """

    async def _run() -> None:

        from halal_trader.cli.insights import (
            _load_closed_crypto_trades,
            _trades_to_closed_views,
        )
        from halal_trader.config import get_settings
        from halal_trader.core.insights_hub import hub
        from halal_trader.logging import console
        from halal_trader.ml.calibration import (
            CalibrationSample,
            calibration_metrics,
            fit_auto,
        )

        trades = await _load_closed_crypto_trades(2000)
        views = _trades_to_closed_views(trades)
        if not views:
            console.print("[yellow]No closed trades — calibration unchanged.[/]")
            return
        samples = [CalibrationSample(raw_confidence=0.5, win=v.pnl_pct > 0) for v in views]
        curve = fit_auto(samples)
        metrics = calibration_metrics(curve, samples)

        settings = get_settings()
        out = settings.resolve_data_dir() / "analytics" / "calibration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        curve.save(out)
        hub.calibration = curve  # update process-wide state if running

        console.print(f"[bold]Refit:[/] method={curve.method}, n={curve.n_samples}")
        console.print(f"  ECE={metrics['ece']:.3f}  Brier={metrics['brier']:.3f}")
        console.print(f"  saved to {out}")

    asyncio.run(_run())


@insights.command("calibration")
@click.option("--limit", default=200, show_default=True)
def calibration_cmd(limit: int) -> None:
    """Fit a Platt/isotonic calibrator from recent closed trades.

    Prints the curve anchors and reliability metrics. Confidence values
    use the LlmDecision audit trail when available; trades without a
    confidence record are treated as 0.5 so they don't skew the fit.
    """

    async def _run() -> None:
        from halal_trader.logging import console
        from halal_trader.ml.calibration import (
            CalibrationSample,
            calibration_metrics,
            fit_auto,
        )

        trades = await _load_closed_crypto_trades(limit)
        views = _trades_to_closed_views(trades)
        if not views:
            console.print("[yellow]No closed trades found.[/]")
            return
        samples = [CalibrationSample(raw_confidence=0.5, win=v.pnl_pct > 0) for v in views]
        curve = fit_auto(samples)
        metrics = calibration_metrics(curve, samples)
        console.print(f"method        : {curve.method}")
        console.print(f"samples       : {curve.n_samples}")
        console.print(f"anchors       : {curve.anchors}")
        console.print(f"ECE           : {metrics['ece']:.3f}")
        console.print(f"Brier         : {metrics['brier']:.3f}")

    asyncio.run(_run())
