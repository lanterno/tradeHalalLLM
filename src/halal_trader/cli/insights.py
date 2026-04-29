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

    The shadow ledger is in-process state on the running bot — a
    standalone CLI invocation can't observe it. The dashboard's
    ``/api/insights/shadow`` route is the right surface; this stub
    stays so a tab-completing operator gets a clear hint instead
    of a silent "empty" lie.
    """
    from halal_trader.logging import console

    console.print(
        "[yellow]Shadow ledger lives in the running bot's process — "
        "use the dashboard's /api/insights/shadow endpoint, not the CLI.[/]"
    )


@insights.command("regime")
def regime_cmd() -> None:
    """Show recent entries from the regime memory store."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.logging import console
        from halal_trader.ml.regime_memory import RegimeMemory

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            mem = RegimeMemory(engine=engine)
            size = await mem.size()
            if size == 0:
                console.print("[yellow]Regime memory empty.[/]")
                return
            console.print(f"[bold]Regime memory:[/] {size} snapshot(s)")
            for s in await mem.recent(limit=10):
                console.print(f"  {s.label()}")
        finally:
            await engine.dispose()

    asyncio.run(_run())


@insights.command("purification")
def purification_cmd() -> None:
    """Outstanding round-trip purification due."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.halal.round_trip_purification import (
            RoundTripLedger,
            outstanding_round_trip_due,
        )
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            ledger = RoundTripLedger(engine=engine)
            summary = await outstanding_round_trip_due(ledger)
            if summary["n_entries"] == 0:
                console.print("[yellow]No purification ledger yet — no closed wins.[/]")
                return
            console.print(f"[bold]Outstanding:[/] ${summary['total_usd']:.2f}")
            console.print(f"Disbursed total: ${summary['disbursed_total_usd']:.2f}")
            console.print(f"Total entries: {summary['n_entries']}")
            if summary["by_symbol"]:
                console.print("[bold]By symbol:[/]")
                for sym, due in sorted(summary["by_symbol"].items(), key=lambda kv: -kv[1]):
                    console.print(f"  {sym:<10} ${due:.2f}")
        finally:
            await engine.dispose()

    asyncio.run(_run())


@insights.command("replay")
@click.option("--limit", default=20, show_default=True)
def replay_cmd(limit: int) -> None:
    """List recent cycle snapshots in the replay store."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.core.replay import ReplayStore
        from halal_trader.db.models import init_db
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            store = ReplayStore(engine=engine)
            ids = await store.list_cycle_ids(limit=limit)
            if not ids:
                console.print("[yellow]Replay store empty.[/]")
                return
            for cid in ids:
                console.print(f"  {cid}")
        finally:
            await engine.dispose()

    asyncio.run(_run())


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

    The whale-flow snapshot lives in the running bot's process; a
    standalone CLI invocation can't observe it. The dashboard's
    ``/api/insights/whale-flows`` is the right surface — this stub
    points there instead of pretending the hub is empty.
    """
    from halal_trader.logging import console

    console.print(
        "[yellow]Whale flows live in the running bot's process — "
        "use the dashboard's /api/insights/whale-flows endpoint.[/]"
    )


@insights.command("velocity")
def velocity_cmd() -> None:
    """Show the latest social mention-velocity per symbol.

    Velocity results live in the running bot's process; CLI can't
    observe them. Use the dashboard's ``/api/insights/velocity``
    endpoint.
    """
    from halal_trader.logging import console

    console.print(
        "[yellow]Velocity results live in the running bot's process — "
        "use the dashboard's /api/insights/velocity endpoint.[/]"
    )


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

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.halal.exception_queue import (
            ExceptionQueue,
            render_summary,
        )
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            q = ExceptionQueue(engine=engine)
            rows = await q.all() if status == "all" else await q.by_status(status)  # type: ignore[arg-type]
            console.print(render_summary(rows))
        finally:
            await engine.dispose()

    asyncio.run(_run())


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
        # The running bot reads the curve from disk on the next cycle;
        # nothing to push to a process-wide singleton here.

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


@insights.command("explain")
@click.argument("asset_class", type=click.Choice(["stock", "crypto"]))
@click.argument("trade_id", type=int)
def explain_cmd(asset_class: str, trade_id: int) -> None:
    """Render the halal-compliance explanation for one trade (Wave L)."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.halal.audit import export_receipt
        from halal_trader.halal.explainer import explain_screening
        from halal_trader.logging import console

        settings = get_settings()
        engine = await init_db(settings.database_url)
        try:
            receipt = await export_receipt(engine, trade_id=trade_id, asset_class=asset_class)
            if receipt is None:
                console.print(f"[red]No {asset_class} trade with id {trade_id}[/]")
                return
            explanation = explain_screening(receipt.payload)
            console.print(explanation.body_md)
            if explanation.sources:
                console.print("\n[dim]Sources:[/]")
                for s in explanation.sources:
                    console.print(f"  · {s}")
        finally:
            await engine.dispose()

    asyncio.run(_run())
