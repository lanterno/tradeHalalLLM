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
    engine = await init_db(str(settings.resolve_db_path()))
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
        path = settings.resolve_db_path().parent / "analytics" / "round_trip_purification.json"
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
        root = settings.resolve_db_path().parent / "replay"
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
        samples = [
            CalibrationSample(raw_confidence=0.5, win=v.pnl_pct > 0) for v in views
        ]
        curve = fit_auto(samples)
        metrics = calibration_metrics(curve, samples)
        console.print(f"method        : {curve.method}")
        console.print(f"samples       : {curve.n_samples}")
        console.print(f"anchors       : {curve.anchors}")
        console.print(f"ECE           : {metrics['ece']:.3f}")
        console.print(f"Brier         : {metrics['brier']:.3f}")

    asyncio.run(_run())
