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
from typing import Any

import click

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """halabot — market-understanding engine (read-only)."""


@cli.command("shadow")
@click.option("--once", is_flag=True, default=False, help="One poll then exit (else run forever).")
@click.option("--interval", default=900.0, show_default=True, help="Poll/heartbeat seconds.")
@click.option("--timeframe", default="1Hour", show_default=True, help="Bar timeframe.")
@click.option("--days", default=5, show_default=True, help="Bar lookback window (days).")
def shadow(once: bool, interval: float, timeframe: str, days: int) -> None:
    """Run the read-only engine on live Alpaca data, logging shadow proposals."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_run_shadow(once=once, interval=interval, timeframe=timeframe, days=days))


async def _run_shadow(*, once: bool, interval: float, timeframe: str, days: int) -> None:
    # Lazy imports — legacy config/MCP/DB only loaded when actually running.
    from halabot.app import build_engine
    from halabot.perception.base import SourceSupervisor
    from halabot.perception.sources.alpaca_bars import AlpacaBarSource
    from halabot.platform.clock import SystemClock
    from halabot.platform.events import EventType, new_event
    from halabot.policy.sizing import PolicyConfig
    from halal_trader.config import get_settings
    from halal_trader.db.models import init_db
    from halal_trader.db.repository import Repository
    from halal_trader.mcp.client import AlpacaMCPClient

    settings = get_settings()
    clock = SystemClock()
    # Cold-start bands tuned to the observed raw-conviction scale (a single
    # momentum signal tops ~0.35 in a normal tape — see the shadow's own output).
    # Deliberately calibrated to the scale per REARCHITECTURE B.2's cold-start
    # note; replaced by the fitted calibrator (L4/L8) once enough closed outcomes
    # exist to map raw → P(win). Until then the shadow proposes on the genuinely
    # trending names so the Phase-3 A/B has signal.
    cold_start_policy = PolicyConfig(
        conviction_entry_band=0.25,
        conviction_exit_band=0.15,
        max_weight_per_asset=0.20,
        max_gross_exposure=1.0,
        target_rebalance_threshold=0.03,
    )
    engine = await build_engine(
        database_url=settings.database_url, policy_config=cold_start_policy
    )
    ht_engine = await init_db(settings.database_url)  # legacy DB, for the halal universe
    repo = Repository(ht_engine)
    mcp = AlpacaMCPClient()
    await mcp.connect()

    async def universe() -> list[str]:
        return await repo.get_halal_symbols()

    from halabot.perception.sources.finnhub_news import FinnhubNewsSource

    bar_source = AlpacaBarSource(
        mcp, universe, clock, timeframe=timeframe, days=days, interval_s=interval
    )
    sources: list[Any] = [bar_source]
    finnhub_key = getattr(getattr(settings, "finnhub", None), "api_key", "") or ""
    news_source = None
    if finnhub_key:
        news_source = FinnhubNewsSource(
            finnhub_key, universe, clock, interval_s=min(60.0, interval)
        )
        sources.append(news_source)
    supervisor = SourceSupervisor()

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
        if once:
            total = 0
            for s in sources:
                total += await s.poll_once(engine.bus.publish)
            click.echo(f"emitted {total} observations")
        else:
            supervisor.start(sources, engine.bus.publish)
            click.echo(f"shadow running (poll/heartbeat every {interval:.0f}s) — Ctrl-C to stop")
            try:
                while True:
                    await asyncio.sleep(interval)
                    await engine.bus.publish(
                        new_event(clock, EventType.SYSTEM_HEARTBEAT, source="halabot.cli")
                    )
            except (KeyboardInterrupt, asyncio.CancelledError):
                click.echo("stopping…")

        await _print_summary(engine)
    finally:
        await supervisor.stop()
        if news_source is not None:
            await news_source.aclose()
        await mcp.disconnect()
        await engine.stop()
        await ht_engine.dispose()


@cli.command("ab-report")
@click.option("--days", default=1, show_default=True, help="Look-back window (days).")
def ab_report_cmd(days: int) -> None:
    """Compare shadow proposals vs the live cycle's trades (Phase-3 gate)."""
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_run_ab_report(days=days))


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
