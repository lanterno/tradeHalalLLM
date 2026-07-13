"""``halal-trader quant`` — quantitative range-model tools (advisory).

``quant calibrate`` runs the pooled walk-forward z-calibration over the
curated halal universe and writes the versioned artifact that flips the
daily recommendation's bands from UNCALIBRATED to measured-coverage.
``quant outlook SYMBOL`` prints one symbol's PriceOutlook. Heavy modules
are imported inside the command bodies so ``--help`` stays fast.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click

from halal_trader.logging import console

_CACHE_DIR = Path("data/bar_cache")


def _cache_path(symbol: str, days: int) -> Path:
    return _CACHE_DIR / f"{symbol}_1Day_{days}.json"


async def _fetch_universe_bars(
    symbols: list[str], days: int, *, cache_read: bool
) -> dict[str, Any]:
    """Fetch (or reload cached) raw daily-bar payloads per symbol.

    Every live fetch is also written to the cache — same-bars reruns are
    the only valid way to compare calibration variants (house rule).
    """
    payloads: dict[str, Any] = {}
    missing: list[str] = []
    if cache_read:
        for sym in symbols:
            path = _cache_path(sym, days)
            if path.exists():
                payloads[sym] = json.loads(path.read_text())
            else:
                missing.append(sym)
        if not missing:
            return payloads
        console.print(f"[dim]cache miss for {len(missing)} symbols — fetching[/dim]")
    else:
        missing = list(symbols)

    from halal_trader.mcp.client import AlpacaMCPClient

    mcp = AlpacaMCPClient()
    await mcp.connect()
    try:
        for sym in missing:
            try:
                bars = await mcp.get_stock_bars(sym, days=days, timeframe="1Day")
            except Exception as exc:  # noqa: BLE001 — skip a flaky symbol
                console.print(f"[yellow]{sym}: bars fetch failed ({exc})[/yellow]")
                continue
            payloads[sym] = bars
            path = _cache_path(sym, days)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bars))
    finally:
        await mcp.disconnect()
    return payloads


def _payloads_to_ohlc(
    payloads: dict[str, Any],
) -> dict[str, tuple[list[float], list[float], list[float], list[float]]]:
    from halal_trader.trading.bars import bars_to_klines

    out: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {}
    for sym, payload in payloads.items():
        klines = bars_to_klines(payload)
        if len(klines) < 30:
            continue
        out[sym] = (
            [k.open for k in klines],
            [k.high for k in klines],
            [k.low for k in klines],
            [k.close for k in klines],
        )
    return out


async def _record_trial(**kwargs: Any) -> None:
    """Best-effort write to the quant_trials ledger (never blocks the tool)."""
    try:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

        engine = await init_db(get_settings().database_url)
        await QuantTrialRepoImpl(engine).record_trial(**kwargs)
        await engine.dispose()
    except Exception as exc:  # noqa: BLE001 — the ledger must not block research
        console.print(f"[yellow]trials ledger write failed: {exc}[/yellow]")


@click.group()
def quant() -> None:
    """Quantitative range-model tools (advisory — never trades)."""


@quant.command()
@click.option("--prefix", default=None, help="Filter by name prefix (e.g. levels.)")
@click.option("--limit", default=20, show_default=True)
def trials(prefix: str | None, limit: int) -> None:
    """List the quant trials ledger (the honest variant count for DSR)."""

    async def _run() -> None:
        from halal_trader.config import get_settings
        from halal_trader.db.models import init_db
        from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

        engine = await init_db(get_settings().database_url)
        repo = QuantTrialRepoImpl(engine)
        rows = await repo.get_trials(name_prefix=prefix, limit=limit)
        total = await repo.count_trials(name_prefix=prefix)
        await engine.dispose()
        scope = f" matching {prefix!r}" if prefix else ""
        console.print(f"[bold]{total}[/bold] recorded trials{scope} (showing {len(rows)})")
        for r in rows:
            verdict = r.get("verdict") or "—"
            color = {"pass": "green", "fail": "red"}.get(verdict, "yellow")
            created = str(r.get("created_at") or "")[:16]
            console.print(
                f"  [{color}]{verdict:12}[/{color}] {r['name']}  "
                f"[dim]{r['kind']} · {r['window']} · {created} · cfg {r['config_hash']}[/dim]"
            )

    asyncio.run(_run())


@quant.command()
@click.option("--days", default=400, show_default=True, help="Calendar days of daily bars")
@click.option("--cache-read", is_flag=True, help="Reuse cached bars instead of fetching")
def overnight(days: int, cache_read: bool) -> None:
    """Overnight vs intraday drift decomposition (diagnostic, not a signal).

    Quantifies how much of the universe's drift accrues close→open — the
    structural headwind any intraday-only stock strategy fights.
    """

    async def _run() -> None:
        from halal_trader.halal.cache import DEFAULT_HALAL_SYMBOLS
        from halal_trader.quant.diagnostics import (
            overnight_intraday_split,
            suspect_split_gaps,
        )

        payloads = await _fetch_universe_bars(
            list(DEFAULT_HALAL_SYMBOLS), days, cache_read=cache_read
        )
        ohlc = _payloads_to_ohlc(payloads)
        if not ohlc:
            console.print("[red]No usable bar data — aborting.[/red]")
            return
        total_on = 0.0
        total_in = 0.0
        n = 0
        for sym, (o, _h, _lo, c) in sorted(ohlc.items()):
            split = overnight_intraday_split(o, c)
            gaps = suspect_split_gaps(o, c)
            gap_note = f" [yellow]⚠ {len(gaps)} suspect split gap(s)[/yellow]" if gaps else ""
            console.print(
                f"  {sym:6} overnight {split.cum_overnight_pct:+7.1f}% · "
                f"intraday {split.cum_intraday_pct:+7.1f}% "
                f"({split.n_days}d){gap_note}"
            )
            total_on += split.mean_overnight_pct
            total_in += split.mean_intraday_pct
            n += 1
        console.print(
            f"[bold]Universe mean per day:[/bold] overnight "
            f"{total_on / n:+.3f}% · intraday {total_in / n:+.3f}%"
        )
        console.print(
            "[dim]Diagnostic only. If the overnight leg dominates, an "
            "intraday-only strategy forfeits most of the drift — an "
            "expectations anchor, not a signal (capturing overnight drift "
            "at retail costs is a known non-starter).[/dim]"
        )

    asyncio.run(_run())


@quant.command("validate-levels")
@click.option("--days", default=400, show_default=True, help="Calendar days of daily bars")
@click.option("--horizon", default=5, show_default=True, help="Forward test window (bars)")
@click.option("--cache-read", is_flag=True, help="Reuse cached bars instead of fetching")
@click.option("--placebo-seed", default=7, show_default=True, help="Placebo RNG seed")
def validate_levels(days: int, horizon: int, cache_read: bool, placebo_seed: int) -> None:
    """Touch-and-hold validation of each level family vs a placebo.

    A family earns a prompt slot only if its walk-forward hold rate beats
    the distance-matched placebo (daily-bar approximation — intraday bars
    are the honest mode; treat these numbers as a first screen).
    """

    async def _run() -> None:
        import numpy as np

        from halal_trader.core.sample_guard import SampleGate
        from halal_trader.halal.cache import DEFAULT_HALAL_SYMBOLS
        from halal_trader.quant.level_eval import (
            evaluate_family,
            merge_stats,
            placebo_uplift,
        )
        from halal_trader.quant.levels import (
            prior_extreme_levels,
            round_number_levels,
            swing_zones,
        )
        from halal_trader.recommendation.scorecard import _ohlc_by_date

        families = {
            "prior_extremes": lambda d, h, lo, c, atr: [
                lvl.price for lvl in prior_extreme_levels(d, h, lo)
            ],
            "swing_zones": lambda d, h, lo, c, atr: [z.price for z in swing_zones(h, lo, atr)],
            "round_numbers": lambda d, h, lo, c, atr: [
                r.price for r in round_number_levels(float(c[-1]))
            ],
        }
        symbols = list(DEFAULT_HALAL_SYMBOLS)
        payloads = await _fetch_universe_bars(symbols, days, cache_read=cache_read)
        series: dict[str, tuple[list[str], Any, Any, Any]] = {}
        for sym, payload in payloads.items():
            rows = _ohlc_by_date(payload)
            if len(rows) < 60:
                continue
            # rows are (date, open, high, low, close); the harness wants
            # (dates, highs, lows, closes).
            series[sym] = (
                [r[0] for r in rows],
                np.asarray([r[2] for r in rows]),
                np.asarray([r[3] for r in rows]),
                np.asarray([r[4] for r in rows]),
            )
        if not series:
            console.print("[red]No usable bar data — aborting.[/red]")
            return
        console.print(
            f"[dim]{len(series)} symbols · horizon {horizon} bars · daily-bar "
            f"approximation (touch=±0.25·ATR, reject=1·ATR, reach ≤3·ATR)[/dim]"
        )
        for name, fn in families.items():
            real_parts = []
            placebo_parts = []
            for dates, h, lo, c in series.values():
                real_parts.append(evaluate_family(dates, h, lo, c, fn, label=name, horizon=horizon))
                placebo_parts.append(
                    evaluate_family(
                        dates,
                        h,
                        lo,
                        c,
                        fn,
                        label=f"{name}:placebo",
                        horizon=horizon,
                        placebo_seed=placebo_seed,
                    )
                )
            real = merge_stats(name, real_parts)
            placebo = merge_stats(f"{name}:placebo", placebo_parts)
            uplift = placebo_uplift(real, placebo)
            gate = SampleGate(real.decided)
            verdict = "—"
            if uplift is not None:
                if not gate.sufficient:
                    verdict = "[yellow]thin sample[/yellow]"
                elif uplift > 0:
                    verdict = f"[green]+{uplift:.1%} vs placebo[/green]"
                else:
                    verdict = f"[red]{uplift:.1%} vs placebo[/red]"

            def _hr(s: Any) -> str:
                return f"{s.hold_rate:.0%}" if s.hold_rate is not None else "—"

            console.print(
                f"[bold]{name:15}[/bold] hold {_hr(real)} "
                f"(touch {real.touches}, decided {real.decided}) · "
                f"placebo {_hr(placebo)} (decided {placebo.decided}) → {verdict}"
            )
            # Ledger verdict: "pass" is reserved for disjoint-OOS runs —
            # a single-window screen can at best be inconclusive.
            ledger_verdict = "inconclusive"
            if uplift is not None and uplift <= 0:
                ledger_verdict = "fail"
            await _record_trial(
                name=f"levels.{name}.touch_hold",
                kind="level_family",
                config={
                    "horizon": horizon,
                    "days": days,
                    "placebo_seed": placebo_seed,
                    "eps_atr": 0.25,
                    "hold_atr": 1.0,
                },
                window=f"{days}d x {len(series)}sym daily (single window)",
                metrics={
                    "hold_rate": real.hold_rate,
                    "placebo_hold_rate": placebo.hold_rate,
                    "uplift": uplift,
                    "touches": real.touches,
                    "decided": real.decided,
                },
                criterion=(
                    "hold rate beats distance-matched placebo on disjoint "
                    "OOS windows with sufficient sample"
                ),
                verdict=ledger_verdict,
            )
        console.print(
            "[dim]Verdicts recorded in the trials ledger (`quant trials`). "
            "Prompt inclusion additionally requires DISJOINT OOS windows.[/dim]"
        )

    asyncio.run(_run())


@quant.command()
@click.option("--days", default=400, show_default=True, help="Calendar days of daily bars")
@click.option("--coverage", default=0.8, show_default=True, help="Target two-sided path coverage")
@click.option("--cache-read", is_flag=True, help="Reuse cached bars instead of fetching")
def calibrate(days: int, coverage: float, cache_read: bool) -> None:
    """Run the pooled walk-forward z-calibration and write the artifact."""

    async def _run() -> None:
        from halal_trader.halal.cache import DEFAULT_HALAL_SYMBOLS
        from halal_trader.quant.calibration import (
            DEFAULT_ARTIFACT_PATH,
            run_pooled_calibration,
            save_artifact,
        )

        symbols = list(DEFAULT_HALAL_SYMBOLS)
        console.print(
            f"[dim]Calibrating on {len(symbols)} symbols, {days} calendar days, "
            f"target coverage {coverage:.0%}…[/dim]"
        )
        payloads = await _fetch_universe_bars(symbols, days, cache_read=cache_read)
        ohlc = _payloads_to_ohlc(payloads)
        if not ohlc:
            console.print("[red]No usable bar data — aborting.[/red]")
            return
        artifact, report = run_pooled_calibration(ohlc, horizons=(1, 5), target_coverage=coverage)
        save_artifact(artifact)
        console.print(f"[green]Saved {artifact.version}[/green] → {DEFAULT_ARTIFACT_PATH}")
        for h, cal in sorted(artifact.horizons.items()):
            console.print(f"  h={h}d: z={cal.z:.3f} (n={cal.n})")
        console.print("[bold]Per-symbol coverage of the pooled z:[/bold]")
        for sym in sorted(report):
            parts = [
                f"h={h}d {v['coverage']:.0%} (n={v['n']})" for h, v in sorted(report[sym].items())
            ]
            console.print(f"  {sym:6} " + " · ".join(parts))
        console.print(
            "[dim]Watch for symbols far from the target — that is the signal "
            "pooling needs per-symbol shrinkage.[/dim]"
        )
        deviations = [
            abs(v["coverage"] - coverage) for by_h in report.values() for v in by_h.values()
        ]
        await _record_trial(
            name="bands.zcal.pooled_walkforward",
            kind="band_calibration",
            config={"days": days, "coverage": coverage, "horizons": [1, 5]},
            window=f"{days}d x {len(ohlc)}sym daily",
            metrics={
                "version": artifact.version,
                **{f"z_{h}d": cal.z for h, cal in sorted(artifact.horizons.items())},
                "max_symbol_coverage_deviation": max(deviations) if deviations else None,
            },
            criterion="per-symbol coverage of the pooled z within ±10pp of target",
            verdict=("pass" if deviations and max(deviations) <= 0.10 else "inconclusive"),
        )

    asyncio.run(_run())


@quant.command()
@click.argument("symbol")
@click.option("--days", default=200, show_default=True, help="Calendar days of daily bars")
def outlook(symbol: str, days: int) -> None:
    """Print SYMBOL's quantitative PriceOutlook (bands, vol state)."""

    async def _run() -> None:
        from halal_trader.quant.calibration import load_default_artifact
        from halal_trader.quant.outlook import build_outlook

        payloads = await _fetch_universe_bars([symbol.upper()], days, cache_read=False)
        ohlc = _payloads_to_ohlc(payloads)
        data = ohlc.get(symbol.upper())
        if data is None:
            console.print(f"[red]No usable bars for {symbol.upper()}.[/red]")
            return
        out = build_outlook(*data, calibration=load_default_artifact())
        if out is None:
            console.print(f"[yellow]{symbol.upper()}: series too thin for an outlook.[/yellow]")
            return
        tag = (
            f"[green]calibrated ({out.calibration_version})[/green]"
            if out.calibrated
            else "[yellow]UNCALIBRATED[/yellow]"
        )
        console.print(
            f"[bold]{symbol.upper()}[/bold] close=${out.close:,.2f} bars={out.n_bars} {tag}"
        )
        for h, hb in sorted(out.bands.items()):
            b = hb.band
            console.print(
                f"  {h}d band: ${b.low:,.2f} .. ${b.high:,.2f} "
                f"(σ={b.sigma_daily:.4f}/d via {hb.sigma_source}, z={b.z:.2f}, "
                f"E\\[range]=${b.expected_range:,.2f})"
            )
        if out.vol_percentile is not None:
            console.print(f"  vol percentile vs own history: {out.vol_percentile:.0%}")
        if out.atr_baseline_5d is not None:
            a = out.atr_baseline_5d
            console.print(f"  ATR baseline 5d: ${a.low:,.2f} .. ${a.high:,.2f}")

    asyncio.run(_run())
