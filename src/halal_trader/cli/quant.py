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


@click.group()
def quant() -> None:
    """Quantitative range-model tools (advisory — never trades)."""


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
            series[sym] = (
                [r[0] for r in rows],
                np.asarray([r[1] for r in rows]),
                np.asarray([r[2] for r in rows]),
                np.asarray([r[3] for r in rows]),
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
        console.print(
            "[dim]Prompt inclusion requires beating placebo on DISJOINT OOS "
            "windows with a sufficient sample — record the verdict in the "
            "trials ledger before wiring anything.[/dim]"
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
