"""`halal-trader events h1`: the pre-registered H1 trial, one step at a time (spec §G.15).

register -> stage-a -> train -> validation -> verdict -> sensitivities and
implementability. Each step refuses until the one before it is on the
ledger; the work is ``halal_trader.events.h1``. Attached to ``events`` in
``cli/__init__.py``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import click

from halal_trader.cli._run import fail, run_db
from halal_trader.logging import console

_WORKERS = click.option(
    "--workers", default=6, show_default=True, help="Simulator workers (serial on macOS)."
)


def _num(x: Any, fmt: str) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return format(x, fmt)


def _line(text: str) -> None:
    console.print(text, markup=False, highlight=False)


@click.group("h1")
def h1() -> None:
    """H1, the overreaction bounce: the pre-registered trial (spec §G)."""


@h1.command("register")
@click.option("--dry-run", is_flag=True, help="Run the checks and print them; record nothing.")
@click.option("--no-scan", is_flag=True, help="Leave out D5's full reader scan (dry runs only).")
def register_cmd(dry_run: bool, no_scan: bool) -> None:
    """Check the gates and the data preconditions, then register H1 (once)."""
    if no_scan and not dry_run:
        fail("--no-scan is for dry runs: a registration needs every check")

    async def _run(engine: Any, settings: Any) -> tuple[Any, str | None, int | None]:
        from halal_trader.events import h1 as trial

        existing = await trial.existing_registration(engine)
        if existing is not None and not dry_run:
            return None, f"H1 is registered already (quant_trials {existing})", None
        report = await trial.preconditions(engine, scan=not no_scan)
        if dry_run or not report.ok:
            return report, None, None
        try:
            return report, None, await trial.register(engine, report=report)
        except trial.RegistrationRefused as exc:
            return report, str(exc), None

    report, error, trial_id = run_db(_run)
    if report is not None:
        for check in report.checks:
            mark = "[green]ok  [/green]" if check.ok else "[red]FAIL[/red]"
            console.print(f"{mark} {check.id:<3} {check.detail}")
        for missing in report.missing:
            console.print(f"[yellow]--  [/yellow] {missing:<3} not run")
        d9 = report.get("D9")
        if d9 is not None:
            for text in d9.data.get("table", []):
                _line(text)
    if error:
        fail(error)
    if trial_id is not None:
        console.print(f"H1 registered: quant_trials {trial_id}")
    elif not dry_run:
        fail("preconditions fail: nothing registered")


@h1.command("stage-a")
@_WORKERS
def stage_a_cmd(workers: int) -> None:
    """Count entries in both windows without any exit or return; decide eligible cells."""
    from halal_trader.events.h1 import H1Locked, StoriesStale

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.h1 import stage_a

        return await stage_a(engine, workers=workers)

    try:
        result = run_db(_run)
    except (H1Locked, StoriesStale) as exc:
        fail(str(exc))
    _line(
        f"{'window':<11}{'cell':<14}{'stories':>8}{'nsn':>7}{'elig':>7}{'blocked':>8}"
        f"{'trig':>7}{'armed':>7}{'entries':>8}{'dates':>7}{'/yr':>8}{'skips':>7}{'skip%':>7}"
    )
    for (window, key), c in sorted(result.counts.items()):
        _line(
            f"{window:<11}{key:<14}{c.stories:>8}{c.nsn:>7}{c.eligible:>7}{c.blocked_open:>8}"
            f"{c.triggered:>7}{c.armed:>7}{c.entries:>8}{c.dates:>7}{c.per_year:>8.1f}"
            f"{c.data_skips:>7}{c.skip_share:>7.2%}"
        )
        if c.skips:
            _line("  skips: " + ", ".join(f"{k} {n}" for k, n in sorted(c.skips.items())))
        if c.expired:
            _line("  expired: " + ", ".join(f"{k} {n}" for k, n in sorted(c.expired.items())))
    eligible = ", ".join(c.key for c in result.eligible) or "none"
    console.print(
        f"Stage A ({result.verdict}): eligible cells {eligible}; quant_trials {result.trial_id}"
    )


def _print_window(results: Any) -> None:
    _line(
        f"{'cell':<14}{'trades':>7}{'dates':>7}{'mean':>9}{'t_cr1':>7}{'t_nw':>7}"
        f"{'x1.5':>9}{'beta':>9}{'ex-cov':>9}{'unres':>6}  T1-T6   {'status':<13}{'dsr':>6}"
        f"{'dsr+6':>7}"
    )
    for cell, ws in results.items():
        tests = "".join("y" if ok else "n" for ok in (ws.t1, ws.t2, ws.t3, ws.t4, ws.t5, ws.t6))
        t_cr1 = ws.cr1.t if ws.cr1 is not None else None
        t_nw = ws.nw.t if ws.nw is not None else None
        _line(
            f"{cell.key:<14}{ws.n:>7}{ws.dates:>7}{_num(ws.mean, '+.3%'):>9}"
            f"{_num(t_cr1, '+.2f'):>7}{_num(t_nw, '+.2f'):>7}{_num(ws.mean_cost15, '+.3%'):>9}"
            f"{_num(ws.mean_beta, '+.3%'):>9}{_num(ws.mean_ex_covid, '+.3%'):>9}"
            f"{ws.unresolved:>6}  {tests:<8}{ws.status:<13}{_num(ws.dsr, '.2f'):>6}"
            f"{_num(ws.dsr_plus6, '.2f'):>7}"
        )


def _window(window: Literal["train", "validation"], workers: int, amend: str | None) -> None:
    from halal_trader.events.h1 import H1Locked, StoriesStale

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.h1 import run_window

        return await run_window(engine, window, workers=workers, amend=amend)

    try:
        results = run_db(_run)
    except (H1Locked, StoriesStale, ValueError) as exc:
        fail(str(exc))
    _print_window(results)


_AMEND = click.option(
    "--amend", default=None, help="Rerun a window that has run; the amendment's reason."
)


@h1.command("train")
@_WORKERS
@_AMEND
def train_cmd(workers: int, amend: str | None) -> None:
    """Run Stage A's eligible cells on train (2016-10-03..2021-12-31) and record each trial."""
    _window("train", workers, amend)


@h1.command("validation")
@_WORKERS
@_AMEND
def validation_cmd(workers: int, amend: str | None) -> None:
    """Run the train passers on validation (2022-01-03..2024-12-31) and record each trial."""
    _window("validation", workers, amend)


@h1.command("verdict")
def verdict_cmd() -> None:
    """Record H1's verdict: pass, fail or inconclusive."""
    from halal_trader.events.h1 import H1Locked

    async def _run(engine: Any, settings: Any) -> str:
        from halal_trader.events.h1 import verdict

        return await verdict(engine)

    try:
        decision = run_db(_run)
    except H1Locked as exc:
        fail(str(exc))
    console.print(f"H1 verdict: {decision}")


@h1.command("sensitivities")
@_WORKERS
def sensitivities_cmd(workers: int) -> None:
    """Every sensitivity (spec §G.12), recorded and never counted as a trial."""
    from halal_trader.events.h1 import H1Locked, StoriesStale

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.h1 import sensitivities

        return await sensitivities(engine, workers=workers)

    try:
        results = run_db(_run)
    except (H1Locked, StoriesStale) as exc:
        fail(str(exc))

    def show(label: str, numbers: Any) -> None:
        if isinstance(numbers, dict) and "trades" in numbers:
            _line(
                f"{label:<54}{numbers['trades']:>7}{_num(numbers['mean_trade'], '+.3%'):>9}"
                f"{_num(numbers['t_cr1'], '+.2f'):>7}{_num(numbers['t_nw'], '+.2f'):>7}"
            )
        elif isinstance(numbers, dict):
            for part, inner in numbers.items():
                show(f"{label} {part}", inner)
        else:
            _line(f"{label:<54}{numbers}")

    _line(f"{'sensitivity':<54}{'trades':>7}{'mean':>9}{'t_cr1':>7}{'t_nw':>7}")
    for name, by_cell in results.items():
        for where, numbers in by_cell.items():
            show(f"{name} {where}", numbers)


@h1.command("implementability")
@_WORKERS
def implementability_cmd(workers: int) -> None:
    """The cells that passed H1 on the delayed SIP feed (a counted trial, spec §G.13)."""
    from halal_trader.events.h1 import H1Locked, StoriesStale

    async def _run(engine: Any, settings: Any) -> Any:
        from halal_trader.events.h1 import implementability

        return await implementability(engine, workers=workers)

    try:
        results = run_db(_run)
    except (H1Locked, StoriesStale) as exc:
        fail(str(exc))
    if not results:
        console.print("no cell passed H1: the implementability trial does not run")
        return
    for window, by_cell in results.items():
        console.print(f"[bold]{window}[/bold] (sip-delayed)")
        _print_window(by_cell)
