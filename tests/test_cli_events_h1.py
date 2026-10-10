"""`halal-trader events h1 ...`: the steps print their checks and refuse out of order."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.cli import cli
from halal_trader.events import h1
from halal_trader.events.h1 import Check, Preconditions


def _report(*, failing: str | None = None) -> Preconditions:
    checks = []
    for i in h1.REQUIRED_CHECKS:
        data: dict[str, Any] = {"table": ["== all stories ==", "ALL  12"]} if i == "D9" else {}
        if i == "C0":
            data = {"commit": "c" * 40}
        checks.append(Check(i, i != failing, f"{i} says so", data))
    return Preconditions(tuple(checks))


def _patch(monkeypatch: pytest.MonkeyPatch, report: Preconditions, seen: list[bool]) -> None:
    async def preconditions(engine: Any, *, scan: bool = True, **_: Any) -> Preconditions:
        seen.append(scan)
        return report

    monkeypatch.setattr(h1, "preconditions", preconditions)


def _count(database_url: str) -> int:
    async def go() -> int:
        engine = create_async_engine(database_url)
        try:
            async with engine.connect() as conn:
                return int(await conn.scalar(text("SELECT count(*) FROM quant_trials")) or 0)
        finally:
            await engine.dispose()

    return asyncio.run(go())


def test_a_dry_run_prints_every_check_and_records_nothing(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    _patch(monkeypatch, _report(failing="D3"), seen)
    result = CliRunner().invoke(cli, ["events", "h1", "register", "--dry-run", "--no-scan"])
    assert result.exit_code == 0, result.output
    assert "ok   C0  C0 says so" in result.output and "FAIL D3  D3 says so" in result.output
    assert "ALL  12" in result.output and seen == [False]
    assert _count(database_url) == 0


def test_the_reported_checks_print_as_information_and_never_block(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report()
    reported = (
        Check("D7", True, "not measured: no Alpaca client (reported, not gating)"),
        Check("D8", False, "a probe that broke"),
    )
    _patch(monkeypatch, Preconditions(report.checks + reported), [])
    result = CliRunner().invoke(cli, ["events", "h1", "register"])
    assert result.exit_code == 0, result.output
    assert "info D7  not measured" in result.output and "info D8  a probe" in result.output
    assert "H1 registered" in result.output and _count(database_url) == 1


def test_no_scan_is_for_dry_runs_only() -> None:
    result = CliRunner().invoke(cli, ["events", "h1", "register", "--no-scan"])
    assert result.exit_code == 1 and "for dry runs" in result.output


def test_a_failing_check_registers_nothing(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _report(failing="G"), [])
    result = CliRunner().invoke(cli, ["events", "h1", "register"])
    assert result.exit_code == 1 and "preconditions fail" in result.output
    assert "FAIL G" in result.output and "Traceback" not in result.output
    assert _count(database_url) == 0


def test_h1_registers_once(database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    _patch(monkeypatch, _report(), seen)
    result = CliRunner().invoke(cli, ["events", "h1", "register"])
    assert result.exit_code == 0, result.output
    assert "H1 registered: quant_trials" in result.output and seen == [True]
    again = CliRunner().invoke(cli, ["events", "h1", "register"])
    assert again.exit_code == 1 and "registered already" in again.output
    assert seen == [True]  # refused before any check ran again
    assert _count(database_url) == 1


@pytest.mark.usefixtures("database_url")
@pytest.mark.parametrize(
    "step",
    [
        ["stage-a", "--workers", "1"],
        ["train", "--workers", "1"],
        ["validation", "--workers", "1"],
        ["verdict"],
        ["sensitivities", "--workers", "1"],
        ["implementability", "--workers", "1"],
    ],
)
def test_every_step_refuses_before_the_registration(step: list[str]) -> None:
    result = CliRunner().invoke(cli, ["events", "h1", *step])
    assert result.exit_code == 1, result.output
    assert "not registered" in result.output and "Traceback" not in result.output
