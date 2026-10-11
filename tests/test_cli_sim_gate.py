"""`halal-trader events sim-gate {lookahead,reactor,sue,all}`: one line per gate, exit 1
when a gate fails or is refused (the runs themselves: test_sim_gate_db.py)."""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from halal_trader.cli import cli
from halal_trader.events import sim_gate
from halal_trader.events.sim_gate import GateResult, GateRun


def test_the_command_prints_every_gate_and_exits_1_on_a_failure_or_refusal(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, int, str | None]] = []

    async def run_gates(engine: Any, group: str, *, workers: int, repin: str | None) -> GateRun:
        seen.append((group, workers, repin))
        if group == "reactor":
            return GateRun([GateResult("r0", True, {"checks": {}}), GateResult("r1", True, {})])
        return GateRun([GateResult("s0", False, {})], {"s2": "s2/s3: 3 of 9 units are not done"})

    monkeypatch.setattr(sim_gate, "run_gates", run_gates)
    ok = CliRunner().invoke(cli, ["events", "sim-gate", "reactor", "--workers", "3"])
    assert ok.exit_code == 0, ok.output
    assert "PASS r0" in ok.output and "PASS r1" in ok.output and seen == [("reactor", 3, None)]
    bad = CliRunner().invoke(cli, ["events", "sim-gate", "sue"])
    assert bad.exit_code == 1
    assert "FAIL s0" in bad.output and "REFUSED s2" in bad.output
    assert "3 of 9 units are not done" in bad.output
    assert "failed: s0; refused: s2" in bad.output and seen[-1] == ("sue", 6, None)
    assert "Traceback" not in bad.output
    repinned = CliRunner().invoke(
        cli, ["events", "sim-gate", "reactor", "--repin", "  aliases rebuilt (2026-10-11) "]
    )
    assert repinned.exit_code == 0, repinned.output
    assert seen[-1] == ("reactor", 6, "aliases rebuilt (2026-10-11)")


def test_a_repin_needs_a_reason(database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    async def run_gates(engine: Any, group: str, **_: Any) -> GateRun:
        called.append(group)
        return GateRun()

    monkeypatch.setattr(sim_gate, "run_gates", run_gates)
    for blank in ("", "   "):
        result = CliRunner().invoke(cli, ["events", "sim-gate", "all", "--repin", blank])
        assert result.exit_code == 2 and "a re-pin needs a reason" in result.output
    assert called == []
    assert "--repin" in CliRunner().invoke(cli, ["events", "sim-gate", "--help"]).output


def test_the_command_takes_only_the_four_groups() -> None:
    result = CliRunner().invoke(cli, ["events", "sim-gate", "g2"])
    assert result.exit_code == 2 and "lookahead" in result.output
    assert "Phase 0 gates" in CliRunner().invoke(cli, ["events", "sim-gate", "--help"]).output
