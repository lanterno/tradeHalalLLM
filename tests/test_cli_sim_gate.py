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
    seen: list[tuple[str, int]] = []

    async def run_gates(engine: Any, group: str, *, workers: int) -> GateRun:
        seen.append((group, workers))
        if group == "reactor":
            return GateRun([GateResult("r0", True, {"checks": {}}), GateResult("r1", True, {})])
        return GateRun([GateResult("s0", False, {})], {"s2": "s2/s3: 3 of 9 units are not done"})

    monkeypatch.setattr(sim_gate, "run_gates", run_gates)
    ok = CliRunner().invoke(cli, ["events", "sim-gate", "reactor", "--workers", "3"])
    assert ok.exit_code == 0, ok.output
    assert "PASS r0" in ok.output and "PASS r1" in ok.output and seen == [("reactor", 3)]
    bad = CliRunner().invoke(cli, ["events", "sim-gate", "sue"])
    assert bad.exit_code == 1
    assert "FAIL s0" in bad.output and "REFUSED s2" in bad.output
    assert "3 of 9 units are not done" in bad.output
    assert "failed: s0; refused: s2" in bad.output and seen[-1] == ("sue", 6)
    assert "Traceback" not in bad.output


def test_the_command_takes_only_the_four_groups() -> None:
    result = CliRunner().invoke(cli, ["events", "sim-gate", "g2"])
    assert result.exit_code == 2 and "lookahead" in result.output
    assert "Phase 0 gates" in CliRunner().invoke(cli, ["events", "sim-gate", "--help"]).output
