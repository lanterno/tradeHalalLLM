"""Guard: the execution layer is DORMANT — build_engine never wires a venue.

The whole point of Batch E is that execution code exists + is tested but the
engine NEVER trades until Phase-4 (ENGINE_LIVE + SAFEGUARD floors + a passed
significance gate). These tests fail loudly if a future change accidentally
wires an executor/venue/monitor into the read-only engine."""

from __future__ import annotations

import subprocess
import sys

from halabot.app import Engine


def test_engine_has_no_execution_fields():
    # The Engine dataclass must not expose an executor / venue / monitor.
    field_names = set(Engine.__dataclass_fields__)
    forbidden = {"executor", "venue", "monitor", "position_manager", "reconciler"}
    assert field_names.isdisjoint(forbidden), f"execution leaked into Engine: {field_names}"


def test_importing_app_does_not_transitively_import_execution():
    # Transitive (not lexical) guard: in a FRESH interpreter, importing the
    # composition root must not pull in ANY halabot.execution.* module — proving
    # the dormant layer is never reachable from build_engine's import graph.
    code = (
        "import halabot.app, sys; "
        "leaked = sorted(m for m in sys.modules if m.startswith('halabot.execution')); "
        "assert not leaked, leaked; print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"execution leaked into app import graph: {result.stderr}"


def test_live_defaults_off():
    from halabot.platform.config import HalabotSettings

    assert HalabotSettings().live_enabled is False  # shadow-only by default
