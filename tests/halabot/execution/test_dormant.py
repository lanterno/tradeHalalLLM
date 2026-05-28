"""Guard: the execution layer is DORMANT — build_engine never wires a venue.

The whole point of Batch E is that execution code exists + is tested but the
engine NEVER trades until Phase-4 (ENGINE_LIVE + SAFEGUARD floors + a passed
significance gate). These tests fail loudly if a future change accidentally
wires an executor/venue/monitor into the read-only engine."""

from __future__ import annotations

import inspect

from halabot import app
from halabot.app import Engine


def test_engine_has_no_execution_fields():
    # The Engine dataclass must not expose an executor / venue / monitor.
    field_names = set(Engine.__dataclass_fields__)
    forbidden = {"executor", "venue", "monitor", "position_manager", "reconciler"}
    assert field_names.isdisjoint(forbidden), f"execution leaked into Engine: {field_names}"


def test_build_engine_source_does_not_import_execution():
    # build_engine's module must not reference the execution package (dormant).
    src = inspect.getsource(app)
    assert "halabot.execution" not in src, "app.py wired the dormant execution layer"
    assert "Executor(" not in src
    assert "PositionMonitor(" not in src


def test_live_defaults_off():
    from halabot.platform.config import HalabotSettings

    assert HalabotSettings().live_enabled is False  # shadow-only by default
