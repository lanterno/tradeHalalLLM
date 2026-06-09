"""Watchdog dead-man helper: rotation-aware activity detection."""

from __future__ import annotations

import json

from halal_trader.cli.watchdog import _find_last_activity


def _line(name: str, ts: str, event: str | None = None) -> str:
    d: dict = {"name": name, "timestamp": ts}
    if event:
        d["event"] = event
    return json.dumps(d)


def test_find_last_activity_in_current_file(tmp_path):
    log = tmp_path / "halal_trader.log"
    log.write_text(
        _line("halal_trader.trading.cycle", "2026-06-09 15:30:00,000", "cycle.start") + "\n"
    )
    res = _find_last_activity(log, "halal_trader.trading")
    assert res is not None and res[0] == "cycle.start"


def test_find_last_activity_falls_back_to_rotated_file(tmp_path):
    """REGRESSION (2026-06-09): the bot's DEBUG flood rotated the log ~5x/session;
    reading the fresh post-rotation log found no matching activity and returned
    None, which crashed the watchdog's alert path. It must fall back to .log.1."""
    log = tmp_path / "halal_trader.log"
    (tmp_path / "halal_trader.log.1").write_text(
        _line("halal_trader.trading.cycle", "2026-06-09 15:30:00,000", "cycle.start") + "\n"
    )
    # Current file: only unrelated lines (just rotated).
    log.write_text(_line("halal_trader.web.metrics", "2026-06-09 15:35:00,000") + "\n")
    res = _find_last_activity(log, "halal_trader.trading")
    assert res is not None and res[0] == "cycle.start"  # found in .log.1, not None


def test_find_last_activity_none_when_no_match_anywhere(tmp_path):
    log = tmp_path / "halal_trader.log"
    log.write_text(_line("other.module", "2026-06-09 15:30:00,000") + "\n")
    assert _find_last_activity(log, "halal_trader.trading") is None
