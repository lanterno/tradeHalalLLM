"""HalabotSettings — env loading, nesting, aliases, and the live-mode gate."""

from __future__ import annotations

import pytest

from halabot.platform.config import HalabotSettings


def test_defaults_are_shadow_only():
    s = HalabotSettings()
    assert s.live == ""
    assert s.live_enabled is False
    # The shared URL is an asyncpg Postgres URL (env may point it at the test DB).
    assert s.database_url.startswith("postgresql+asyncpg://")
    # Cold-start policy bands satisfy the exit < entry invariant.
    assert s.policy.conviction_exit_band < s.policy.conviction_entry_band


def test_database_url_reads_plain_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h:1/db")
    s = HalabotSettings()
    assert s.database_url == "postgresql+asyncpg://u:p@h:1/db"


def test_engine_live_arms_live_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENGINE_LIVE", "stocks")
    s = HalabotSettings()
    assert s.live == "stocks"
    assert s.live_enabled is True


def test_blank_engine_live_stays_shadow(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENGINE_LIVE", "   ")
    assert HalabotSettings().live_enabled is False


def test_nested_override_via_delimiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HALABOT_POLICY__CONVICTION_ENTRY_BAND", "0.4")
    monkeypatch.setenv("HALABOT_ENGINE__HEARTBEAT_INTERVAL_S", "30")
    s = HalabotSettings()
    assert s.policy.conviction_entry_band == 0.4
    assert s.engine.heartbeat_interval_s == 30.0


def test_execution_group_nests_with_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HALABOT_EXECUTION__MIN_NOTIONAL_USD", "75")
    s = HalabotSettings()
    assert s.execution.min_notional_usd == 75.0


def test_safeguard_floors_present():
    s = HalabotSettings()
    assert s.safeguard.live_max_account_usd > 0
    assert s.safeguard.live_max_order_usd > 0
    assert 0 < s.safeguard.live_daily_loss_floor_pct <= 1
