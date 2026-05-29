"""LiveModeChecker — flag-off default, dated token, un-loosenable SAFEGUARD (INV-9)."""

from __future__ import annotations

from datetime import date

from halabot.execution.live_mode import LiveModeChecker, expected_token
from halabot.platform.config import HalabotSettings

TODAY = date(2026, 5, 28)
CHECKER = LiveModeChecker()


def _settings(monkeypatch, **env) -> HalabotSettings:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return HalabotSettings()


def test_default_is_shadow_only(monkeypatch):
    monkeypatch.delenv("ENGINE_LIVE", raising=False)
    monkeypatch.delenv("ENGINE_LIVE_TOKEN", raising=False)
    d = CHECKER.check(HalabotSettings(), TODAY)
    assert d.armed is False
    assert "shadow only" in d.reason


def test_live_set_but_no_token_refused(monkeypatch):
    monkeypatch.setenv("ENGINE_LIVE", "stocks")
    monkeypatch.delenv("ENGINE_LIVE_TOKEN", raising=False)
    d = CHECKER.check(HalabotSettings(), TODAY)
    assert d.armed is False and "token" in d.reason.lower()


def test_stale_token_refused(monkeypatch):
    s = _settings(monkeypatch, ENGINE_LIVE="stocks", ENGINE_LIVE_TOKEN="LIVE-2020-01-01")
    d = CHECKER.check(s, TODAY)
    assert d.armed is False and "stale" in d.reason


def test_valid_dated_token_arms(monkeypatch):
    s = _settings(
        monkeypatch, ENGINE_LIVE="stocks", ENGINE_LIVE_TOKEN=expected_token(TODAY)
    )
    d = CHECKER.check(s, TODAY)
    assert d.armed is True
    assert d.market == "stocks"


def test_safeguard_order_cap_is_unloosenable(monkeypatch):
    # Even if config asks for a $50k order, the effective cap clamps to SAFEGUARD.
    s = _settings(
        monkeypatch,
        ENGINE_LIVE="stocks",
        ENGINE_LIVE_TOKEN=expected_token(TODAY),
        HALABOT_SAFEGUARD__LIVE_MAX_ORDER_USD="1000",
    )
    d = CHECKER.check(s, TODAY, requested_max_order_usd=50_000.0)
    assert d.armed is True
    assert d.max_order_usd == 1000.0  # clamped down, never up (INV-9)


def test_config_cannot_raise_floors_above_absolute_ceiling(monkeypatch):
    # INV-9 regression: raising the SAFEGUARD config in env must NOT raise the
    # effective ceilings above the hard-coded absolute floors.
    from halabot.execution.live_mode import (
        ABS_MAX_ACCOUNT_USD,
        ABS_MAX_ORDER_USD,
    )

    s = _settings(
        monkeypatch,
        ENGINE_LIVE="stocks",
        ENGINE_LIVE_TOKEN=expected_token(TODAY),
        HALABOT_SAFEGUARD__LIVE_MAX_ORDER_USD="1000000",
        HALABOT_SAFEGUARD__LIVE_MAX_ACCOUNT_USD="10000000",
    )
    d = CHECKER.check(s, TODAY, requested_max_order_usd=500_000.0)
    assert d.max_order_usd == ABS_MAX_ORDER_USD  # capped at the code floor, not the env
    assert d.max_account_usd == ABS_MAX_ACCOUNT_USD


def test_malformed_token_refused(monkeypatch):
    s = _settings(monkeypatch, ENGINE_LIVE="stocks", ENGINE_LIVE_TOKEN="yolo")
    d = CHECKER.check(s, TODAY)
    assert d.armed is False and "malformed" in d.reason


def test_effective_caps_present_even_when_unarmed(monkeypatch):
    # The decision always reports the clamped caps (for display), even unarmed.
    d = CHECKER.check(HalabotSettings(), TODAY)
    assert d.max_order_usd > 0 and d.max_account_usd > 0
