"""INVARIANT: randomised (sandbox) halal verdicts never decide what the bot trades."""

from __future__ import annotations

from types import SimpleNamespace

from halal_trader.trading.scheduler import _zoya_for


def _settings(key: str, sandbox: bool) -> SimpleNamespace:
    return SimpleNamespace(zoya=SimpleNamespace(api_key=key, use_sandbox=sandbox))


def test_a_sandbox_key_is_not_trusted() -> None:
    assert _zoya_for(_settings("sk-sandbox", sandbox=True)) is None


def test_no_key_means_the_default_list() -> None:
    assert _zoya_for(_settings("", sandbox=False)) is None


def test_a_production_key_is_used() -> None:
    client = _zoya_for(_settings("sk-prod", sandbox=False))
    assert client is not None and client.api_key == "sk-prod"
