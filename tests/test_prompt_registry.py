"""Prompt registry tests — hashing, idempotence, and existing registrations."""

import pytest

from halal_trader.core.llm.prompts import get_version, register
from halal_trader.core.llm.prompts.registry import _reset_for_tests


def test_register_returns_stable_hash():
    _reset_for_tests()
    pv = register("crypto.test.system", "You are an expert trader.\nFollow the rules.")
    # 12-char prefix of sha256 — known fixture so we catch silent algo changes.
    assert pv.version_id == "2c8a650bf4b8"
    assert pv.short == "crypto.test.system@2c8a650bf4b8"


def test_register_is_idempotent_for_identical_template():
    _reset_for_tests()
    a = register("p", "hello")
    b = register("p", "hello")
    assert a is b


def test_register_rejects_silent_overwrite():
    _reset_for_tests()
    register("p", "hello")
    with pytest.raises(ValueError, match="already registered"):
        register("p", "hello, world")


def test_get_version_returns_registered_entry():
    _reset_for_tests()
    register("p", "hello")
    assert get_version("p").template == "hello"


def test_existing_strategy_prompts_expose_version_constants():
    """The strategy modules expose ``PROMPT_VERSION`` constants from the registry.

    We assert against the module attributes (rather than the live registry)
    because Python's import cache means an earlier test that called
    ``_reset_for_tests`` would otherwise produce a false negative — the
    modules don't re-register on subsequent imports.
    """
    import halal_trader.crypto.prompts as crypto_prompts
    import halal_trader.trading.strategy as trading_strategy

    assert crypto_prompts.PROMPT_VERSION.name == "crypto.strategy.system"
    assert trading_strategy.PROMPT_VERSION.name == "trading.strategy.system"
    assert trading_strategy.USER_PROMPT_VERSION.name == "trading.strategy.user"
    for pv in (
        crypto_prompts.PROMPT_VERSION,
        trading_strategy.PROMPT_VERSION,
        trading_strategy.USER_PROMPT_VERSION,
    ):
        assert len(pv.version_id) == 12
        int(pv.version_id, 16)
