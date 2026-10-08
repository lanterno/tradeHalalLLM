"""Prompt registry tests — hashing, idempotence, and existing registrations."""

import pytest

from halal_trader.core.llm.prompts import register, registry


@pytest.fixture(autouse=True)
def _empty_registry():
    """Each test starts from an empty registry and leaves the real one intact."""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_register_returns_stable_hash():
    pv = register("stocks.test.system", "You are an expert trader.\nFollow the rules.")
    # 12-char prefix of sha256 — known fixture so we catch silent algo changes.
    assert pv.version_id == "2c8a650bf4b8"
    assert pv.short == "stocks.test.system@2c8a650bf4b8"


def test_register_is_idempotent_for_identical_template():
    a = register("p", "hello")
    b = register("p", "hello")
    assert a is b


def test_register_rejects_silent_overwrite():
    register("p", "hello")
    with pytest.raises(ValueError, match="already registered"):
        register("p", "hello, world")


def test_existing_strategy_prompts_expose_version_constants():
    """The strategy modules expose ``PROMPT_VERSION`` constants from the registry.

    We assert against the module attributes (rather than the live registry)
    because the autouse fixture empties the registry for each test and the
    modules don't re-register on subsequent imports.
    """
    import halal_trader.trading.strategy as trading_strategy

    assert trading_strategy.PROMPT_VERSION.name == "trading.strategy.system"
    assert trading_strategy.USER_PROMPT_VERSION.name == "trading.strategy.user"
    for pv in (
        trading_strategy.PROMPT_VERSION,
        trading_strategy.USER_PROMPT_VERSION,
    ):
        assert len(pv.version_id) == 12
        int(pv.version_id, 16)
