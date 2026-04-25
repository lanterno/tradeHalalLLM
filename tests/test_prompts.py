"""Tests for crypto/prompts.py — the unified prompt builder."""

from __future__ import annotations

import pytest

from halal_trader.crypto.prompts import (
    PromptContext,
    StrategyParams,
    build_prompts,
    prompt_cache_key,
)
from halal_trader.domain.models import CryptoAccount


def _account(total: float = 1000.0, free: float = 800.0) -> CryptoAccount:
    return CryptoAccount(
        total_balance_usdt=total,
        available_balance_usdt=free,
        in_order_usdt=total - free,
        usdt_free=free,
    )


def _params(**overrides) -> StrategyParams:
    base = dict(
        max_position_pct=0.25,
        daily_loss_limit=0.03,
        daily_return_target=0.01,
        max_positions=4,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )
    base.update(overrides)
    return StrategyParams(**base)


def test_build_prompts_returns_strings():
    ctx = PromptContext(account=_account(), halal_pairs=["BTCUSDT"])
    system, user = build_prompts(ctx, _params())
    assert isinstance(system, str) and isinstance(user, str)
    assert "PORTFOLIO STATUS" in user
    assert "TASK" in user


def test_build_prompts_includes_strategy_knobs_in_system():
    system, _ = build_prompts(
        PromptContext(account=_account()),
        _params(stop_loss_pct=0.015, take_profit_pct=0.025),
    )
    assert "1.5%" in system
    assert "2.5%" in system


def test_build_prompts_collapses_empty_optional_sections():
    ctx = PromptContext(account=_account())
    _, user = build_prompts(ctx, _params())
    # Every optional section starts with "===" so when empty its block is dropped.
    assert "SOCIAL SENTIMENT" not in user
    assert "MULTI-TIMEFRAME" not in user
    assert "ML MODEL SIGNALS" not in user
    assert "MARKET REGIME" not in user
    assert "PORTFOLIO RISK" not in user


def test_build_prompts_keeps_optional_sections_when_set():
    ctx = PromptContext(
        account=_account(),
        sentiment_text="reddit buzz: high",
        regime_text="TRENDING_UP confidence=0.8",
        risk_text="heat=0.02 drawdown=0.01",
    )
    _, user = build_prompts(ctx, _params())
    assert "SOCIAL SENTIMENT" in user
    assert "reddit buzz" in user
    assert "MARKET REGIME" in user
    assert "TRENDING_UP" in user
    assert "PORTFOLIO RISK" in user
    assert "drawdown=0.01" in user


def test_build_prompts_sell_only_mode_at_max_positions():
    ctx = PromptContext(account=_account(), open_position_count=4)
    system, user = build_prompts(ctx, _params(max_positions=4))
    assert "SELL-ONLY MODE" in system
    assert "POSITION LIMIT REACHED" in user


def test_build_prompts_warning_at_one_slot_remaining():
    ctx = PromptContext(account=_account(), open_position_count=3)
    system, user = build_prompts(ctx, _params(max_positions=4))
    assert "SELL-ONLY MODE" not in system
    assert "Only 1 position slot remaining" in user


def test_build_prompts_zero_balance_falls_back_to_1000():
    ctx = PromptContext(account=_account(total=0.0, free=0.0))
    _, user = build_prompts(ctx, _params())
    assert "$0.00 USDT" in user  # rendered total
    assert "Today's P&L" in user


def test_prompt_cache_key_stable_across_calls():
    ctx = PromptContext(account=_account(), halal_pairs=["BTCUSDT"])
    s1, u1 = build_prompts(ctx, _params())
    s2, u2 = build_prompts(ctx, _params())
    assert prompt_cache_key(s1, u1) == prompt_cache_key(s2, u2)


def test_prompt_cache_key_differs_on_template_change():
    ctx = PromptContext(account=_account(), halal_pairs=["BTCUSDT"])
    s, u = build_prompts(ctx, _params())
    k1 = prompt_cache_key(s, u)
    # Mutate the user prompt — simulating a template edit.
    k2 = prompt_cache_key(s, u + "\n# manual edit")
    assert k1 != k2


def test_prompt_cache_key_differs_on_indicator_change():
    a = build_prompts(
        PromptContext(account=_account(), risk_text="heat=0.02"),
        _params(),
    )
    b = build_prompts(
        PromptContext(account=_account(), risk_text="heat=0.05"),
        _params(),
    )
    assert prompt_cache_key(*a) != prompt_cache_key(*b)
