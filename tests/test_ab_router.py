"""A/B prompt-router tests — determinism, weighting, edge cases."""

from __future__ import annotations

from collections import Counter

import pytest

from halal_trader.core.llm.ab import ABRouter, PromptVariant, expected_split
from halal_trader.core.llm.prompts.registry import (
    PromptVersion,
    _reset_for_tests,
    register,
)


def _v(name: str, text: str) -> PromptVersion:
    _reset_for_tests()
    return register(name, text)


def _variant(name: str = "v1", text: str = "system v1", *, weight: float = 1.0) -> PromptVariant:
    return PromptVariant(version=_v(name, text), system=text, user="user", weight=weight)


def test_router_with_one_variant_always_returns_it():
    only = _variant()
    router = ABRouter([only])
    for key in ("a", "b", "c"):
        assert router.choose(key).version == only.version


def test_router_deterministic_for_same_key():
    a = PromptVariant(version=register("a", "sys-a"), system="sys-a", user="u")
    b = PromptVariant(version=register("b", "sys-b"), system="sys-b", user="u")
    router = ABRouter([a, b])
    first = router.choose("cycle-42")
    for _ in range(20):
        assert router.choose("cycle-42") is first


def test_router_distributes_evenly_at_50_50():
    _reset_for_tests()
    a = PromptVariant(version=register("a", "sys-a"), system="sys-a", user="u", weight=1)
    b = PromptVariant(version=register("b", "sys-b"), system="sys-b", user="u", weight=1)
    router = ABRouter([a, b])

    counts = Counter()
    for i in range(2000):
        counts[router.choose(f"cycle-{i}").version.name] += 1
    # SHA-256 over a uniform key space → close to 50/50; assert within ±10%.
    assert 800 < counts["a"] < 1200
    assert 800 < counts["b"] < 1200


def test_router_respects_weights():
    _reset_for_tests()
    a = PromptVariant(version=register("a", "sys-a"), system="sys-a", user="u", weight=9)
    b = PromptVariant(version=register("b", "sys-b"), system="sys-b", user="u", weight=1)
    router = ABRouter([a, b])

    counts = Counter()
    for i in range(5000):
        counts[router.choose(f"cycle-{i}").version.name] += 1
    # Expect ~90/10. Allow ±2 percentage points.
    a_share = counts["a"] / 5000
    assert 0.85 < a_share < 0.95


def test_router_zero_or_negative_weight_raises():
    _reset_for_tests()
    a = PromptVariant(version=register("a", "x"), system="x", user="u", weight=0)
    with pytest.raises(ValueError):
        ABRouter([a])


def test_router_no_variants_raises():
    with pytest.raises(ValueError):
        ABRouter([])


def test_expected_split_normalises_weights():
    _reset_for_tests()
    a = PromptVariant(version=register("a", "x"), system="x", user="u", weight=3)
    b = PromptVariant(version=register("b", "y"), system="y", user="u", weight=1)
    router = ABRouter([a, b])
    split = expected_split(router)
    assert split[f"a@{a.version.version_id}"] == 0.75
    assert split[f"b@{b.version.version_id}"] == 0.25
