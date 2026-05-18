"""Tests for domain/policies.py — Round-5 Wave 0.G."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from halal_trader.domain.policies import (
    known_policy_ids,
    policies,
    policy_class,
    register_policy,
    register_policy_decorator,
    render_registry,
    reset_registry_for_testing,
    snapshot_from_dict,
    snapshot_to_dict,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


# Sample policies for testing


class _Mode(str, Enum):
    A = "a"
    B = "b"


@dataclass(frozen=True)
class _ExamplePolicy:
    threshold: float = 0.5
    mode: _Mode = _Mode.A
    label: str = "default"


@dataclass(frozen=True)
class _OtherPolicy:
    n: int = 1


# --- Registration ------------------------------------------------------------


def test_register_policy_returns_class():
    out = register_policy("example", _ExamplePolicy)
    assert out is _ExamplePolicy


def test_register_empty_id_rejected():
    with pytest.raises(ValueError):
        register_policy("", _ExamplePolicy)


def test_register_non_dataclass_rejected():
    class Plain:
        pass

    with pytest.raises(TypeError):
        register_policy("plain", Plain)


def test_register_idempotent_same_class():
    register_policy("example", _ExamplePolicy)
    # Second call with same class is OK
    register_policy("example", _ExamplePolicy)
    assert policy_class("example") is _ExamplePolicy


def test_register_different_class_under_same_id_rejected():
    register_policy("example", _ExamplePolicy)
    with pytest.raises(ValueError):
        register_policy("example", _OtherPolicy)


def test_register_decorator_form():
    @register_policy_decorator("example")
    @dataclass(frozen=True)
    class P:
        x: int = 0

    assert policy_class("example") is P


def test_policy_class_unknown_returns_none():
    assert policy_class("never-registered") is None


def test_known_policy_ids_in_registration_order():
    register_policy("first", _ExamplePolicy)
    register_policy("second", _OtherPolicy)
    assert known_policy_ids() == ("first", "second")


def test_policies_view_is_a_copy():
    """Mutating the returned mapping should NOT affect the registry."""
    register_policy("example", _ExamplePolicy)
    view = policies()
    view["example"] = _OtherPolicy  # type: ignore[index]
    assert policy_class("example") is _ExamplePolicy


# --- Snapshot round-trip ----------------------------------------------------


def test_snapshot_to_dict_basic():
    p = _ExamplePolicy(threshold=0.7, mode=_Mode.B, label="x")
    snap = snapshot_to_dict(p)
    assert snap == {"threshold": 0.7, "mode": "b", "label": "x"}


def test_snapshot_to_dict_with_defaults():
    snap = snapshot_to_dict(_ExamplePolicy())
    assert snap == {"threshold": 0.5, "mode": "a", "label": "default"}


def test_snapshot_to_dict_nested_dataclass():
    @dataclass(frozen=True)
    class Inner:
        v: int = 1

    @dataclass(frozen=True)
    class Outer:
        inner: Inner = Inner()

    snap = snapshot_to_dict(Outer())
    assert snap == {"inner": {"v": 1}}


def test_snapshot_to_dict_with_frozenset():
    @dataclass(frozen=True)
    class P:
        flags: frozenset[int] = frozenset({3, 1, 2})

    snap = snapshot_to_dict(P())
    assert snap == {"flags": [1, 2, 3]}


def test_snapshot_to_dict_rejects_class():
    with pytest.raises(TypeError):
        snapshot_to_dict(_ExamplePolicy)


def test_snapshot_to_dict_rejects_non_dataclass():
    with pytest.raises(TypeError):
        snapshot_to_dict({"x": 1})  # type: ignore[arg-type]


def test_snapshot_roundtrip_simple():
    register_policy("example", _ExamplePolicy)
    p = _ExamplePolicy(threshold=0.9, mode=_Mode.B)
    snap = snapshot_to_dict(p)
    p2 = snapshot_from_dict("example", snap)
    assert p2 == p


def test_snapshot_from_dict_unknown_id_raises():
    with pytest.raises(KeyError):
        snapshot_from_dict("nope", {})


def test_snapshot_from_dict_unknown_field_rejected():
    register_policy("example", _ExamplePolicy)
    with pytest.raises(ValueError):
        snapshot_from_dict("example", {"threshold": 0.5, "extra": "x"})


def test_snapshot_from_dict_partial_uses_defaults():
    register_policy("example", _ExamplePolicy)
    p = snapshot_from_dict("example", {"threshold": 0.99})
    assert p == _ExamplePolicy(threshold=0.99)


def test_snapshot_from_dict_non_mapping_rejected():
    register_policy("example", _ExamplePolicy)
    with pytest.raises(TypeError):
        snapshot_from_dict("example", [("threshold", 0.5)])  # type: ignore[arg-type]


# --- Render -----------------------------------------------------------------


def test_render_empty_registry():
    out = render_registry()
    assert "No policies registered" in out


def test_render_lists_registered():
    register_policy("example", _ExamplePolicy)
    register_policy("other", _OtherPolicy)
    out = render_registry()
    assert "example" in out
    assert "other" in out
    assert "_ExamplePolicy" in out


def test_render_sorted():
    register_policy("zeta", _ExamplePolicy)
    register_policy("alpha", _OtherPolicy)
    out = render_registry()
    # alpha appears before zeta
    assert out.index("alpha") < out.index("zeta")


# --- E2E --------------------------------------------------------------------


def test_e2e_replay_compatible_roundtrip():
    """A registered policy round-trips byte-identical through snapshot."""
    register_policy("example", _ExamplePolicy)
    original = _ExamplePolicy(threshold=0.123, mode=_Mode.B, label="replay")
    snap1 = snapshot_to_dict(original)
    rebuilt = snapshot_from_dict("example", snap1)
    snap2 = snapshot_to_dict(rebuilt)
    assert snap1 == snap2
    assert rebuilt == original


def test_e2e_use_real_existing_policy_via_registry():
    """Smoke test that an existing project Policy can be registered + roundtripped."""
    from halal_trader.halal.maysir_screen import MaysirPolicy

    register_policy("maysir", MaysirPolicy)
    p = MaysirPolicy()
    snap = snapshot_to_dict(p)
    p2 = snapshot_from_dict("maysir", snap)
    assert p == p2
