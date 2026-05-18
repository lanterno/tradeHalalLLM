"""Tests for `ml/feature_store.py` (feature schema + migration).

Pins the dtype matching rules (including the bool/int subtype
asymmetry), the schema duplicate-name rejection, every migration
rule (Rename / Drop / AddDefault / Cast) including the lossy-cast
refusals, the migrate() driver's plan-lookup contract, and the
validate() / render output.
"""

from __future__ import annotations

import pytest

from halal_trader.ml.feature_store import (
    AddDefault,
    Cast,
    Drop,
    FeatureDType,
    FeatureSchema,
    FeatureSpec,
    MigrationPlan,
    Rename,
    ValidationOutcome,
    migrate,
    render_validation,
    validate,
)
from halal_trader.ml.registry import Semver


def _spec(
    name: str = "rsi_14",
    dtype: FeatureDType = FeatureDType.FLOAT,
    *,
    required: bool = True,
) -> FeatureSpec:
    return FeatureSpec(name=name, dtype=dtype, required=required)


def _schema(
    *,
    name: str = "crypto_features",
    version: Semver = Semver(1, 0, 0),
    features: tuple[FeatureSpec, ...] = (),
) -> FeatureSchema:
    return FeatureSchema(name=name, version=version, features=features)


# ── dtype matching ───────────────────────────────────────


def test_float_dtype_accepts_float_and_int():
    """Pin: int auto-promotes to float — common in numpy interop
    where `arr[0]` may return either."""
    s = _schema(features=(_spec("x", FeatureDType.FLOAT),))
    assert validate({"x": 1.5}, s).passed
    assert validate({"x": 1}, s).passed


def test_int_dtype_rejects_float():
    """Pin: an explicit INT can't accept a float — refusing the
    silent truncation."""
    s = _schema(features=(_spec("x", FeatureDType.INT),))
    out = validate({"x": 1.5}, s)
    assert not out.passed


def test_int_dtype_rejects_bool():
    """Pin: bool is a subtype of int in Python, but a True passed
    where INT was expected is almost certainly a bug. Asymmetric
    with the FLOAT case on purpose."""
    s = _schema(features=(_spec("x", FeatureDType.INT),))
    out = validate({"x": True}, s)
    assert not out.passed


def test_bool_dtype_accepts_only_bool():
    s = _schema(features=(_spec("x", FeatureDType.BOOL),))
    assert validate({"x": True}, s).passed
    assert validate({"x": False}, s).passed
    assert not validate({"x": 1}, s).passed
    assert not validate({"x": 0}, s).passed


def test_str_dtype_accepts_only_str():
    s = _schema(features=(_spec("regime", FeatureDType.STR),))
    assert validate({"regime": "uptrend"}, s).passed
    assert not validate({"regime": 1}, s).passed


# ── FeatureSchema validation ─────────────────────────────


def test_schema_rejects_duplicate_feature_names():
    """Pin: a corrupt config with `rsi_14` listed twice would
    silently drop one definition under a dict-of-features model;
    refuse construction instead."""
    with pytest.raises(ValueError, match="duplicate feature"):
        _schema(features=(_spec("x"), _spec("x")))


def test_schema_feature_by_name_finds_and_misses():
    s = _schema(features=(_spec("rsi_14"),))
    assert s.feature_by_name("rsi_14") is not None
    assert s.feature_by_name("missing") is None


def test_schema_names_returns_set_of_features():
    s = _schema(features=(_spec("rsi_14"), _spec("macd")))
    assert s.names() == {"rsi_14", "macd"}


# ── Rename rule ──────────────────────────────────────────


def test_rename_relabels_a_feature_in_place():
    payload = {"rsi_14": 35.0, "macd": 0.001}
    out = Rename("rsi_14", "rsi_period_14").apply(payload)
    assert "rsi_14" not in out
    assert out["rsi_period_14"] == 35.0
    assert out["macd"] == 0.001


def test_rename_is_no_op_when_source_missing():
    """Pin: a previous migration may already have removed the
    source. Rename must not crash on absent source — the chain
    should compose."""
    out = Rename("missing", "still_missing").apply({"x": 1.0})
    assert out == {"x": 1.0}


def test_rename_rejects_collision_with_existing_target():
    """Pin: silently overwriting an existing target value would
    lose data. Refuse instead."""
    with pytest.raises(ValueError, match="collision"):
        Rename("a", "b").apply({"a": 1, "b": 2})


def test_rename_rejects_no_op():
    """Pin: from_name == to_name is a config bug, not a no-op."""
    with pytest.raises(ValueError, match="no-op"):
        Rename("x", "x").apply({"x": 1})


def test_rename_does_not_mutate_input():
    payload = {"rsi_14": 35.0}
    Rename("rsi_14", "rsi_period_14").apply(payload)
    assert payload == {"rsi_14": 35.0}


# ── Drop rule ────────────────────────────────────────────


def test_drop_removes_a_feature():
    out = Drop("stale").apply({"stale": 1, "keep": 2})
    assert "stale" not in out
    assert out["keep"] == 2


def test_drop_is_no_op_when_feature_absent():
    """Pin: idempotent — drop of an already-removed feature is
    fine. Migrations chain; a second drop must not crash."""
    out = Drop("missing").apply({"x": 1})
    assert out == {"x": 1}


# ── AddDefault rule ──────────────────────────────────────


def test_add_default_inserts_when_missing():
    out = AddDefault("regime", "neutral").apply({"x": 1})
    assert out["regime"] == "neutral"
    assert out["x"] == 1


def test_add_default_skips_when_present():
    """Pin: never overwrite an existing value — operator's
    intent in adding a default is "fill in if missing", not
    "force this value"."""
    out = AddDefault("regime", "neutral").apply({"regime": "uptrend"})
    assert out["regime"] == "uptrend"


# ── Cast rule ────────────────────────────────────────────


def test_cast_float_to_int_when_integer_valued():
    """Pin: 1.0 → 1 is fine (no information loss)."""
    out = Cast("x", FeatureDType.INT).apply({"x": 5.0})
    assert out["x"] == 5
    assert isinstance(out["x"], int)


def test_cast_float_to_int_rejects_lossy_value():
    """Pin: refuse 1.7 → 1 silent rounding."""
    with pytest.raises(ValueError, match="lossy"):
        Cast("x", FeatureDType.INT).apply({"x": 1.7})


def test_cast_int_to_float_succeeds():
    out = Cast("x", FeatureDType.FLOAT).apply({"x": 5})
    assert out["x"] == 5.0
    assert isinstance(out["x"], float)


def test_cast_to_bool_accepts_zero_one_and_bool():
    assert Cast("x", FeatureDType.BOOL).apply({"x": 1})["x"] is True
    assert Cast("x", FeatureDType.BOOL).apply({"x": 0})["x"] is False
    assert Cast("x", FeatureDType.BOOL).apply({"x": True})["x"] is True


def test_cast_to_bool_rejects_ambiguous_values():
    """Pin: refuse `5 → True` and `"yes" → True`. A value that
    isn't already 0/1/bool is ambiguous; surface immediately."""
    with pytest.raises(ValueError, match="ambiguous"):
        Cast("x", FeatureDType.BOOL).apply({"x": 5})
    with pytest.raises(ValueError, match="ambiguous"):
        Cast("x", FeatureDType.BOOL).apply({"x": "yes"})


def test_cast_to_str_works_on_anything():
    out = Cast("x", FeatureDType.STR).apply({"x": 1.5})
    assert out["x"] == "1.5"


def test_cast_failure_wraps_original_error():
    with pytest.raises(ValueError, match="cast of"):
        Cast("x", FeatureDType.FLOAT).apply({"x": "not a number"})


def test_cast_no_op_when_feature_absent():
    out = Cast("missing", FeatureDType.FLOAT).apply({"x": 1})
    assert out == {"x": 1}


# ── MigrationPlan ────────────────────────────────────────


def test_plan_applies_rules_in_order():
    """Pin: order matters. Drop → Rename can't run as Rename →
    Drop without changing semantics."""
    plan = MigrationPlan(
        schema_name="s",
        from_version=Semver(1, 0, 0),
        to_version=Semver(1, 1, 0),
        rules=(
            Rename("a", "b"),
            Drop("c"),
            AddDefault("d", 0.0),
        ),
    )
    out = plan.apply({"a": 1, "c": 2, "x": 3})
    assert out == {"b": 1, "d": 0.0, "x": 3}


def test_plan_with_no_rules_is_a_copy():
    """Pin no-mutation invariant on the empty plan path."""
    plan = MigrationPlan(
        schema_name="s",
        from_version=Semver(1, 0, 0),
        to_version=Semver(1, 0, 1),
    )
    payload = {"x": 1}
    out = plan.apply(payload)
    assert out == payload
    assert out is not payload


# ── migrate driver ───────────────────────────────────────


def test_migrate_returns_copy_when_versions_match():
    """Pin: equal versions skip migration but still copy the
    payload — caller mustn't accidentally share the dict."""
    schema = _schema(version=Semver(1, 0, 0))
    payload = {"x": 1}
    out = migrate(payload, from_schema=schema, to_schema=schema)
    assert out == payload
    assert out is not payload


def test_migrate_applies_matching_plan():
    s_from = _schema(version=Semver(1, 0, 0))
    s_to = _schema(version=Semver(1, 1, 0))
    plan = MigrationPlan(
        schema_name="crypto_features",
        from_version=Semver(1, 0, 0),
        to_version=Semver(1, 1, 0),
        rules=(Rename("rsi_14", "rsi_period_14"),),
    )
    out = migrate({"rsi_14": 35.0}, from_schema=s_from, to_schema=s_to, plans=[plan])
    assert out == {"rsi_period_14": 35.0}


def test_migrate_rejects_schema_name_mismatch():
    """Pin: migrating across schema names is nonsense and
    surfaces immediately."""
    a = _schema(name="rsi_features")
    b = _schema(name="sentiment_features")
    with pytest.raises(ValueError, match="schema name"):
        migrate({}, from_schema=a, to_schema=b)


def test_migrate_rejects_missing_plan():
    """Pin: no matching plan = explicit error, not silent
    "guess what to do"."""
    with pytest.raises(ValueError, match="no migration plan"):
        migrate(
            {},
            from_schema=_schema(version=Semver(1, 0, 0)),
            to_schema=_schema(version=Semver(1, 1, 0)),
            plans=[],
        )


def test_migrate_rejects_ambiguous_plans():
    """Pin: two plans for the same (name, from, to) is a config
    bug. Surface rather than picking the first."""
    plan_a = MigrationPlan(
        schema_name="crypto_features",
        from_version=Semver(1, 0, 0),
        to_version=Semver(1, 1, 0),
    )
    plan_b = MigrationPlan(
        schema_name="crypto_features",
        from_version=Semver(1, 0, 0),
        to_version=Semver(1, 1, 0),
        rules=(Drop("x"),),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate(
            {},
            from_schema=_schema(version=Semver(1, 0, 0)),
            to_schema=_schema(version=Semver(1, 1, 0)),
            plans=[plan_a, plan_b],
        )


# ── validate ─────────────────────────────────────────────


def test_validate_empty_schema_passes_anything():
    out = validate({"x": 1}, _schema())
    assert out.passed
    assert out.extras == ["x"]


def test_validate_required_feature_missing_fails():
    s = _schema(features=(_spec("rsi_14"),))
    out = validate({}, s)
    assert not out.passed
    assert any("rsi_14" in v for v in out.violations)


def test_validate_optional_feature_missing_passes():
    s = _schema(features=(_spec("optional_x", required=False),))
    out = validate({}, s)
    assert out.passed


def test_validate_optional_feature_with_wrong_dtype_still_fails():
    """Pin: a feature that's present must satisfy its dtype, even
    if it's optional. The optional flag controls *presence*, not
    type laxity."""
    s = _schema(features=(_spec("x", FeatureDType.INT, required=False),))
    out = validate({"x": "not an int"}, s)
    assert not out.passed


def test_validate_extras_listed_separately_not_failing():
    s = _schema(features=(_spec("rsi_14"),))
    out = validate({"rsi_14": 35.0, "stale_feature": 1}, s)
    assert out.passed
    assert "stale_feature" in out.extras


# ── render ───────────────────────────────────────────────


def test_render_pass_payload():
    out = ValidationOutcome(passed=True)
    assert "PASS" in render_validation(out)


def test_render_fail_payload_lists_violations():
    out = ValidationOutcome(
        passed=False,
        violations=["missing required feature 'rsi_14'"],
    )
    text = render_validation(out)
    assert "FAIL" in text
    assert "rsi_14" in text


def test_render_fail_payload_includes_violation_count():
    out = ValidationOutcome(
        passed=False,
        violations=["a", "b", "c"],
    )
    text = render_validation(out)
    assert "3 violations" in text


def test_render_lists_extras_section():
    out = ValidationOutcome(passed=True, extras=["stale_a", "stale_b"])
    text = render_validation(out)
    assert "Extras" in text
    assert "stale_a" in text
