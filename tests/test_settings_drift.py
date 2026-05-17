"""Tests for `halal_trader.ops.settings_drift`.

Auxiliary primitive for operator settings audit. Covers: numeric
drift magnitude bands, boolean / string drift, bounds detection,
no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.ops.settings_drift import (
    DriftEntry,
    DriftMagnitude,
    DriftReport,
    NumericBounds,
    SettingSpec,
    build_drift_report,
    detect_drift_for_setting,
    filter_drifted,
    render_entry,
    render_report,
)

# --------------------------- Enum string pins --------------------------------


def test_drift_magnitude_string_values_pinned() -> None:
    assert DriftMagnitude.NONE.value == "none"
    assert DriftMagnitude.TINY.value == "tiny"
    assert DriftMagnitude.MODERATE.value == "moderate"
    assert DriftMagnitude.SIGNIFICANT.value == "significant"
    assert DriftMagnitude.EXTREME.value == "extreme"


# --------------------------- NumericBounds -----------------------------------


def test_bounds_basic() -> None:
    b = NumericBounds(min_value=0.0, max_value=1.0)
    assert b.contains(0.5)
    assert b.contains(0.0)
    assert b.contains(1.0)
    assert not b.contains(-0.1)
    assert not b.contains(1.1)


def test_bounds_rejects_max_at_min() -> None:
    with pytest.raises(ValueError, match="max_value"):
        NumericBounds(min_value=1.0, max_value=1.0)


def test_bounds_rejects_max_below_min() -> None:
    with pytest.raises(ValueError, match="max_value"):
        NumericBounds(min_value=2.0, max_value=1.0)


def test_bounds_is_frozen() -> None:
    b = NumericBounds(min_value=0.0, max_value=1.0)
    with pytest.raises(FrozenInstanceError):
        b.min_value = 0.5  # type: ignore[misc]


# --------------------------- SettingSpec -------------------------------------


def test_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        SettingSpec(name="", default_value=1.0, description="x")


def test_spec_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        SettingSpec(name="x", default_value=1.0, description="")


def test_spec_rejects_bounds_on_string() -> None:
    """Pin: bounds only apply to numeric defaults."""

    with pytest.raises(ValueError, match="bounds"):
        SettingSpec(
            name="x",
            default_value="hello",
            description="x",
            bounds=NumericBounds(min_value=0.0, max_value=1.0),
        )


def test_spec_rejects_default_out_of_bounds() -> None:
    """Pin: default itself must be within declared bounds."""

    with pytest.raises(ValueError, match="bounds"):
        SettingSpec(
            name="x",
            default_value=2.0,
            description="x",
            bounds=NumericBounds(min_value=0.0, max_value=1.0),
        )


def test_spec_is_frozen() -> None:
    s = SettingSpec(name="x", default_value=1.0, description="x")
    with pytest.raises(FrozenInstanceError):
        s.default_value = 2.0  # type: ignore[misc]


# --------------------------- detect_drift: boolean ---------------------------


def test_boolean_no_drift() -> None:
    spec = SettingSpec(name="enabled", default_value=True, description="x")
    entry = detect_drift_for_setting(spec, True)
    assert entry.magnitude is DriftMagnitude.NONE


def test_boolean_flip_is_moderate() -> None:
    """Pin: any boolean flip is at least MODERATE."""

    spec = SettingSpec(name="enabled", default_value=True, description="x")
    entry = detect_drift_for_setting(spec, False)
    assert entry.magnitude is DriftMagnitude.MODERATE


def test_boolean_with_non_bool_current_raises() -> None:
    spec = SettingSpec(name="enabled", default_value=True, description="x")
    with pytest.raises(TypeError, match="bool"):
        detect_drift_for_setting(spec, 1)


# --------------------------- detect_drift: string ---------------------------


def test_string_no_drift() -> None:
    spec = SettingSpec(name="provider", default_value="ollama", description="x")
    entry = detect_drift_for_setting(spec, "ollama")
    assert entry.magnitude is DriftMagnitude.NONE


def test_string_change_is_moderate() -> None:
    """Pin: non-empty string change is MODERATE."""

    spec = SettingSpec(name="provider", default_value="ollama", description="x")
    entry = detect_drift_for_setting(spec, "openai")
    assert entry.magnitude is DriftMagnitude.MODERATE


def test_empty_to_set_is_significant() -> None:
    """Pin: empty default → set value is SIGNIFICANT (likely a credential)."""

    spec = SettingSpec(name="api_key", default_value="", description="x")
    entry = detect_drift_for_setting(spec, "actual-key-123")
    assert entry.magnitude is DriftMagnitude.SIGNIFICANT


def test_set_to_empty_is_significant() -> None:
    spec = SettingSpec(name="api_key", default_value="default-value", description="x")
    entry = detect_drift_for_setting(spec, "")
    assert entry.magnitude is DriftMagnitude.SIGNIFICANT


def test_string_with_non_str_current_raises() -> None:
    spec = SettingSpec(name="provider", default_value="ollama", description="x")
    with pytest.raises(TypeError, match="str"):
        detect_drift_for_setting(spec, 42)


# --------------------------- detect_drift: numeric --------------------------


def test_numeric_no_drift() -> None:
    spec = SettingSpec(name="lr", default_value=0.001, description="x")
    entry = detect_drift_for_setting(spec, 0.001)
    assert entry.magnitude is DriftMagnitude.NONE


def test_numeric_tiny_drift_at_5pct() -> None:
    """Pin: 5% drift below 10% threshold → TINY."""

    spec = SettingSpec(name="lr", default_value=1.0, description="x")
    entry = detect_drift_for_setting(spec, 1.05)
    assert entry.magnitude is DriftMagnitude.TINY


def test_numeric_moderate_drift_at_30pct() -> None:
    """Pin: 30% drift in [10%, 50%] → MODERATE."""

    spec = SettingSpec(name="lr", default_value=1.0, description="x")
    entry = detect_drift_for_setting(spec, 1.3)
    assert entry.magnitude is DriftMagnitude.MODERATE


def test_numeric_significant_drift_at_75pct() -> None:
    """Pin: 75% drift in (50%, 100%] → SIGNIFICANT."""

    spec = SettingSpec(name="lr", default_value=1.0, description="x")
    entry = detect_drift_for_setting(spec, 1.75)
    assert entry.magnitude is DriftMagnitude.SIGNIFICANT


def test_numeric_extreme_drift_above_100pct() -> None:
    """Pin: >100% drift → EXTREME."""

    spec = SettingSpec(name="lr", default_value=1.0, description="x")
    entry = detect_drift_for_setting(spec, 3.0)  # 200% drift
    assert entry.magnitude is DriftMagnitude.EXTREME


def test_numeric_drift_negative_direction() -> None:
    """Pin: drift detection is direction-agnostic."""

    spec = SettingSpec(name="lr", default_value=1.0, description="x")
    entry = detect_drift_for_setting(spec, 0.3)  # 70% drift below
    assert entry.magnitude is DriftMagnitude.SIGNIFICANT


def test_numeric_default_zero_treated_specially() -> None:
    """Pin: default=0 + non-zero current uses absolute scale.

    Since rel_drift would divide by zero, fall back to absolute
    magnitude bands.
    """

    spec = SettingSpec(name="x", default_value=0, description="x")

    # Tiny absolute (< 0.1)
    entry = detect_drift_for_setting(spec, 0.05)
    assert entry.magnitude is DriftMagnitude.TINY

    # Moderate (0.1 to 1.0)
    entry = detect_drift_for_setting(spec, 0.5)
    assert entry.magnitude is DriftMagnitude.MODERATE

    # Significant (1.0 to 10.0)
    entry = detect_drift_for_setting(spec, 5.0)
    assert entry.magnitude is DriftMagnitude.SIGNIFICANT

    # Extreme (>= 10.0)
    entry = detect_drift_for_setting(spec, 100.0)
    assert entry.magnitude is DriftMagnitude.EXTREME


def test_numeric_with_int_default() -> None:
    """Pin: int defaults work the same as float."""

    spec = SettingSpec(name="batch_size", default_value=64, description="x")
    entry = detect_drift_for_setting(spec, 128)
    # 64 → 128 = 100% drift → EXTREME boundary; >100% would be EXTREME,
    # exactly 100% (rel_drift > 1.0 is False at 1.0 exactly) is SIGNIFICANT
    assert entry.magnitude is DriftMagnitude.SIGNIFICANT


def test_numeric_with_non_numeric_current_raises() -> None:
    spec = SettingSpec(name="lr", default_value=0.001, description="x")
    with pytest.raises(TypeError, match="numeric"):
        detect_drift_for_setting(spec, "fast")


def test_numeric_bool_not_treated_as_int() -> None:
    """Pin: True/False shouldn't be interpreted as 1/0 numeric values
    when the default is numeric (would be confusing)."""

    spec = SettingSpec(name="lr", default_value=0.001, description="x")
    with pytest.raises(TypeError, match="numeric"):
        detect_drift_for_setting(spec, True)


# --------------------------- detect_drift: out-of-bounds --------------------


def test_out_of_bounds_high_is_extreme() -> None:
    """Pin: out-of-bounds always classifies as EXTREME."""

    spec = SettingSpec(
        name="max_position_pct",
        default_value=0.05,
        description="x",
        bounds=NumericBounds(min_value=0.001, max_value=0.5),
    )
    entry = detect_drift_for_setting(spec, 5.0)  # 500% — out of bounds
    assert entry.magnitude is DriftMagnitude.EXTREME
    assert entry.out_of_bounds is True


def test_out_of_bounds_low_is_extreme() -> None:
    spec = SettingSpec(
        name="lr",
        default_value=0.01,
        description="x",
        bounds=NumericBounds(min_value=0.0001, max_value=0.1),
    )
    entry = detect_drift_for_setting(spec, 0.00001)  # below min
    assert entry.magnitude is DriftMagnitude.EXTREME
    assert entry.out_of_bounds is True


def test_within_bounds_uses_normal_magnitude() -> None:
    """Pin: within bounds, magnitude is computed normally."""

    spec = SettingSpec(
        name="lr",
        default_value=0.01,
        description="x",
        bounds=NumericBounds(min_value=0.0001, max_value=0.1),
    )
    entry = detect_drift_for_setting(spec, 0.011)  # 10% drift, in bounds
    assert entry.out_of_bounds is False
    assert entry.magnitude in (DriftMagnitude.TINY, DriftMagnitude.MODERATE)


def test_at_bounds_boundary_inclusive() -> None:
    """Pin: bounds are inclusive [min, max]."""

    spec = SettingSpec(
        name="x",
        default_value=0.5,
        description="x",
        bounds=NumericBounds(min_value=0.0, max_value=1.0),
    )
    entry = detect_drift_for_setting(spec, 1.0)
    assert entry.out_of_bounds is False


# --------------------------- DriftEntry / DriftReport ----------------------


def test_drift_entry_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        DriftEntry(
            name="",
            default_value=1.0,
            current_value=1.0,
            magnitude=DriftMagnitude.NONE,
            out_of_bounds=False,
            is_secret=False,
        )


def test_drift_entry_is_frozen() -> None:
    entry = DriftEntry(
        name="x",
        default_value=1.0,
        current_value=1.0,
        magnitude=DriftMagnitude.NONE,
        out_of_bounds=False,
        is_secret=False,
    )
    with pytest.raises(FrozenInstanceError):
        entry.magnitude = DriftMagnitude.EXTREME  # type: ignore[misc]


def test_drift_report_rejects_drifted_above_total() -> None:
    """Pin: drifted_count can't exceed total_settings."""

    with pytest.raises(ValueError, match="drifted_count"):
        DriftReport(
            entries=(),
            total_settings=5,
            drifted_count=10,
            out_of_bounds_count=0,
        )


def test_drift_report_rejects_out_of_bounds_above_drifted() -> None:
    """Pin: out_of_bounds is a subset of drifted."""

    with pytest.raises(ValueError, match="out_of_bounds_count"):
        DriftReport(
            entries=(),
            total_settings=10,
            drifted_count=2,
            out_of_bounds_count=5,
        )


# --------------------------- build_drift_report -----------------------------


def test_build_report_no_drift() -> None:
    catalogue = [
        SettingSpec(name="lr", default_value=0.001, description="x"),
        SettingSpec(name="enabled", default_value=True, description="x"),
    ]
    current = {"lr": 0.001, "enabled": True}
    report = build_drift_report(catalogue, current)
    assert report.total_settings == 2
    assert report.drifted_count == 0
    assert report.out_of_bounds_count == 0


def test_build_report_some_drift() -> None:
    catalogue = [
        SettingSpec(name="lr", default_value=0.001, description="x"),
        SettingSpec(name="enabled", default_value=True, description="x"),
        SettingSpec(name="provider", default_value="ollama", description="x"),
    ]
    current = {
        "lr": 0.005,  # extreme drift (400%)
        "enabled": False,  # moderate (boolean flip)
        "provider": "ollama",  # no drift
    }
    report = build_drift_report(catalogue, current)
    assert report.total_settings == 3
    assert report.drifted_count == 2


def test_build_report_missing_setting_raises() -> None:
    """Pin: every catalogued setting must be in current_values."""

    catalogue = [SettingSpec(name="lr", default_value=0.001, description="x")]
    with pytest.raises(KeyError, match="lr"):
        build_drift_report(catalogue, {})


def test_build_report_extra_keys_silently_ignored() -> None:
    """Pin: operator-defined extras not in catalogue are ignored."""

    catalogue = [SettingSpec(name="lr", default_value=0.001, description="x")]
    current = {"lr": 0.001, "MY_CUSTOM_VAR": "irrelevant"}
    report = build_drift_report(catalogue, current)
    assert report.total_settings == 1
    # MY_CUSTOM_VAR doesn't appear in entries
    names = {e.name for e in report.entries}
    assert "MY_CUSTOM_VAR" not in names


def test_build_report_entries_sorted_by_name() -> None:
    """Pin: deterministic display order."""

    catalogue = [
        SettingSpec(name="zulu", default_value=1, description="x"),
        SettingSpec(name="alpha", default_value=1, description="x"),
        SettingSpec(name="mike", default_value=1, description="x"),
    ]
    current = {"zulu": 1, "alpha": 1, "mike": 1}
    report = build_drift_report(catalogue, current)
    names = [e.name for e in report.entries]
    assert names == ["alpha", "mike", "zulu"]


def test_build_report_out_of_bounds_count() -> None:
    catalogue = [
        SettingSpec(
            name="lr",
            default_value=0.01,
            description="x",
            bounds=NumericBounds(min_value=0.0001, max_value=0.1),
        ),
    ]
    current = {"lr": 1.0}  # out of bounds
    report = build_drift_report(catalogue, current)
    assert report.out_of_bounds_count == 1


def test_build_report_is_deterministic() -> None:
    catalogue = [SettingSpec(name="lr", default_value=0.001, description="x")]
    current = {"lr": 0.005}
    a = build_drift_report(catalogue, current)
    b = build_drift_report(catalogue, current)
    assert a == b


# --------------------------- filter_drifted ----------------------------------


def test_filter_drifted_default_includes_tiny_and_above() -> None:
    catalogue = [
        SettingSpec(name="a", default_value=1.0, description="x"),
        SettingSpec(name="b", default_value=1.0, description="x"),
        SettingSpec(name="c", default_value=1.0, description="x"),
    ]
    current = {
        "a": 1.0,  # NONE
        "b": 1.05,  # TINY
        "c": 3.0,  # EXTREME
    }
    report = build_drift_report(catalogue, current)
    drifted = filter_drifted(report)
    names = {e.name for e in drifted}
    assert names == {"b", "c"}


def test_filter_drifted_minimum_significant() -> None:
    """Pin: filtering at SIGNIFICANT excludes TINY + MODERATE."""

    catalogue = [
        SettingSpec(name="a", default_value=1.0, description="x"),
        SettingSpec(name="b", default_value=1.0, description="x"),
        SettingSpec(name="c", default_value=1.0, description="x"),
    ]
    current = {
        "a": 1.05,  # TINY
        "b": 1.3,  # MODERATE
        "c": 1.75,  # SIGNIFICANT
    }
    report = build_drift_report(catalogue, current)
    drifted = filter_drifted(report, minimum=DriftMagnitude.SIGNIFICANT)
    names = {e.name for e in drifted}
    assert names == {"c"}


# --------------------------- render ------------------------------------------


def test_render_entry_includes_emoji() -> None:
    catalogue = [SettingSpec(name="lr", default_value=0.001, description="x")]
    report = build_drift_report(catalogue, {"lr": 1.0})
    entry = report.entries[0]
    out = render_entry(entry)
    assert "🔴" in out  # extreme magnitude
    assert "lr" in out


def test_render_entry_secret_uses_placeholder() -> None:
    """Pin: secret renders as <secret>, not the actual value."""

    catalogue = [
        SettingSpec(
            name="api_key",
            default_value="",
            description="x",
            is_secret=True,
        )
    ]
    report = build_drift_report(catalogue, {"api_key": "real-secret-value-123"})
    entry = report.entries[0]
    out = render_entry(entry)
    assert "<secret>" in out
    assert "real-secret-value-123" not in out


def test_render_entry_out_of_bounds_marker() -> None:
    catalogue = [
        SettingSpec(
            name="lr",
            default_value=0.01,
            description="x",
            bounds=NumericBounds(min_value=0.0001, max_value=0.1),
        )
    ]
    report = build_drift_report(catalogue, {"lr": 5.0})
    entry = report.entries[0]
    out = render_entry(entry)
    assert "OUT OF BOUNDS" in out


def test_render_entry_truncates_long_strings() -> None:
    """Pin: long string values are truncated for display."""

    catalogue = [SettingSpec(name="prompt", default_value="default", description="x")]
    long_value = "x" * 100
    report = build_drift_report(catalogue, {"prompt": long_value})
    entry = report.entries[0]
    out = render_entry(entry)
    # Truncated to 37 chars + "..." somewhere in the output
    assert "..." in out
    # Shouldn't contain the full 100-char value
    assert long_value not in out


def test_render_report_no_drift() -> None:
    catalogue = [SettingSpec(name="lr", default_value=0.001, description="x")]
    report = build_drift_report(catalogue, {"lr": 0.001})
    out = render_report(report)
    assert "no drift detected" in out
    assert "✅" in out


def test_render_report_with_drift() -> None:
    catalogue = [
        SettingSpec(name="lr", default_value=0.001, description="x"),
        SettingSpec(name="enabled", default_value=True, description="x"),
    ]
    report = build_drift_report(catalogue, {"lr": 0.005, "enabled": False})
    out = render_report(report)
    assert "drifted: 2" in out
    assert "lr" in out
    assert "enabled" in out


def test_render_no_secret_leak_in_full_report() -> None:
    """Pin: no secret value leaks even in the full report."""

    catalogue = [
        SettingSpec(
            name="broker_api_key",
            default_value="",
            description="x",
            is_secret=True,
        ),
        SettingSpec(
            name="lr",
            default_value=0.001,
            description="x",
        ),
    ]
    report = build_drift_report(
        catalogue,
        {"broker_api_key": "sk_live_VERYSENSITIVE12345", "lr": 0.005},
    )
    out = render_report(report)
    assert "sk_live" not in out
    assert "VERYSENSITIVE" not in out
    assert "<secret>" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_realistic_settings_audit() -> None:
    """Real-world: operator runs audit, sees their tuned knobs."""

    catalogue = [
        SettingSpec(
            name="max_position_pct",
            default_value=0.05,
            description="Per-position size cap as fraction of equity",
            bounds=NumericBounds(min_value=0.001, max_value=0.5),
        ),
        SettingSpec(
            name="stop_loss_pct",
            default_value=0.02,
            description="Default stop-loss distance",
            bounds=NumericBounds(min_value=0.005, max_value=0.10),
        ),
        SettingSpec(
            name="halt_engaged",
            default_value=False,
            description="Operator-engaged kill-switch",
        ),
        SettingSpec(
            name="llm_provider",
            default_value="ollama",
            description="Active LLM provider",
        ),
        SettingSpec(
            name="broker_api_key",
            default_value="",
            description="Broker API key (vault ref)",
            is_secret=True,
        ),
    ]
    current = {
        "max_position_pct": 0.10,  # 100% drift = EXTREME
        "stop_loss_pct": 0.02,  # default
        "halt_engaged": True,  # MODERATE flip
        "llm_provider": "anthropic",  # MODERATE
        "broker_api_key": "real-key-123",  # SIGNIFICANT (empty → set)
    }
    report = build_drift_report(catalogue, current)
    assert report.total_settings == 5
    assert report.drifted_count == 4  # all but stop_loss_pct
    out = render_report(report)
    # Pin: broker_api_key shows as <secret>, not the real key
    assert "real-key-123" not in out
    assert "<secret>" in out


def test_e2e_misconfigured_max_position_caught() -> None:
    """Pin: max_position_pct=5.0 (operator typo for 500%) caught as EXTREME +
    out-of-bounds."""

    catalogue = [
        SettingSpec(
            name="max_position_pct",
            default_value=0.05,
            description="x",
            bounds=NumericBounds(min_value=0.001, max_value=0.5),
        )
    ]
    report = build_drift_report(catalogue, {"max_position_pct": 5.0})
    entry = report.entries[0]
    assert entry.magnitude is DriftMagnitude.EXTREME
    assert entry.out_of_bounds is True


def test_e2e_replay_consistency() -> None:
    catalogue = [SettingSpec(name="lr", default_value=0.001, description="x")]
    current = {"lr": 0.005}
    a = build_drift_report(catalogue, current)
    b = build_drift_report(catalogue, current)
    assert a == b
