"""Tests for `halal_trader.ops.deprecation`.

Auxiliary primitive complementing Wave 9.F API reference. Covers:
deprecation lifecycle, sunset timeline enforcement, warning
emission, removal-readiness gate.
"""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.ops.deprecation import (
    DEFAULT_POLICY,
    DeprecatedSymbol,
    DeprecationPolicy,
    DeprecationStage,
    StageTransitionError,
    SymbolRemovedError,
    advance_stage,
    announce_deprecation,
    assert_not_removed,
    emit_warning_if_needed,
    filter_overdue,
    is_overdue_for_advancement,
    render_record,
    scheduled_deprecated_at,
    scheduled_removal_at,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_stage_string_values_pinned() -> None:
    assert DeprecationStage.ANNOUNCED.value == "announced"
    assert DeprecationStage.DEPRECATED.value == "deprecated"
    assert DeprecationStage.REMOVED.value == "removed"


# --------------------------- DeprecationPolicy -------------------------------


def test_default_policy() -> None:
    """Pin: 60-day announce + 90-day deprecated = 150-day total sunset."""

    assert DEFAULT_POLICY.announce_window == timedelta(days=60)
    assert DEFAULT_POLICY.deprecated_window == timedelta(days=90)


def test_policy_rejects_short_announce_window() -> None:
    """Pin: announce window must be at least 30 days."""

    with pytest.raises(ValueError, match="announce_window"):
        DeprecationPolicy(announce_window=timedelta(days=15))


def test_policy_accepts_30_day_announce() -> None:
    p = DeprecationPolicy(
        announce_window=timedelta(days=30),
        deprecated_window=timedelta(days=60),
    )
    assert p.announce_window == timedelta(days=30)


def test_policy_rejects_short_deprecated_window() -> None:
    with pytest.raises(ValueError, match="deprecated_window"):
        DeprecationPolicy(deprecated_window=timedelta(days=30))


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.announce_window = timedelta(days=1)  # type: ignore[misc]


# --------------------------- DeprecatedSymbol --------------------------------


def test_symbol_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="symbol"):
        DeprecatedSymbol(
            symbol="",
            announced_at=T0,
            stage=DeprecationStage.ANNOUNCED,
        )


def test_symbol_rejects_naive_announced_at() -> None:
    with pytest.raises(ValueError, match="announced_at"):
        DeprecatedSymbol(
            symbol="x",
            announced_at=datetime(2026, 5, 1),
            stage=DeprecationStage.ANNOUNCED,
        )


def test_symbol_is_frozen() -> None:
    s = announce_deprecation(symbol="x", now=T0)
    with pytest.raises(FrozenInstanceError):
        s.symbol = "y"  # type: ignore[misc]


# --------------------------- announce_deprecation ----------------------------


def test_announce_basic() -> None:
    record = announce_deprecation(symbol="old_method", now=T0)
    assert record.stage is DeprecationStage.ANNOUNCED
    assert record.replacement == ""
    assert record.migration_url == ""


def test_announce_with_replacement() -> None:
    record = announce_deprecation(
        symbol="old_method",
        now=T0,
        replacement="new_method",
        migration_url="https://docs.example.com/migrate",
        reason="Cleaner API",
    )
    assert record.replacement == "new_method"
    assert record.migration_url == "https://docs.example.com/migrate"
    assert record.reason == "Cleaner API"


def test_announce_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol"):
        announce_deprecation(symbol="", now=T0)


def test_announce_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        announce_deprecation(symbol="x", now=datetime(2026, 5, 1))


# --------------------------- advance_stage -----------------------------------


def test_advance_announced_to_deprecated() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0 + timedelta(days=60))
    assert record.stage is DeprecationStage.DEPRECATED


def test_advance_full_lifecycle() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0 + timedelta(days=60))
    record = advance_stage(record, now=T0 + timedelta(days=150))
    assert record.stage is DeprecationStage.REMOVED


def test_advance_from_removed_rejected() -> None:
    """Pin: REMOVED is terminal."""

    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0)
    record = advance_stage(record, now=T0)
    with pytest.raises(StageTransitionError):
        advance_stage(record, now=T0)


def test_advance_naive_now_rejected() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    with pytest.raises(ValueError, match="now"):
        advance_stage(record, now=datetime(2026, 5, 1))


def test_advance_preserves_metadata() -> None:
    """Pin: replacement / migration_url / reason carry forward."""

    record = announce_deprecation(
        symbol="x",
        now=T0,
        replacement="new_x",
        migration_url="https://migrate",
        reason="API cleanup",
    )
    record = advance_stage(record, now=T0)
    assert record.replacement == "new_x"
    assert record.migration_url == "https://migrate"
    assert record.reason == "API cleanup"


# --------------------------- scheduled_*_at ----------------------------------


def test_scheduled_deprecated_at_default_60_days() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    assert scheduled_deprecated_at(record) == T0 + timedelta(days=60)


def test_scheduled_removal_at_default_150_days() -> None:
    """Pin: 60d announce + 90d deprecated = 150d total."""

    record = announce_deprecation(symbol="x", now=T0)
    assert scheduled_removal_at(record) == T0 + timedelta(days=150)


def test_scheduled_with_custom_policy() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    custom = DeprecationPolicy(
        announce_window=timedelta(days=30),
        deprecated_window=timedelta(days=60),
    )
    assert scheduled_deprecated_at(record, policy=custom) == T0 + timedelta(days=30)
    assert scheduled_removal_at(record, policy=custom) == T0 + timedelta(days=90)


# --------------------------- is_overdue_for_advancement ----------------------


def test_overdue_announced_at_60_day_boundary() -> None:
    """Pin: 60d boundary inclusive (>=) for ANNOUNCED → DEPRECATED."""

    record = announce_deprecation(symbol="x", now=T0)
    assert is_overdue_for_advancement(record, now=T0 + timedelta(days=60)) is True
    assert is_overdue_for_advancement(record, now=T0 + timedelta(days=59)) is False


def test_overdue_deprecated_at_150_day_boundary() -> None:
    """Pin: 150d boundary inclusive for DEPRECATED → REMOVED."""

    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0 + timedelta(days=60))  # → DEPRECATED
    assert is_overdue_for_advancement(record, now=T0 + timedelta(days=150)) is True
    assert is_overdue_for_advancement(record, now=T0 + timedelta(days=149)) is False


def test_overdue_removed_never_overdue() -> None:
    """Pin: REMOVED is terminal; never overdue regardless of time."""

    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0)
    record = advance_stage(record, now=T0)  # → REMOVED
    assert is_overdue_for_advancement(record, now=T0 + timedelta(days=1000)) is False


def test_overdue_naive_now_rejected() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    with pytest.raises(ValueError, match="now"):
        is_overdue_for_advancement(record, now=datetime(2026, 5, 1))


# --------------------------- emit_warning_if_needed --------------------------


def test_no_warning_in_announced_stage() -> None:
    """Pin: ANNOUNCED stage emits no warning (silent grace period)."""

    record = announce_deprecation(symbol="old_x", now=T0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        emit_warning_if_needed(record)
    assert len(captured) == 0


def test_warning_in_deprecated_stage() -> None:
    """Pin: DEPRECATED stage emits DeprecationWarning."""

    record = announce_deprecation(symbol="old_x", now=T0)
    record = advance_stage(record, now=T0)  # → DEPRECATED
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        emit_warning_if_needed(record)
    assert len(captured) == 1
    assert issubclass(captured[0].category, DeprecationWarning)
    assert "old_x" in str(captured[0].message)


def test_warning_includes_replacement_when_set() -> None:
    record = announce_deprecation(
        symbol="old_x",
        now=T0,
        replacement="new_x",
        migration_url="https://migrate",
    )
    record = advance_stage(record, now=T0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        emit_warning_if_needed(record)
    msg = str(captured[0].message)
    assert "new_x" in msg
    assert "https://migrate" in msg


def test_no_warning_in_removed_stage() -> None:
    """Pin: REMOVED stage doesn't emit warning (callers should
    raise SymbolRemovedError via assert_not_removed instead)."""

    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0)
    record = advance_stage(record, now=T0)  # → REMOVED
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        emit_warning_if_needed(record)
    assert len(captured) == 0


# --------------------------- assert_not_removed ------------------------------


def test_assert_not_removed_passes_on_announced() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    assert_not_removed(record)  # no raise


def test_assert_not_removed_passes_on_deprecated() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0)
    assert_not_removed(record)  # no raise


def test_assert_not_removed_raises_on_removed() -> None:
    """Pin: REMOVED stage raises SymbolRemovedError."""

    record = announce_deprecation(
        symbol="old_x",
        now=T0,
        replacement="new_x",
        migration_url="https://migrate",
    )
    record = advance_stage(record, now=T0)
    record = advance_stage(record, now=T0)
    with pytest.raises(SymbolRemovedError) as exc_info:
        assert_not_removed(record)
    assert exc_info.value.symbol == "old_x"
    assert exc_info.value.replacement == "new_x"
    assert exc_info.value.migration_url == "https://migrate"


def test_symbol_removed_error_inherits_runtime_error() -> None:
    """Pin: SymbolRemovedError is RuntimeError so generic handlers catch it."""

    err = SymbolRemovedError(symbol="x", replacement="y")
    assert isinstance(err, RuntimeError)


def test_symbol_removed_error_message_includes_replacement() -> None:
    err = SymbolRemovedError(
        symbol="old",
        replacement="new",
        migration_url="https://migrate",
    )
    msg = str(err)
    assert "old" in msg
    assert "new" in msg
    assert "https://migrate" in msg


def test_symbol_removed_error_works_without_replacement() -> None:
    """Pin: not all deprecations have replacements (just-gone features)."""

    err = SymbolRemovedError(symbol="old")
    msg = str(err)
    assert "old" in msg


# --------------------------- filter_overdue ----------------------------------


def test_filter_overdue_returns_only_overdue() -> None:
    fresh = announce_deprecation(symbol="fresh", now=T0)
    old_announced = announce_deprecation(
        symbol="old_announced", now=T0 - timedelta(days=70)
    )  # past 60d → overdue for ANNOUNCED → DEPRECATED
    old_deprecated = announce_deprecation(symbol="old_deprecated", now=T0 - timedelta(days=160))
    old_deprecated = advance_stage(
        old_deprecated, now=T0 - timedelta(days=100)
    )  # past 150d → overdue for DEPRECATED → REMOVED
    overdue = filter_overdue([fresh, old_announced, old_deprecated], now=T0)
    names = {r.symbol for r in overdue}
    assert names == {"old_announced", "old_deprecated"}


def test_filter_overdue_sorted_by_announced_at() -> None:
    """Pin: oldest announcement first (operators tackle the longest-overdue first)."""

    a = announce_deprecation(symbol="newer", now=T0 - timedelta(days=70))
    b = announce_deprecation(symbol="older", now=T0 - timedelta(days=120))
    overdue = filter_overdue([a, b], now=T0)
    names = [r.symbol for r in overdue]
    assert names == ["older", "newer"]


def test_filter_overdue_empty() -> None:
    assert filter_overdue([], now=T0) == ()


def test_filter_overdue_naive_now_rejected() -> None:
    with pytest.raises(ValueError, match="now"):
        filter_overdue([], now=datetime(2026, 5, 1))


# --------------------------- render ------------------------------------------


def test_render_announced_shows_scheduled_deprecated_date() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    out = render_record(record)
    assert "📣" in out
    assert "x" in out
    assert "DEPRECATED scheduled" in out
    # 60 days after T0 = 2026-06-30
    assert "2026-06-30" in out


def test_render_deprecated_shows_scheduled_removal_date() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0 + timedelta(days=60))
    out = render_record(record)
    assert "⚠️" in out
    assert "REMOVED scheduled" in out


def test_render_removed_no_schedule_lines() -> None:
    """Pin: REMOVED records don't show a future scheduled date."""

    record = announce_deprecation(symbol="x", now=T0)
    record = advance_stage(record, now=T0)
    record = advance_stage(record, now=T0)
    out = render_record(record)
    assert "🗑️" in out
    assert "scheduled" not in out


def test_render_includes_replacement_and_url() -> None:
    record = announce_deprecation(
        symbol="x",
        now=T0,
        replacement="y",
        migration_url="https://migrate.example",
        reason="API cleanup",
    )
    out = render_record(record)
    assert "y" in out
    assert "https://migrate.example" in out
    assert "API cleanup" in out


def test_render_omits_optional_when_empty() -> None:
    record = announce_deprecation(symbol="x", now=T0)
    out = render_record(record)
    assert "replacement:" not in out
    assert "migration:" not in out
    assert "reason:" not in out


def test_render_no_secret_leak() -> None:
    """Pin: dataclass doesn't carry operator email / Slack handles, so
    render is structurally secret-free."""

    record = announce_deprecation(symbol="x", now=T0, reason="test")
    out = render_record(record)
    assert "@" not in out
    assert "slack" not in out.lower()
    assert "email" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_full_sunset_lifecycle() -> None:
    """Real-world: announce on Q1 → deprecated by Q2 → removed by Q3."""

    record = announce_deprecation(
        symbol="old_api",
        now=T0,  # 2026-05-01
        replacement="new_api",
        migration_url="https://docs.example.com/migrate-api",
    )

    # Day 30: still in announce window (no warning, not overdue)
    day_30 = T0 + timedelta(days=30)
    assert is_overdue_for_advancement(record, now=day_30) is False

    # Day 60: overdue for ANNOUNCED → DEPRECATED transition
    day_60 = T0 + timedelta(days=60)
    assert is_overdue_for_advancement(record, now=day_60) is True

    # Operator advances to DEPRECATED
    record = advance_stage(record, now=day_60)
    assert record.stage is DeprecationStage.DEPRECATED

    # Day 150: overdue for DEPRECATED → REMOVED
    day_150 = T0 + timedelta(days=150)
    assert is_overdue_for_advancement(record, now=day_150) is True

    # Operator advances to REMOVED
    record = advance_stage(record, now=day_150)
    assert record.stage is DeprecationStage.REMOVED

    # Now any caller that still uses old_api raises SymbolRemovedError
    with pytest.raises(SymbolRemovedError) as exc_info:
        assert_not_removed(record)
    assert exc_info.value.replacement == "new_api"


def test_e2e_warning_during_deprecated_window() -> None:
    """Real-world: user calls deprecated API during DEPRECATED stage,
    sees DeprecationWarning with migration link."""

    record = announce_deprecation(
        symbol="legacy_func",
        now=T0,
        replacement="modern_func",
        migration_url="https://migrate.example/legacy_func",
    )
    record = advance_stage(record, now=T0 + timedelta(days=60))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # Caller uses the deprecated function:
        assert_not_removed(record)  # OK; not removed yet
        emit_warning_if_needed(record)  # Fires DeprecationWarning

    assert len(captured) == 1
    msg = str(captured[0].message)
    assert "legacy_func" in msg
    assert "modern_func" in msg
    assert "https://migrate.example/legacy_func" in msg


def test_e2e_replay_consistency() -> None:
    """Same operations produce equal records."""

    def build() -> DeprecatedSymbol:
        record = announce_deprecation(symbol="x", now=T0)
        return advance_stage(record, now=T0)

    a = build()
    b = build()
    assert a == b
