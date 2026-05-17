"""Tests for `halal_trader.web.feature_flags`.

Auxiliary primitive complementing Wave 10.F edition gating. Covers:
rollout kinds, deterministic per-user evaluation, registry queries,
no-secret render contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.web.feature_flags import (
    FeatureFlag,
    FlagRegistry,
    RolloutKind,
    all_flags,
    enabled_count,
    is_enabled,
    is_enabled_in,
    lookup,
    render_flag,
    render_registry,
)

# --------------------------- Enum string pins --------------------------------


def test_rollout_kind_string_values_pinned() -> None:
    assert RolloutKind.OFF.value == "off"
    assert RolloutKind.ON.value == "on"
    assert RolloutKind.PERCENTAGE.value == "percentage"
    assert RolloutKind.COHORT_ALLOWLIST.value == "cohort_allowlist"


# --------------------------- FeatureFlag validation --------------------------


def test_flag_off_basic() -> None:
    flag = FeatureFlag(
        flag_id="new_dashboard",
        description="The redesigned dashboard",
        kind=RolloutKind.OFF,
    )
    assert flag.kind is RolloutKind.OFF


def test_flag_on_basic() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.ON,
    )
    assert flag.kind is RolloutKind.ON


def test_flag_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="flag_id"):
        FeatureFlag(
            flag_id="",
            description="x",
            kind=RolloutKind.OFF,
        )


def test_flag_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        FeatureFlag(
            flag_id="x",
            description="",
            kind=RolloutKind.OFF,
        )


def test_flag_rejects_percentage_below_zero() -> None:
    with pytest.raises(ValueError, match="percentage"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=-1,
        )


def test_flag_rejects_percentage_above_100() -> None:
    with pytest.raises(ValueError, match="percentage"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=101,
        )


def test_flag_rejects_percentage_kind_with_zero_percent() -> None:
    """Pin: PERCENTAGE with 0% is meaningless — use OFF instead."""

    with pytest.raises(ValueError, match="OFF"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=0,
        )


def test_flag_rejects_percentage_kind_with_100_percent() -> None:
    """Pin: PERCENTAGE with 100% is meaningless — use ON instead."""

    with pytest.raises(ValueError, match="ON"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=100,
        )


def test_flag_percentage_50_accepted() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=50,
    )
    assert flag.percentage == 50


def test_flag_cohort_requires_user_ids() -> None:
    """Pin: COHORT_ALLOWLIST requires non-empty cohort_user_ids."""

    with pytest.raises(ValueError, match="cohort_user_ids"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.COHORT_ALLOWLIST,
            cohort_user_ids=frozenset(),
        )


def test_flag_off_must_not_have_cohort() -> None:
    """Pin: OFF kind must not carry a cohort (confused state)."""

    with pytest.raises(ValueError, match="cohort_user_ids"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.OFF,
            cohort_user_ids=frozenset({"u1"}),
        )


def test_flag_on_must_not_have_cohort() -> None:
    with pytest.raises(ValueError, match="cohort_user_ids"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.ON,
            cohort_user_ids=frozenset({"u1"}),
        )


def test_flag_percentage_must_not_have_cohort() -> None:
    with pytest.raises(ValueError, match="cohort_user_ids"):
        FeatureFlag(
            flag_id="x",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=50,
            cohort_user_ids=frozenset({"u1"}),
        )


def test_flag_is_frozen() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.OFF,
    )
    with pytest.raises(FrozenInstanceError):
        flag.kind = RolloutKind.ON  # type: ignore[misc]


# --------------------------- is_enabled: OFF / ON ----------------------------


def test_is_enabled_off_always_false() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.OFF,
    )
    assert is_enabled(flag, user_id="alice") is False
    assert is_enabled(flag, user_id="bob") is False


def test_is_enabled_on_always_true() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.ON,
    )
    assert is_enabled(flag, user_id="alice") is True
    assert is_enabled(flag, user_id="bob") is True


def test_is_enabled_rejects_empty_user_id() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.OFF,
    )
    with pytest.raises(ValueError, match="user_id"):
        is_enabled(flag, user_id="")


# --------------------------- is_enabled: PERCENTAGE --------------------------


def test_percentage_deterministic_per_user() -> None:
    """Pin: same (flag, user) always returns same answer (no flicker)."""

    flag = FeatureFlag(
        flag_id="new_feature",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=50,
    )
    a1 = is_enabled(flag, user_id="alice")
    a2 = is_enabled(flag, user_id="alice")
    a3 = is_enabled(flag, user_id="alice")
    assert a1 == a2 == a3


def test_percentage_50_roughly_half_of_users() -> None:
    """Pin: 50% rollout enables roughly half a sample of 1000 users."""

    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=50,
    )
    enabled = sum(1 for i in range(1000) if is_enabled(flag, user_id=f"u{i}"))
    # Should be in [400, 600] for SHA-256 distribution
    assert 400 <= enabled <= 600


def test_percentage_10_roughly_10_percent() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=10,
    )
    enabled = sum(1 for i in range(1000) if is_enabled(flag, user_id=f"u{i}"))
    assert 70 <= enabled <= 130


def test_percentage_correlation_free() -> None:
    """Pin: two flags rolling out at 50% don't enable the same user-half.

    The flag_id salt means flag_a's half is not flag_b's half;
    overlap should be ~25% (not ~50% which would mean correlated).
    """

    flag_a = FeatureFlag(
        flag_id="flag_a",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=50,
    )
    flag_b = FeatureFlag(
        flag_id="flag_b",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=50,
    )
    both = sum(
        1
        for i in range(1000)
        if is_enabled(flag_a, user_id=f"u{i}") and is_enabled(flag_b, user_id=f"u{i}")
    )
    # Independent 50% × 50% = 25%; allow 18-32% range
    assert 180 <= both <= 320


# --------------------------- is_enabled: COHORT_ALLOWLIST --------------------


def test_cohort_user_in_list() -> None:
    flag = FeatureFlag(
        flag_id="beta",
        description="Beta tester features",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"alice", "bob"}),
    )
    assert is_enabled(flag, user_id="alice") is True
    assert is_enabled(flag, user_id="bob") is True


def test_cohort_user_not_in_list() -> None:
    flag = FeatureFlag(
        flag_id="beta",
        description="x",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"alice", "bob"}),
    )
    assert is_enabled(flag, user_id="charlie") is False


def test_cohort_explicit_not_hash_based() -> None:
    """Pin: cohort allowlist is explicit; not hash-based.

    A user added to the cohort is enabled regardless of hash;
    a user removed from the cohort flips off cleanly.
    """

    cohort_in = FeatureFlag(
        flag_id="beta",
        description="x",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"u_special"}),
    )
    cohort_out = FeatureFlag(
        flag_id="beta",
        description="x",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"u_other"}),
    )
    assert is_enabled(cohort_in, user_id="u_special") is True
    assert is_enabled(cohort_out, user_id="u_special") is False


# --------------------------- FlagRegistry ------------------------------------


def test_registry_basic() -> None:
    f1 = FeatureFlag(
        flag_id="f1",
        description="x",
        kind=RolloutKind.OFF,
    )
    f2 = FeatureFlag(
        flag_id="f2",
        description="x",
        kind=RolloutKind.ON,
    )
    registry = FlagRegistry(flags=frozenset({f1, f2}))
    assert len(registry.flags) == 2


def test_registry_rejects_duplicate_flag_ids() -> None:
    f1 = FeatureFlag(
        flag_id="dup",
        description="x",
        kind=RolloutKind.OFF,
    )
    f2 = FeatureFlag(
        flag_id="dup",
        description="y",
        kind=RolloutKind.ON,
    )
    with pytest.raises(ValueError, match="duplicate"):
        FlagRegistry(flags=frozenset({f1, f2}))


def test_registry_is_frozen() -> None:
    registry = FlagRegistry(flags=frozenset())
    with pytest.raises(FrozenInstanceError):
        registry.flags = frozenset()  # type: ignore[misc]


# --------------------------- lookup + is_enabled_in --------------------------


def test_lookup_finds_flag() -> None:
    f1 = FeatureFlag(
        flag_id="f1",
        description="x",
        kind=RolloutKind.OFF,
    )
    registry = FlagRegistry(flags=frozenset({f1}))
    assert lookup(registry, "f1") is f1


def test_lookup_unknown_raises() -> None:
    registry = FlagRegistry(flags=frozenset())
    with pytest.raises(KeyError):
        lookup(registry, "nonexistent")


def test_is_enabled_in_combines_lookup_and_eval() -> None:
    f1 = FeatureFlag(
        flag_id="f1",
        description="x",
        kind=RolloutKind.ON,
    )
    registry = FlagRegistry(flags=frozenset({f1}))
    assert is_enabled_in(registry, "f1", user_id="alice") is True


def test_is_enabled_in_unknown_flag_raises() -> None:
    registry = FlagRegistry(flags=frozenset())
    with pytest.raises(KeyError):
        is_enabled_in(registry, "missing", user_id="alice")


# --------------------------- all_flags / enabled_count -----------------------


def test_all_flags_sorted() -> None:
    """Pin: deterministic order (sorted by flag_id)."""

    f_z = FeatureFlag(
        flag_id="zebra",
        description="x",
        kind=RolloutKind.OFF,
    )
    f_a = FeatureFlag(
        flag_id="apple",
        description="x",
        kind=RolloutKind.ON,
    )
    registry = FlagRegistry(flags=frozenset({f_z, f_a}))
    flags = all_flags(registry)
    assert [f.flag_id for f in flags] == ["apple", "zebra"]


def test_enabled_count_basic() -> None:
    flag = FeatureFlag(
        flag_id="f1",
        description="x",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"alice", "bob"}),
    )
    count = enabled_count(flag, sample_user_ids=["alice", "bob", "charlie"])
    assert count == 2


def test_enabled_count_off_returns_zero() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.OFF,
    )
    assert enabled_count(flag, sample_user_ids=["a", "b", "c"]) == 0


def test_enabled_count_on_returns_all() -> None:
    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.ON,
    )
    assert enabled_count(flag, sample_user_ids=["a", "b", "c"]) == 3


# --------------------------- render ------------------------------------------


def test_render_flag_off_shows_emoji() -> None:
    flag = FeatureFlag(
        flag_id="f1",
        description="Test feature",
        kind=RolloutKind.OFF,
    )
    out = render_flag(flag)
    assert "⚫" in out
    assert "f1" in out
    assert "Test feature" in out


def test_render_flag_percentage_shows_value() -> None:
    flag = FeatureFlag(
        flag_id="f1",
        description="Rollout in progress",
        kind=RolloutKind.PERCENTAGE,
        percentage=25,
    )
    out = render_flag(flag)
    assert "📊" in out
    assert "25%" in out


def test_render_flag_cohort_shows_count_not_ids() -> None:
    """Pin: render shows cohort SIZE, not individual user IDs."""

    flag = FeatureFlag(
        flag_id="beta",
        description="Beta",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"alice@example.com", "bob@example.com", "charlie@example.com"}),
    )
    out = render_flag(flag)
    assert "👥" in out
    assert "3 users" in out
    # Pin: no individual user IDs leak
    assert "alice" not in out
    assert "bob" not in out
    assert "charlie" not in out
    assert "@" not in out


def test_render_flag_no_secret_leak() -> None:
    """Pin: render never includes individual user_ids."""

    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"sensitive_user_handle_xyz"}),
    )
    out = render_flag(flag)
    assert "sensitive_user_handle_xyz" not in out


def test_render_registry_includes_summary_counts() -> None:
    flags = frozenset(
        {
            FeatureFlag(
                flag_id="f1",
                description="x",
                kind=RolloutKind.OFF,
            ),
            FeatureFlag(
                flag_id="f2",
                description="x",
                kind=RolloutKind.ON,
            ),
            FeatureFlag(
                flag_id="f3",
                description="x",
                kind=RolloutKind.PERCENTAGE,
                percentage=50,
            ),
        }
    )
    registry = FlagRegistry(flags=flags)
    out = render_registry(registry)
    assert "3 flags total" in out
    assert "off: 1" in out
    assert "on: 1" in out
    assert "percentage: 1" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_gradual_rollout() -> None:
    """Real-world: roll a feature out 0 → 10 → 50 → 100% with a fixed
    user cohort, verify each user's enabled state monotonically
    becomes True (never flips back to False at higher percentages)."""

    user_ids = [f"u{i}" for i in range(100)]

    rollouts = [
        # 0% (off)
        FeatureFlag(
            flag_id="my_feature",
            description="x",
            kind=RolloutKind.OFF,
        ),
        # 10%
        FeatureFlag(
            flag_id="my_feature",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=10,
        ),
        # 50%
        FeatureFlag(
            flag_id="my_feature",
            description="x",
            kind=RolloutKind.PERCENTAGE,
            percentage=50,
        ),
        # 100% (on)
        FeatureFlag(
            flag_id="my_feature",
            description="x",
            kind=RolloutKind.ON,
        ),
    ]

    enabled_history: list[set[str]] = []
    for flag in rollouts:
        enabled_now = {uid for uid in user_ids if is_enabled(flag, user_id=uid)}
        enabled_history.append(enabled_now)

    # Pin: monotonic non-decreasing enabled set across rollout stages
    for i in range(len(enabled_history) - 1):
        # Each user enabled at stage i is still enabled at stage i+1
        assert enabled_history[i] <= enabled_history[i + 1], (
            f"stage {i} → {i + 1} flickered: lost users "
            f"{enabled_history[i] - enabled_history[i + 1]}"
        )


def test_e2e_cohort_then_percentage() -> None:
    """Beta cohort gets feature first, then graduates to percentage rollout."""

    cohort_flag = FeatureFlag(
        flag_id="my_feature",
        description="Beta phase",
        kind=RolloutKind.COHORT_ALLOWLIST,
        cohort_user_ids=frozenset({"alice", "bob"}),
    )
    percentage_flag = FeatureFlag(
        flag_id="my_feature",
        description="Percentage phase",
        kind=RolloutKind.PERCENTAGE,
        percentage=50,
    )

    # Beta cohort: only alice + bob enabled
    assert is_enabled(cohort_flag, user_id="alice") is True
    assert is_enabled(cohort_flag, user_id="charlie") is False

    # Percentage phase: ~half of users enabled (deterministic by hash)
    enabled = sum(1 for i in range(100) if is_enabled(percentage_flag, user_id=f"u{i}"))
    assert 30 <= enabled <= 70


def test_e2e_replay_consistency() -> None:
    """Same flag + user always returns same answer."""

    flag = FeatureFlag(
        flag_id="x",
        description="x",
        kind=RolloutKind.PERCENTAGE,
        percentage=33,
    )
    a = is_enabled(flag, user_id="alice")
    b = is_enabled(flag, user_id="alice")
    c = is_enabled(flag, user_id="alice")
    assert a == b == c
