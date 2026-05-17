"""Tests for `halal_trader.web.feature_gate` (Wave 10.F).

Covers: edition-tier availability matrix, OSS-strict-subset pin,
core-features-stay-OSS regression pin, tier ordering, render
matrix output, immutability of the registry.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.web.feature_gate import (
    Edition,
    Feature,
    FeatureNotAvailableError,
    FeatureSpec,
    all_features,
    feature_spec,
    features_available,
    is_feature_available,
    render_context_summary,
    render_matrix,
    require_feature,
)
from halal_trader.web.quotas import Tier

# --------------------------- Enum string pins --------------------------------


def test_edition_string_values_pinned() -> None:
    assert Edition.OSS.value == "oss"
    assert Edition.HOSTED.value == "hosted"


def test_feature_string_values_pinned() -> None:
    """Pin: every feature's string value (DB / JSON stability)."""

    assert Feature.CYCLE_RUN.value == "cycle_run"
    assert Feature.STOCK_TRADING.value == "stock_trading"
    assert Feature.CRYPTO_TRADING.value == "crypto_trading"
    assert Feature.HALAL_SCREENER.value == "halal_screener"
    assert Feature.LOCAL_DASHBOARD.value == "local_dashboard"
    assert Feature.LOCAL_BACKTEST.value == "local_backtest"
    assert Feature.PURIFICATION_LEDGER.value == "purification_ledger"
    assert Feature.MULTI_USER_AUTH.value == "multi_user_auth"
    assert Feature.PER_USER_VAULT.value == "per_user_vault"
    assert Feature.PER_USER_QUOTAS.value == "per_user_quotas"
    assert Feature.BILLING.value == "billing"
    assert Feature.ADMIN_CONSOLE.value == "admin_console"
    assert Feature.LEADERBOARD.value == "leaderboard"
    assert Feature.ONBOARDING_FLOW.value == "onboarding_flow"
    assert Feature.LIVE_LLM_CYCLES.value == "live_llm_cycles"
    assert Feature.PREMIUM_DATASETS.value == "premium_datasets"
    assert Feature.SCHOLAR_REVIEW_QUEUE.value == "scholar_review_queue"
    assert Feature.PUBLIC_RESEARCH_API.value == "public_research_api"


# --------------------------- FeatureSpec -------------------------------------


def test_feature_spec_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        FeatureSpec(
            feature=Feature.CYCLE_RUN,
            oss_available=True,
            min_tier=Tier.FREE,
            description="",
        )


def test_feature_spec_rejects_whitespace_description() -> None:
    with pytest.raises(ValueError, match="description"):
        FeatureSpec(
            feature=Feature.CYCLE_RUN,
            oss_available=True,
            min_tier=Tier.FREE,
            description="   ",
        )


def test_feature_spec_is_frozen() -> None:
    spec = feature_spec(Feature.CYCLE_RUN)
    with pytest.raises(FrozenInstanceError):
        spec.oss_available = False  # type: ignore[misc]


# --------------------------- registry coverage -------------------------------


def test_all_features_returns_one_per_enum() -> None:
    """Pin: every Feature enum value has a registry entry."""

    specs = all_features()
    assert len(specs) == len(Feature)
    feature_set = {spec.feature for spec in specs}
    for feature in Feature:
        assert feature in feature_set


def test_all_features_canonical_order() -> None:
    """Pin: all_features() respects Feature-enum order."""

    specs = all_features()
    enum_order = list(Feature)
    spec_order = [s.feature for s in specs]
    assert spec_order == enum_order


def test_feature_spec_returns_correct_spec() -> None:
    spec = feature_spec(Feature.LIVE_LLM_CYCLES)
    assert spec.feature is Feature.LIVE_LLM_CYCLES
    assert spec.min_tier is Tier.PRO
    assert spec.oss_available is False


# --------------------------- core features pinned in OSS ---------------------


def test_core_trading_features_available_in_oss() -> None:
    """Regression pin: core trading + screener stays in OSS forever.

    A future PR walling off the core engine behind a paywall fails CI.
    """

    for feature in (
        Feature.CYCLE_RUN,
        Feature.STOCK_TRADING,
        Feature.CRYPTO_TRADING,
        Feature.HALAL_SCREENER,
    ):
        assert is_feature_available(feature, edition=Edition.OSS) is True, feature


def test_core_dashboard_and_backtest_available_in_oss() -> None:
    for feature in (
        Feature.LOCAL_DASHBOARD,
        Feature.LOCAL_BACKTEST,
        Feature.PURIFICATION_LEDGER,
    ):
        assert is_feature_available(feature, edition=Edition.OSS) is True, feature


# --------------------------- hosted-only features ----------------------------


def test_multi_user_features_unavailable_in_oss() -> None:
    """Pin: hosted-only features are NOT in OSS regardless."""

    for feature in (
        Feature.MULTI_USER_AUTH,
        Feature.PER_USER_VAULT,
        Feature.PER_USER_QUOTAS,
        Feature.BILLING,
        Feature.ADMIN_CONSOLE,
        Feature.LEADERBOARD,
        Feature.ONBOARDING_FLOW,
    ):
        assert is_feature_available(feature, edition=Edition.OSS) is False, feature


def test_tier_gated_features_unavailable_in_oss() -> None:
    for feature in (
        Feature.LIVE_LLM_CYCLES,
        Feature.PREMIUM_DATASETS,
        Feature.SCHOLAR_REVIEW_QUEUE,
        Feature.PUBLIC_RESEARCH_API,
    ):
        assert is_feature_available(feature, edition=Edition.OSS) is False, feature


# --------------------------- OSS strict subset pin ---------------------------


def test_oss_is_strict_subset_of_hosted_enterprise() -> None:
    """Pin: every OSS-available feature is also available at HOSTED+ENTERPRISE.

    OSS users transitioning to HOSTED must never lose features.
    """

    for spec in all_features():
        if spec.oss_available:
            assert is_feature_available(
                spec.feature, edition=Edition.HOSTED, tier=Tier.ENTERPRISE
            ), spec.feature


def test_oss_subset_holds_at_free_tier() -> None:
    """Pin: every OSS-available feature requires at most FREE tier in HOSTED."""

    for spec in all_features():
        if spec.oss_available:
            assert spec.min_tier is Tier.FREE, spec.feature


# --------------------------- tier ordering -----------------------------------


def test_free_user_unavailable_for_pro_feature() -> None:
    assert (
        is_feature_available(Feature.LIVE_LLM_CYCLES, edition=Edition.HOSTED, tier=Tier.FREE)
        is False
    )


def test_pro_user_available_for_pro_feature() -> None:
    assert (
        is_feature_available(Feature.LIVE_LLM_CYCLES, edition=Edition.HOSTED, tier=Tier.PRO) is True
    )


def test_enterprise_user_available_for_pro_feature() -> None:
    """Pin: ENTERPRISE > PRO so PRO features are available."""

    assert (
        is_feature_available(Feature.LIVE_LLM_CYCLES, edition=Edition.HOSTED, tier=Tier.ENTERPRISE)
        is True
    )


def test_pro_user_unavailable_for_enterprise_feature() -> None:
    assert (
        is_feature_available(Feature.PUBLIC_RESEARCH_API, edition=Edition.HOSTED, tier=Tier.PRO)
        is False
    )


def test_enterprise_user_available_for_enterprise_feature() -> None:
    assert (
        is_feature_available(
            Feature.PUBLIC_RESEARCH_API,
            edition=Edition.HOSTED,
            tier=Tier.ENTERPRISE,
        )
        is True
    )


def test_free_user_available_for_free_tier_feature() -> None:
    assert (
        is_feature_available(Feature.MULTI_USER_AUTH, edition=Edition.HOSTED, tier=Tier.FREE)
        is True
    )


# --------------------------- HOSTED requires tier ----------------------------


def test_hosted_without_tier_raises() -> None:
    with pytest.raises(ValueError, match="tier"):
        is_feature_available(Feature.CYCLE_RUN, edition=Edition.HOSTED)


def test_oss_ignores_tier() -> None:
    """Pin: OSS edition treats tier as irrelevant."""

    a = is_feature_available(Feature.CYCLE_RUN, edition=Edition.OSS)
    b = is_feature_available(Feature.CYCLE_RUN, edition=Edition.OSS, tier=Tier.PRO)
    assert a is True
    assert b is True


# --------------------------- require_feature ---------------------------------


def test_require_feature_silent_on_available() -> None:
    """No exception when available."""

    require_feature(Feature.CYCLE_RUN, edition=Edition.OSS)


def test_require_feature_raises_on_oss_blocked() -> None:
    with pytest.raises(FeatureNotAvailableError) as exc_info:
        require_feature(Feature.BILLING, edition=Edition.OSS)
    assert exc_info.value.feature is Feature.BILLING
    assert exc_info.value.edition is Edition.OSS
    assert exc_info.value.tier is None


def test_require_feature_raises_on_tier_below_min() -> None:
    with pytest.raises(FeatureNotAvailableError) as exc_info:
        require_feature(
            Feature.PUBLIC_RESEARCH_API,
            edition=Edition.HOSTED,
            tier=Tier.PRO,
        )
    assert exc_info.value.tier is Tier.PRO


def test_require_feature_silent_on_meeting_min_tier() -> None:
    require_feature(Feature.LIVE_LLM_CYCLES, edition=Edition.HOSTED, tier=Tier.PRO)


# --------------------------- features_available ------------------------------


def test_features_available_oss_excludes_hosted_only() -> None:
    available = features_available(edition=Edition.OSS)
    feature_ids = {spec.feature for spec in available}
    assert Feature.CYCLE_RUN in feature_ids
    assert Feature.BILLING not in feature_ids
    assert Feature.ADMIN_CONSOLE not in feature_ids


def test_features_available_hosted_free_includes_core_and_multiuser() -> None:
    available = features_available(edition=Edition.HOSTED, tier=Tier.FREE)
    feature_ids = {spec.feature for spec in available}
    assert Feature.CYCLE_RUN in feature_ids
    assert Feature.MULTI_USER_AUTH in feature_ids
    assert Feature.LIVE_LLM_CYCLES not in feature_ids


def test_features_available_hosted_pro_includes_pro_features() -> None:
    available = features_available(edition=Edition.HOSTED, tier=Tier.PRO)
    feature_ids = {spec.feature for spec in available}
    assert Feature.LIVE_LLM_CYCLES in feature_ids
    assert Feature.PUBLIC_RESEARCH_API not in feature_ids


def test_features_available_hosted_enterprise_includes_all_hosted() -> None:
    available = features_available(edition=Edition.HOSTED, tier=Tier.ENTERPRISE)
    feature_ids = {spec.feature for spec in available}
    # Every feature should be available
    for feature in Feature:
        assert feature in feature_ids, feature


def test_features_available_hosted_without_tier_raises() -> None:
    with pytest.raises(ValueError, match="tier"):
        features_available(edition=Edition.HOSTED)


def test_features_available_is_deterministic() -> None:
    a = features_available(edition=Edition.HOSTED, tier=Tier.PRO)
    b = features_available(edition=Edition.HOSTED, tier=Tier.PRO)
    assert a == b


# --------------------------- FeatureNotAvailableError ------------------------


def test_error_message_includes_feature_and_edition() -> None:
    err = FeatureNotAvailableError(Feature.BILLING, Edition.OSS, None)
    assert "billing" in str(err)
    assert "oss" in str(err)


def test_error_message_includes_tier_when_set() -> None:
    err = FeatureNotAvailableError(Feature.PUBLIC_RESEARCH_API, Edition.HOSTED, Tier.PRO)
    assert "pro" in str(err)


# --------------------------- render_matrix -----------------------------------


def test_render_matrix_includes_headers() -> None:
    out = render_matrix()
    assert "Feature" in out
    assert "OSS" in out
    assert "Free" in out
    assert "Pro" in out
    assert "Enterprise" in out


def test_render_matrix_includes_every_feature() -> None:
    out = render_matrix()
    for feature in Feature:
        assert feature.value in out, feature


def test_render_matrix_marks_oss_features() -> None:
    out = render_matrix()
    # Find the cycle_run row, ensure it has ✅ on the OSS column
    for line in out.splitlines():
        if line.startswith("cycle_run"):
            # Count ✅ — should be 4 (OSS + Free + Pro + Enterprise)
            assert line.count("✅") == 4
            break
    else:
        pytest.fail("cycle_run row not found")


def test_render_matrix_marks_hosted_only_features() -> None:
    """Pin: a hosted-only feature has '—' under OSS, ✅ under hosted tiers."""

    out = render_matrix()
    for line in out.splitlines():
        if line.startswith("billing"):
            # Should have '—' once (OSS) and ✅ thrice (free/pro/enterprise)
            assert line.count("—") == 1
            assert line.count("✅") == 3
            break
    else:
        pytest.fail("billing row not found")


def test_render_matrix_marks_enterprise_only_features() -> None:
    """Pin: PUBLIC_RESEARCH_API has ✅ only under Enterprise."""

    out = render_matrix()
    for line in out.splitlines():
        if line.startswith("public_research_api"):
            assert line.count("✅") == 1
            break
    else:
        pytest.fail("public_research_api row not found")


def test_render_matrix_no_secret_leak() -> None:
    """Pin: render never names env vars / Stripe IDs / API keys."""

    out = render_matrix()
    assert "cus_" not in out.lower()
    assert "sub_" not in out.lower()
    assert "api_key" not in out.lower()
    assert "STRIPE_" not in out
    assert "SECRET_" not in out


# --------------------------- render_context_summary --------------------------


def test_context_summary_oss_lists_oss_features() -> None:
    out = render_context_summary(edition=Edition.OSS)
    assert "oss" in out.lower()
    assert "cycle_run" in out
    assert "billing" not in out


def test_context_summary_hosted_pro_lists_pro_features() -> None:
    out = render_context_summary(edition=Edition.HOSTED, tier=Tier.PRO)
    assert "live_llm_cycles" in out
    assert "public_research_api" not in out


def test_context_summary_includes_count() -> None:
    out = render_context_summary(edition=Edition.OSS)
    # Format is "N/M features available"
    import re

    assert re.search(r"\d+/\d+ features available", out)


def test_context_summary_hosted_without_tier_raises() -> None:
    with pytest.raises(ValueError, match="tier"):
        render_context_summary(edition=Edition.HOSTED)


# --------------------------- e2e flows ---------------------------------------


def test_e2e_oss_user_can_run_full_trading_loop() -> None:
    """Real-world: OSS user runs cycle + screener + dashboard."""

    require_feature(Feature.CYCLE_RUN, edition=Edition.OSS)
    require_feature(Feature.HALAL_SCREENER, edition=Edition.OSS)
    require_feature(Feature.LOCAL_DASHBOARD, edition=Edition.OSS)
    require_feature(Feature.LOCAL_BACKTEST, edition=Edition.OSS)


def test_e2e_oss_user_blocked_from_billing() -> None:
    """Real-world: OSS user can't use billing because they don't pay."""

    with pytest.raises(FeatureNotAvailableError):
        require_feature(Feature.BILLING, edition=Edition.OSS)


def test_e2e_hosted_free_user_has_basic_features() -> None:
    """FREE user gets multi-user auth + screener but not LLM cycles."""

    require_feature(Feature.MULTI_USER_AUTH, edition=Edition.HOSTED, tier=Tier.FREE)
    require_feature(Feature.HALAL_SCREENER, edition=Edition.HOSTED, tier=Tier.FREE)
    with pytest.raises(FeatureNotAvailableError):
        require_feature(Feature.LIVE_LLM_CYCLES, edition=Edition.HOSTED, tier=Tier.FREE)


def test_e2e_hosted_enterprise_user_has_everything() -> None:
    for feature in Feature:
        require_feature(feature, edition=Edition.HOSTED, tier=Tier.ENTERPRISE)


def test_e2e_pro_to_enterprise_unlocks_research_api() -> None:
    """Pin: tier upgrade unlocks the research API."""

    with pytest.raises(FeatureNotAvailableError):
        require_feature(
            Feature.PUBLIC_RESEARCH_API,
            edition=Edition.HOSTED,
            tier=Tier.PRO,
        )
    require_feature(
        Feature.PUBLIC_RESEARCH_API,
        edition=Edition.HOSTED,
        tier=Tier.ENTERPRISE,
    )
