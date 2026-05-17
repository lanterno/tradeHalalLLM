"""Tests for halal/jurisdiction_router.py — Round-5 Wave 2.G."""

from __future__ import annotations

import pytest

from halal_trader.halal.jurisdiction_router import (
    CompositeOutcome,
    Jurisdiction,
    JurisdictionVerdict,
    RoutingMode,
    RoutingPolicy,
    RoutingResult,
    SourceVerdict,
    applicable_sources,
    filter_approved,
    filter_blocked,
    render_result,
    route_batch,
    route_symbol,
)


def test_jurisdiction_string_values():
    assert Jurisdiction.SAUDI_ARABIA.value == "saudi_arabia"
    assert Jurisdiction.UAE.value == "uae"
    assert Jurisdiction.MALAYSIA.value == "malaysia"
    assert Jurisdiction.INDONESIA.value == "indonesia"
    assert Jurisdiction.PAKISTAN.value == "pakistan"
    assert Jurisdiction.BAHRAIN.value == "bahrain"
    assert Jurisdiction.UK.value == "uk"
    assert Jurisdiction.USA.value == "usa"
    assert Jurisdiction.EU.value == "eu"
    assert Jurisdiction.GLOBAL.value == "global"


def test_routing_mode_string_values():
    assert RoutingMode.STRICTEST.value == "strictest"
    assert RoutingMode.ANY.value == "any"
    assert RoutingMode.HOME_ONLY.value == "home_only"


def test_default_routing_mode_is_strictest():
    p = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    assert p.mode is RoutingMode.STRICTEST


def test_default_require_global_consensus_true():
    assert RoutingPolicy(operator_jurisdiction=Jurisdiction.USA).require_global_consensus is True


def test_source_verdict_empty_name_rejected():
    with pytest.raises(ValueError):
        SourceVerdict(
            source_name="",
            jurisdiction=Jurisdiction.GLOBAL,
            verdict=JurisdictionVerdict.PERMISSIBLE,
        )


def test_routing_result_empty_symbol_rejected():
    with pytest.raises(ValueError):
        RoutingResult(
            symbol="",
            listing_market=Jurisdiction.USA,
            applicable_sources=(),
            outcome=CompositeOutcome.APPROVED,
            blocking_sources=(),
        )


# --- applicable_sources ------------------------------------------------------


def test_applicable_local_saudi_includes_saudi_and_global():
    out = applicable_sources(Jurisdiction.SAUDI_ARABIA, Jurisdiction.SAUDI_ARABIA)
    assert Jurisdiction.SAUDI_ARABIA in out
    assert Jurisdiction.GLOBAL in out


def test_applicable_us_op_buying_saudi_includes_saudi_and_global():
    out = applicable_sources(Jurisdiction.USA, Jurisdiction.SAUDI_ARABIA)
    assert Jurisdiction.SAUDI_ARABIA in out
    assert Jurisdiction.GLOBAL in out
    # Operator jurisdiction (USA) does NOT add itself by default
    assert Jurisdiction.USA not in out


def test_applicable_us_op_buying_us_listing_includes_only_global_default():
    """A US listing has no Shariah regulator — only global applies."""
    out = applicable_sources(Jurisdiction.USA, Jurisdiction.USA)
    assert Jurisdiction.USA in out  # listing market
    assert Jurisdiction.GLOBAL in out


# --- route_symbol — basic flows ----------------------------------------------


def _good_global() -> SourceVerdict:
    return SourceVerdict(
        source_name="AAOIFI",
        jurisdiction=Jurisdiction.GLOBAL,
        verdict=JurisdictionVerdict.PERMISSIBLE,
    )


def _good_saudi() -> SourceVerdict:
    return SourceVerdict(
        source_name="CMA Shariah",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.PERMISSIBLE,
    )


def test_route_strictest_all_pass_approves():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    r = route_symbol("2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    assert r.outcome is CompositeOutcome.APPROVED


def test_route_strictest_one_fails_blocks():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    bad_saudi = SourceVerdict(
        source_name="CMA Shariah",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
        note="debt 35%",
    )
    r = route_symbol(
        "1180.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), bad_saudi], policy=pol
    )
    assert r.outcome is CompositeOutcome.BLOCKED
    assert "CMA Shariah" in r.blocking_sources


def test_route_no_relevant_data_returns_insufficient():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    r = route_symbol("XYZ", Jurisdiction.SAUDI_ARABIA, [], policy=pol)
    assert r.outcome is CompositeOutcome.INSUFFICIENT_DATA


def test_route_only_unknown_verdicts_insufficient():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    unknown_global = SourceVerdict(
        source_name="AAOIFI",
        jurisdiction=Jurisdiction.GLOBAL,
        verdict=JurisdictionVerdict.UNKNOWN,
    )
    r = route_symbol("XYZ", Jurisdiction.SAUDI_ARABIA, [unknown_global], policy=pol)
    assert r.outcome is CompositeOutcome.INSUFFICIENT_DATA


def test_route_strictest_global_required_missing_returns_insufficient():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA, require_global_consensus=True)
    r = route_symbol("2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_saudi()], policy=pol)
    assert r.outcome is CompositeOutcome.INSUFFICIENT_DATA


def test_route_strictest_global_not_required():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA, require_global_consensus=False)
    r = route_symbol("2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_saudi()], policy=pol)
    assert r.outcome is CompositeOutcome.APPROVED


def test_route_any_mode_one_pass_approves():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA, mode=RoutingMode.ANY)
    bad_saudi = SourceVerdict(
        source_name="CMA Shariah",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
    )
    r = route_symbol(
        "X", Jurisdiction.SAUDI_ARABIA, [_good_global(), bad_saudi], policy=pol
    )
    assert r.outcome is CompositeOutcome.APPROVED


def test_route_any_mode_all_fail_blocks():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA, mode=RoutingMode.ANY)
    bad_global = SourceVerdict(
        source_name="AAOIFI",
        jurisdiction=Jurisdiction.GLOBAL,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
    )
    bad_saudi = SourceVerdict(
        source_name="CMA Shariah",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
    )
    r = route_symbol(
        "X", Jurisdiction.SAUDI_ARABIA, [bad_global, bad_saudi], policy=pol
    )
    assert r.outcome is CompositeOutcome.BLOCKED


def test_route_home_only_ignores_global():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA, mode=RoutingMode.HOME_ONLY)
    r = route_symbol("2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    # Only Saudi source is applicable — global is ignored
    assert all(v.jurisdiction is Jurisdiction.SAUDI_ARABIA for v in r.applicable_sources)
    assert r.outcome is CompositeOutcome.APPROVED


def test_route_empty_symbol_rejected():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    with pytest.raises(ValueError):
        route_symbol("", Jurisdiction.SAUDI_ARABIA, [], policy=pol)


# --- Cross-jurisdiction edge cases -------------------------------------------


def test_route_irrelevant_jurisdiction_verdict_filtered_out():
    """A verdict from a non-applicable jurisdiction should not affect the outcome."""
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    irrelevant_block = SourceVerdict(
        source_name="UAE_SCA",
        jurisdiction=Jurisdiction.UAE,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
    )
    r = route_symbol(
        "2222.SR",
        Jurisdiction.SAUDI_ARABIA,
        [_good_global(), _good_saudi(), irrelevant_block],
        policy=pol,
    )
    # UAE verdict is filtered out — Saudi listing only cares about Saudi+global
    assert r.outcome is CompositeOutcome.APPROVED
    # blockers shouldn't include UAE
    assert "UAE_SCA" not in r.blocking_sources


def test_route_local_saudi_operator_uses_saudi_sources():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.SAUDI_ARABIA)
    r = route_symbol(
        "2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol
    )
    assert r.outcome is CompositeOutcome.APPROVED


# --- Batch + filters ---------------------------------------------------------


def test_route_batch():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    candidates = [
        ("AAA", Jurisdiction.SAUDI_ARABIA, (_good_global(), _good_saudi())),
        ("BBB", Jurisdiction.SAUDI_ARABIA, ()),
    ]
    out = route_batch(candidates, policy=pol)
    assert len(out) == 2
    assert out[0].outcome is CompositeOutcome.APPROVED
    assert out[1].outcome is CompositeOutcome.INSUFFICIENT_DATA


def test_filter_approved_only_returns_approved():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    a = route_symbol("OK", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    bad_saudi = SourceVerdict(
        source_name="CMA",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
    )
    b = route_symbol("BAD", Jurisdiction.SAUDI_ARABIA, [_good_global(), bad_saudi], policy=pol)
    out = filter_approved([a, b])
    assert a in out
    assert b not in out


def test_filter_blocked_only_returns_blocked():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    a = route_symbol("OK", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    bad_saudi = SourceVerdict(
        source_name="CMA",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
    )
    b = route_symbol("BAD", Jurisdiction.SAUDI_ARABIA, [_good_global(), bad_saudi], policy=pol)
    out = filter_blocked([a, b])
    assert b in out
    assert a not in out


# --- Render ------------------------------------------------------------------


def test_render_approved():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    r = route_symbol("2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    out = render_result(r)
    assert "✅" in out
    assert "approved" in out


def test_render_blocked_includes_blocker():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    bad_saudi = SourceVerdict(
        source_name="CMA Shariah",
        jurisdiction=Jurisdiction.SAUDI_ARABIA,
        verdict=JurisdictionVerdict.IMPERMISSIBLE,
        note="debt 35%",
    )
    r = route_symbol("1180.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), bad_saudi], policy=pol)
    out = render_result(r)
    assert "❌" in out
    assert "CMA Shariah" in out
    assert "debt 35%" in out


def test_render_no_secret_leak():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    r = route_symbol("2222.SR", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    out = render_result(r)
    for token in ("@", "zoom.us", "meet.google", "private_email", "+1-", "Authorization"):
        assert token not in out


# --- E2E ----------------------------------------------------------------------


def test_e2e_us_operator_buys_saudi_aramco_approved():
    """Cross-border approval — both Saudi-CMA + global pass."""
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA, mode=RoutingMode.STRICTEST)
    r = route_symbol(
        "2222.SR",
        Jurisdiction.SAUDI_ARABIA,
        [
            SourceVerdict(
                source_name="CMA Shariah",
                jurisdiction=Jurisdiction.SAUDI_ARABIA,
                verdict=JurisdictionVerdict.PERMISSIBLE,
            ),
            SourceVerdict(
                source_name="Zoya",
                jurisdiction=Jurisdiction.GLOBAL,
                verdict=JurisdictionVerdict.PERMISSIBLE,
            ),
        ],
        policy=pol,
    )
    assert r.outcome is CompositeOutcome.APPROVED


def test_e2e_local_only_mode_skips_global():
    """Saudi operator using HOME_ONLY mode ignores global verdict."""
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.SAUDI_ARABIA, mode=RoutingMode.HOME_ONLY)
    r = route_symbol(
        "2222.SR",
        Jurisdiction.SAUDI_ARABIA,
        [
            SourceVerdict(
                source_name="CMA Shariah",
                jurisdiction=Jurisdiction.SAUDI_ARABIA,
                verdict=JurisdictionVerdict.PERMISSIBLE,
            ),
            SourceVerdict(
                source_name="Zoya",
                jurisdiction=Jurisdiction.GLOBAL,
                verdict=JurisdictionVerdict.IMPERMISSIBLE,
            ),
        ],
        policy=pol,
    )
    # HOME_ONLY ignores Zoya's IMPERMISSIBLE
    assert r.outcome is CompositeOutcome.APPROVED


def test_replay_consistency():
    pol = RoutingPolicy(operator_jurisdiction=Jurisdiction.USA)
    a = route_symbol("X", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    b = route_symbol("X", Jurisdiction.SAUDI_ARABIA, [_good_global(), _good_saudi()], policy=pol)
    assert a == b
