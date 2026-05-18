"""Tests for the halal options-strategy screener."""

from __future__ import annotations

import dataclasses

import pytest

from halal_trader.web.halal_options import (
    DEFAULT_POLICY,
    OptionsScreenPolicy,
    OptionsScreenRequest,
    OptionsScreenResult,
    OptionsScreenVerdict,
    OptionStrategy,
    render_screen_result,
    screen_options_strategy,
)


def _request(
    *,
    strategy: OptionStrategy = OptionStrategy.PROTECTIVE_PUT,
    underlying_symbol: str = "AAPL",
    underlying_is_halal_screened: bool = True,
) -> OptionsScreenRequest:
    return OptionsScreenRequest(
        strategy=strategy,
        underlying_symbol=underlying_symbol,
        underlying_is_halal_screened=underlying_is_halal_screened,
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_default_policy_no_ssb_approval() -> None:
    p = DEFAULT_POLICY
    assert p.ssb_ruling_id == ""
    assert p.permitted_with_ssb_ruling == frozenset()


def test_policy_rejects_strategies_without_ruling_id() -> None:
    """Pin: SSB-approved strategies require a citable ruling_id."""

    with pytest.raises(ValueError, match="requires ssb_ruling_id"):
        OptionsScreenPolicy(permitted_with_ssb_ruling=frozenset({OptionStrategy.COVERED_CALL}))


def test_policy_rejects_naked_call_in_permitted_set() -> None:
    """Pin: operator can't SSB-approve unconditional gharar strategies."""

    with pytest.raises(ValueError, match="gharar"):
        OptionsScreenPolicy(
            ssb_ruling_id="SSB-2026-Q1-001",
            permitted_with_ssb_ruling=frozenset({OptionStrategy.NAKED_CALL}),
        )


def test_policy_rejects_naked_put_in_permitted_set() -> None:
    with pytest.raises(ValueError, match="gharar"):
        OptionsScreenPolicy(
            ssb_ruling_id="SSB-2026-Q1-001",
            permitted_with_ssb_ruling=frozenset({OptionStrategy.NAKED_PUT}),
        )


def test_policy_rejects_iron_condor_in_permitted_set() -> None:
    """Pin: spreads with embedded short legs cannot be SSB-approved."""

    with pytest.raises(ValueError, match="gharar"):
        OptionsScreenPolicy(
            ssb_ruling_id="SSB-2026-Q1-001",
            permitted_with_ssb_ruling=frozenset({OptionStrategy.IRON_CONDOR}),
        )


def test_policy_rejects_butterfly_in_permitted_set() -> None:
    with pytest.raises(ValueError, match="gharar"):
        OptionsScreenPolicy(
            ssb_ruling_id="SSB-2026-Q1-001",
            permitted_with_ssb_ruling=frozenset({OptionStrategy.BUTTERFLY}),
        )


def test_policy_accepts_covered_call_with_ruling_id() -> None:
    """Pin: COVERED_CALL is in the debated set and CAN be SSB-approved."""

    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.COVERED_CALL}),
    )
    assert OptionStrategy.COVERED_CALL in p.permitted_with_ssb_ruling


def test_policy_accepts_cash_secured_put_with_ruling_id() -> None:
    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.CASH_SECURED_PUT}),
    )
    assert OptionStrategy.CASH_SECURED_PUT in p.permitted_with_ssb_ruling


def test_policy_accepts_multiple_debated_strategies() -> None:
    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset(
            {
                OptionStrategy.COVERED_CALL,
                OptionStrategy.CASH_SECURED_PUT,
            }
        ),
    )
    assert len(p.permitted_with_ssb_ruling) == 2


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_request_rejects_empty_underlying() -> None:
    with pytest.raises(ValueError, match="underlying_symbol"):
        _request(underlying_symbol="")


def test_request_rejects_whitespace_underlying() -> None:
    with pytest.raises(ValueError, match="underlying_symbol"):
        _request(underlying_symbol="   ")


# ---------------------------------------------------------------------------
# Underlying gate — non-halal-screened underlying blocks everything
# ---------------------------------------------------------------------------


def test_non_halal_underlying_blocks_protective_put() -> None:
    """Pin: even a HALAL strategy blocked when underlying not screened."""

    request = _request(
        strategy=OptionStrategy.PROTECTIVE_PUT,
        underlying_is_halal_screened=False,
    )
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL
    assert any("not halal-screened" in f for f in result.failures)


def test_non_halal_underlying_blocks_covered_call() -> None:
    request = _request(
        strategy=OptionStrategy.COVERED_CALL,
        underlying_is_halal_screened=False,
    )
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


def test_non_halal_underlying_blocks_long_call() -> None:
    request = _request(
        strategy=OptionStrategy.LONG_CALL,
        underlying_is_halal_screened=False,
    )
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Naked strategies — unconditional NOT_HALAL
# ---------------------------------------------------------------------------


def test_naked_call_is_not_halal() -> None:
    """Pin: naked call → unconditional gharar."""

    request = _request(strategy=OptionStrategy.NAKED_CALL)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL
    assert any("gharar" in f for f in result.failures)


def test_naked_put_is_not_halal() -> None:
    request = _request(strategy=OptionStrategy.NAKED_PUT)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL
    assert any("gharar" in f for f in result.failures)


def test_naked_call_blocked_even_with_halal_underlying() -> None:
    """Pin: even with halal underlying + AAOIFI ruling on debated set,
    naked legs cannot be approved."""

    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.COVERED_CALL}),
    )
    request = _request(strategy=OptionStrategy.NAKED_CALL)
    result = screen_options_strategy(request, policy=p)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# Multi-leg spreads — unconditional NOT_HALAL
# ---------------------------------------------------------------------------


def test_bull_call_spread_is_not_halal() -> None:
    request = _request(strategy=OptionStrategy.BULL_CALL_SPREAD)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL
    assert any("short leg" in f for f in result.failures)


def test_bear_put_spread_is_not_halal() -> None:
    request = _request(strategy=OptionStrategy.BEAR_PUT_SPREAD)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


def test_iron_condor_is_not_halal() -> None:
    request = _request(strategy=OptionStrategy.IRON_CONDOR)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


def test_butterfly_is_not_halal() -> None:
    request = _request(strategy=OptionStrategy.BUTTERFLY)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


def test_straddle_is_not_halal() -> None:
    """Pin: straddle = long call + long put on same underlying;
    one leg is short-equivalent in some classifications. Reject for
    operational simplicity — operators wanting the strategy can
    construct it as two separate longs explicitly."""

    request = _request(strategy=OptionStrategy.STRADDLE)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


def test_strangle_is_not_halal() -> None:
    request = _request(strategy=OptionStrategy.STRANGLE)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.NOT_HALAL


# ---------------------------------------------------------------------------
# PROTECTIVE_PUT — clean HALAL
# ---------------------------------------------------------------------------


def test_protective_put_is_halal() -> None:
    """Pin: insurance via long put = clean HALAL."""

    request = _request(strategy=OptionStrategy.PROTECTIVE_PUT)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.HALAL
    assert result.failures == ()
    assert result.warnings == ()


def test_protective_put_works_with_no_ssb_ruling() -> None:
    """Pin: protective put doesn't need SSB approval."""

    p = OptionsScreenPolicy()  # no ruling
    request = _request(strategy=OptionStrategy.PROTECTIVE_PUT)
    result = screen_options_strategy(request, policy=p)
    assert result.verdict is OptionsScreenVerdict.HALAL


# ---------------------------------------------------------------------------
# Debated strategies — DOUBTFUL by default
# ---------------------------------------------------------------------------


def test_covered_call_default_is_doubtful() -> None:
    """Pin: covered call is scholar-debated; default DOUBTFUL."""

    request = _request(strategy=OptionStrategy.COVERED_CALL)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.DOUBTFUL
    assert any("scholar-debated" in w for w in result.warnings)


def test_cash_secured_put_default_is_doubtful() -> None:
    request = _request(strategy=OptionStrategy.CASH_SECURED_PUT)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.DOUBTFUL


def test_long_call_default_is_doubtful() -> None:
    """Pin: long call is debated — buying the right is paying premium
    for uncertainty even though no naked-short leg."""

    request = _request(strategy=OptionStrategy.LONG_CALL)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.DOUBTFUL


def test_long_put_default_is_doubtful() -> None:
    request = _request(strategy=OptionStrategy.LONG_PUT)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.DOUBTFUL


# ---------------------------------------------------------------------------
# SSB-approved debated strategies — HALAL_WITH_CONDITIONS
# ---------------------------------------------------------------------------


def test_covered_call_with_ssb_approval_is_halal_with_conditions() -> None:
    """Pin: operator's SSB approves COVERED_CALL → HALAL_WITH_CONDITIONS."""

    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.COVERED_CALL}),
    )
    request = _request(strategy=OptionStrategy.COVERED_CALL)
    result = screen_options_strategy(request, policy=p)
    assert result.verdict is OptionsScreenVerdict.HALAL_WITH_CONDITIONS
    assert result.ssb_ruling_cited == "SSB-2026-Q1-001"
    assert any("conditions" in w.lower() for w in result.warnings)


def test_cash_secured_put_with_ssb_approval_is_halal_with_conditions() -> None:
    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q2-002",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.CASH_SECURED_PUT}),
    )
    request = _request(strategy=OptionStrategy.CASH_SECURED_PUT)
    result = screen_options_strategy(request, policy=p)
    assert result.verdict is OptionsScreenVerdict.HALAL_WITH_CONDITIONS
    assert result.ssb_ruling_cited == "SSB-2026-Q2-002"


def test_ssb_approval_for_one_strategy_does_not_extend_to_others() -> None:
    """Pin: operator approves COVERED_CALL but not CASH_SECURED_PUT —
    the latter remains DOUBTFUL."""

    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.COVERED_CALL}),
    )
    request = _request(strategy=OptionStrategy.CASH_SECURED_PUT)
    result = screen_options_strategy(request, policy=p)
    assert result.verdict is OptionsScreenVerdict.DOUBTFUL


# ---------------------------------------------------------------------------
# OTHER catchall
# ---------------------------------------------------------------------------


def test_other_strategy_is_unknown() -> None:
    """Pin: unrecognised strategy → UNKNOWN with classify-first warning."""

    request = _request(strategy=OptionStrategy.OTHER)
    result = screen_options_strategy(request)
    assert result.verdict is OptionsScreenVerdict.UNKNOWN
    assert any("classify" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_request_is_frozen() -> None:
    r = _request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.strategy = OptionStrategy.NAKED_CALL  # type: ignore[misc]


def test_screen_result_is_frozen() -> None:
    result = screen_options_strategy(_request())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.verdict = OptionsScreenVerdict.NOT_HALAL  # type: ignore[misc]


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_POLICY.ssb_ruling_id = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum string values pinned for JSON / DB stability
# ---------------------------------------------------------------------------


def test_strategy_string_values() -> None:
    assert OptionStrategy.LONG_CALL.value == "long_call"
    assert OptionStrategy.LONG_PUT.value == "long_put"
    assert OptionStrategy.NAKED_CALL.value == "naked_call"
    assert OptionStrategy.NAKED_PUT.value == "naked_put"
    assert OptionStrategy.COVERED_CALL.value == "covered_call"
    assert OptionStrategy.CASH_SECURED_PUT.value == "cash_secured_put"
    assert OptionStrategy.PROTECTIVE_PUT.value == "protective_put"
    assert OptionStrategy.BULL_CALL_SPREAD.value == "bull_call_spread"
    assert OptionStrategy.IRON_CONDOR.value == "iron_condor"


def test_verdict_string_values() -> None:
    assert OptionsScreenVerdict.HALAL.value == "halal"
    assert OptionsScreenVerdict.NOT_HALAL.value == "not_halal"
    assert OptionsScreenVerdict.DOUBTFUL.value == "doubtful"
    assert OptionsScreenVerdict.HALAL_WITH_CONDITIONS.value == "halal_with_conditions"
    assert OptionsScreenVerdict.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------


def test_render_halal_protective_put() -> None:
    result = screen_options_strategy(_request())
    text = render_screen_result(result)
    assert "✅" in text
    assert "AAPL" in text
    assert "protective_put" in text
    assert "HALAL" in text


def test_render_not_halal_naked_call() -> None:
    result = screen_options_strategy(_request(strategy=OptionStrategy.NAKED_CALL))
    text = render_screen_result(result)
    assert "❌" in text
    assert "NOT_HALAL" in text
    assert "failures:" in text


def test_render_doubtful_covered_call() -> None:
    result = screen_options_strategy(_request(strategy=OptionStrategy.COVERED_CALL))
    text = render_screen_result(result)
    assert "⚠️" in text
    assert "DOUBTFUL" in text
    assert "warnings:" in text


def test_render_halal_with_conditions_includes_ruling_id() -> None:
    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset({OptionStrategy.COVERED_CALL}),
    )
    result = screen_options_strategy(_request(strategy=OptionStrategy.COVERED_CALL), policy=p)
    text = render_screen_result(result)
    assert "📋" in text
    assert "HALAL_WITH_CONDITIONS" in text
    assert "SSB-2026-Q1-001" in text


def test_render_unknown_strategy() -> None:
    result = screen_options_strategy(_request(strategy=OptionStrategy.OTHER))
    text = render_screen_result(result)
    assert "❓" in text
    assert "UNKNOWN" in text


def test_render_no_strike_or_expiry() -> None:
    """Pin: render never includes strike prices or expiry dates."""

    result = screen_options_strategy(_request())
    text = render_screen_result(result)
    # The engine doesn't hold strike/expiry fields, so the render
    # naturally can't include them — pinned via test
    assert "strike" not in text.lower()
    assert "expiry" not in text.lower()
    assert "expires" not in text.lower()


# ---------------------------------------------------------------------------
# Closed-set guarantee
# ---------------------------------------------------------------------------


def test_naked_strategies_set_complete() -> None:
    """Pin: both NAKED_CALL and NAKED_PUT are categorically blocked."""

    for s in (OptionStrategy.NAKED_CALL, OptionStrategy.NAKED_PUT):
        result = screen_options_strategy(_request(strategy=s))
        assert result.verdict is OptionsScreenVerdict.NOT_HALAL


def test_spread_strategies_set_complete() -> None:
    """Pin: every multi-leg spread strategy categorically blocked."""

    for s in (
        OptionStrategy.BULL_CALL_SPREAD,
        OptionStrategy.BEAR_PUT_SPREAD,
        OptionStrategy.IRON_CONDOR,
        OptionStrategy.BUTTERFLY,
        OptionStrategy.STRADDLE,
        OptionStrategy.STRANGLE,
    ):
        result = screen_options_strategy(_request(strategy=s))
        assert result.verdict is OptionsScreenVerdict.NOT_HALAL, s


def test_debated_strategies_default_doubtful() -> None:
    """Pin: every debated strategy defaults to DOUBTFUL."""

    for s in (
        OptionStrategy.COVERED_CALL,
        OptionStrategy.CASH_SECURED_PUT,
        OptionStrategy.LONG_CALL,
        OptionStrategy.LONG_PUT,
    ):
        result = screen_options_strategy(_request(strategy=s))
        assert result.verdict is OptionsScreenVerdict.DOUBTFUL, s


# ---------------------------------------------------------------------------
# End-to-end realistic scenarios
# ---------------------------------------------------------------------------


def test_typical_protective_put_journey() -> None:
    """Operator owns AAPL; buys a protective put for downside hedge."""

    result = screen_options_strategy(_request(strategy=OptionStrategy.PROTECTIVE_PUT))
    assert result.verdict is OptionsScreenVerdict.HALAL


def test_covered_call_with_ssb_ruling_journey() -> None:
    """Operator owns AAPL; SSB ruling SSB-2026-Q1-001 permits covered
    calls with conditions; operator writes covered call →
    HALAL_WITH_CONDITIONS."""

    p = OptionsScreenPolicy(
        ssb_ruling_id="SSB-2026-Q1-001",
        permitted_with_ssb_ruling=frozenset(
            {
                OptionStrategy.COVERED_CALL,
                OptionStrategy.CASH_SECURED_PUT,
            }
        ),
    )
    result = screen_options_strategy(_request(strategy=OptionStrategy.COVERED_CALL), policy=p)
    assert result.verdict is OptionsScreenVerdict.HALAL_WITH_CONDITIONS
    assert result.ssb_ruling_cited == "SSB-2026-Q1-001"


def test_attempted_iron_condor_blocked_via_policy_construction() -> None:
    """Pin: even an operator with an SSB ruling that *tries* to approve
    iron condor — the policy construction itself rejects."""

    with pytest.raises(ValueError, match="gharar"):
        OptionsScreenPolicy(
            ssb_ruling_id="SSB-2026-Q1-001",
            permitted_with_ssb_ruling=frozenset({OptionStrategy.IRON_CONDOR}),
        )


def test_screen_result_carries_strategy_and_underlying() -> None:
    """Pin: result preserves the input strategy + underlying for audit."""

    result = screen_options_strategy(
        _request(
            strategy=OptionStrategy.PROTECTIVE_PUT,
            underlying_symbol="MSFT",
        )
    )
    assert isinstance(result, OptionsScreenResult)
    assert result.strategy is OptionStrategy.PROTECTIVE_PUT
    assert result.underlying_symbol == "MSFT"
