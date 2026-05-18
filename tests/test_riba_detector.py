"""Tests for the riba detector across derivative instrument classes."""

from __future__ import annotations

import pytest

from halal_trader.halal.riba_detector import (
    HALAL_BY_DEFAULT_CLASSES,
    InstrumentClass,
    RibaAssessment,
    RibaInputs,
    RibaPolicy,
    RibaType,
    assess_batch,
    assess_riba,
    filter_blocked,
    render_assessment,
)

# --- Enum string-value pins ---------------------------------------------------


def test_riba_type_string_values():
    assert RibaType.NASIYAH.value == "nasiyah"
    assert RibaType.FADL.value == "fadl"
    assert RibaType.EMBEDDED_FINANCING.value == "embedded_financing"
    assert RibaType.DEBT_SALE.value == "debt_sale"
    assert RibaType.LEVERAGE_INTEREST.value == "leverage_interest"


def test_instrument_class_string_values():
    assert InstrumentClass.SPOT_EQUITY.value == "spot_equity"
    assert InstrumentClass.PHYSICAL_COMMODITY.value == "physical_commodity"
    assert InstrumentClass.SUKUK.value == "sukuk"
    assert InstrumentClass.WAAD_FORWARD.value == "waad_forward"
    assert InstrumentClass.SALAM_FORWARD.value == "salam_forward"
    assert InstrumentClass.ARBOUN_OPTION.value == "arboun_option"
    assert InstrumentClass.CONVENTIONAL_BOND.value == "conventional_bond"
    assert InstrumentClass.CONVENTIONAL_FUTURE.value == "conventional_future"
    assert InstrumentClass.CONVENTIONAL_OPTION.value == "conventional_option"
    assert InstrumentClass.INTEREST_RATE_SWAP.value == "interest_rate_swap"
    assert InstrumentClass.CURRENCY_SWAP.value == "currency_swap"
    assert InstrumentClass.CFD.value == "cfd"
    assert InstrumentClass.LEVERAGED_ETF.value == "leveraged_etf"
    assert InstrumentClass.INVERSE_ETF.value == "inverse_etf"


# --- HALAL_BY_DEFAULT_CLASSES module-level set --------------------------------


def test_halal_by_default_set_pin():
    """Pin: exactly the 6 classes whose base verdict is empty."""
    expected = frozenset(
        {
            InstrumentClass.SPOT_EQUITY,
            InstrumentClass.PHYSICAL_COMMODITY,
            InstrumentClass.SUKUK,
            InstrumentClass.WAAD_FORWARD,
            InstrumentClass.SALAM_FORWARD,
            InstrumentClass.ARBOUN_OPTION,
        }
    )
    assert HALAL_BY_DEFAULT_CLASSES == expected


def test_halal_by_default_excludes_conventional():
    """Pin: conventional bond/future/swap/CFD never in halal-by-default."""
    forbidden = {
        InstrumentClass.CONVENTIONAL_BOND,
        InstrumentClass.CONVENTIONAL_FUTURE,
        InstrumentClass.CONVENTIONAL_OPTION,
        InstrumentClass.INTEREST_RATE_SWAP,
        InstrumentClass.CURRENCY_SWAP,
        InstrumentClass.CFD,
        InstrumentClass.LEVERAGED_ETF,
        InstrumentClass.INVERSE_ETF,
    }
    for cls in forbidden:
        assert cls not in HALAL_BY_DEFAULT_CLASSES


# --- Policy validation --------------------------------------------------------


def test_default_policy_pins():
    p = RibaPolicy()
    assert p.flag_margin_as_riba is True
    assert p.flag_borrow_as_riba is True


def test_policy_immutable():
    p = RibaPolicy()
    with pytest.raises(Exception):
        p.flag_margin_as_riba = False  # type: ignore[misc]


# --- RibaInputs validation ---------------------------------------------------


def test_empty_id_rejected():
    with pytest.raises(ValueError, match="instrument_id"):
        RibaInputs(instrument_id="", instrument_class=InstrumentClass.SPOT_EQUITY)


def test_inputs_immutable():
    i = RibaInputs(instrument_id="X", instrument_class=InstrumentClass.SPOT_EQUITY)
    with pytest.raises(Exception):
        i.uses_margin = True  # type: ignore[misc]


# --- Halal-by-default instrument classes (no riba unless flags) --------------


def test_spot_equity_clean():
    inputs = RibaInputs(instrument_id="AAPL", instrument_class=InstrumentClass.SPOT_EQUITY)
    a = assess_riba(inputs)
    assert a.is_clean
    assert a.riba_types == frozenset()


def test_physical_commodity_clean():
    inputs = RibaInputs(
        instrument_id="GOLD_PHYSICAL",
        instrument_class=InstrumentClass.PHYSICAL_COMMODITY,
    )
    a = assess_riba(inputs)
    assert a.is_clean


def test_sukuk_clean():
    inputs = RibaInputs(instrument_id="SAUDI_SUKUK_2030", instrument_class=InstrumentClass.SUKUK)
    a = assess_riba(inputs)
    assert a.is_clean


def test_waad_forward_clean():
    inputs = RibaInputs(
        instrument_id="WAAD_AAPL_30D", instrument_class=InstrumentClass.WAAD_FORWARD
    )
    a = assess_riba(inputs)
    assert a.is_clean


def test_salam_forward_clean():
    inputs = RibaInputs(
        instrument_id="SALAM_WHEAT_90D", instrument_class=InstrumentClass.SALAM_FORWARD
    )
    a = assess_riba(inputs)
    assert a.is_clean


def test_arboun_option_clean():
    inputs = RibaInputs(
        instrument_id="ARBOUN_AAPL_C200_30D",
        instrument_class=InstrumentClass.ARBOUN_OPTION,
    )
    a = assess_riba(inputs)
    assert a.is_clean


# --- Conventional / non-halal instrument classes (always flagged) ------------


def test_conventional_bond_flags_nasiyah():
    """Pin: conventional bond ALWAYS carries NASIYAH."""
    inputs = RibaInputs(instrument_id="UST_10Y", instrument_class=InstrumentClass.CONVENTIONAL_BOND)
    a = assess_riba(inputs)
    assert RibaType.NASIYAH in a.riba_types
    assert not a.is_clean


def test_conventional_future_flags_embedded_financing():
    inputs = RibaInputs(
        instrument_id="ES_DEC_FUTURE", instrument_class=InstrumentClass.CONVENTIONAL_FUTURE
    )
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types


def test_conventional_option_flags_embedded_financing():
    inputs = RibaInputs(
        instrument_id="AAPL_C200_30D",
        instrument_class=InstrumentClass.CONVENTIONAL_OPTION,
    )
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types


def test_interest_rate_swap_flags_nasiyah():
    inputs = RibaInputs(
        instrument_id="USD_LIBOR_5Y", instrument_class=InstrumentClass.INTEREST_RATE_SWAP
    )
    a = assess_riba(inputs)
    assert RibaType.NASIYAH in a.riba_types


def test_currency_swap_flags_embedded_financing():
    inputs = RibaInputs(instrument_id="USD_EUR_5Y", instrument_class=InstrumentClass.CURRENCY_SWAP)
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types


def test_cfd_flags_both_embedded_and_leverage():
    """Pin: CFDs carry BOTH embedded financing AND leverage interest."""
    inputs = RibaInputs(instrument_id="CFD_AAPL", instrument_class=InstrumentClass.CFD)
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types
    assert RibaType.LEVERAGE_INTEREST in a.riba_types


def test_leveraged_etf_flags_leverage():
    inputs = RibaInputs(instrument_id="TQQQ", instrument_class=InstrumentClass.LEVERAGED_ETF)
    a = assess_riba(inputs)
    assert RibaType.LEVERAGE_INTEREST in a.riba_types


def test_inverse_etf_flags_embedded_financing():
    inputs = RibaInputs(instrument_id="SQQQ", instrument_class=InstrumentClass.INVERSE_ETF)
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types


# --- Operator flags add to base verdict --------------------------------------


def test_margin_flag_adds_leverage_interest_to_clean_class():
    """Pin: SPOT_EQUITY with margin → adds LEVERAGE_INTEREST."""
    inputs = RibaInputs(
        instrument_id="AAPL_MARGIN",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        uses_margin=True,
    )
    a = assess_riba(inputs)
    assert RibaType.LEVERAGE_INTEREST in a.riba_types
    assert not a.is_clean


def test_borrow_flag_adds_nasiyah_to_clean_class():
    """Pin: SPOT_EQUITY with borrowed securities → adds NASIYAH (lending fee)."""
    inputs = RibaInputs(
        instrument_id="AAPL_BORROW",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        uses_borrowed_securities=True,
    )
    a = assess_riba(inputs)
    assert RibaType.NASIYAH in a.riba_types


def test_embedded_financing_flag_adds_to_clean_class():
    inputs = RibaInputs(
        instrument_id="WEIRD_SUKUK",
        instrument_class=InstrumentClass.SUKUK,
        has_embedded_financing_rate=True,
    )
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types


def test_fixed_interest_flag_adds_nasiyah():
    inputs = RibaInputs(
        instrument_id="X",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        pays_or_receives_fixed_interest=True,
    )
    a = assess_riba(inputs)
    assert RibaType.NASIYAH in a.riba_types


def test_debt_sale_flag_adds_debt_sale():
    inputs = RibaInputs(
        instrument_id="X",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        is_debt_traded_off_face=True,
    )
    a = assess_riba(inputs)
    assert RibaType.DEBT_SALE in a.riba_types


# --- Policy override (rare permissive operator) ------------------------------


def test_disable_margin_flag_keeps_clean():
    """Pin: operator with flag_margin_as_riba=False can use margin
    (rare permissive scholar position)."""
    permissive = RibaPolicy(flag_margin_as_riba=False)
    inputs = RibaInputs(
        instrument_id="X",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        uses_margin=True,
    )
    a = assess_riba(inputs, policy=permissive)
    assert a.is_clean


def test_disable_borrow_flag_keeps_clean():
    permissive = RibaPolicy(flag_borrow_as_riba=False)
    inputs = RibaInputs(
        instrument_id="X",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        uses_borrowed_securities=True,
    )
    a = assess_riba(inputs, policy=permissive)
    assert a.is_clean


def test_policy_cannot_clear_base_verdict():
    """Pin: even fully-permissive policy cannot remove a class's base
    verdict — fiqh determination, not configuration."""
    permissive = RibaPolicy(flag_margin_as_riba=False, flag_borrow_as_riba=False)
    inputs = RibaInputs(
        instrument_id="UST",
        instrument_class=InstrumentClass.CONVENTIONAL_BOND,
    )
    a = assess_riba(inputs, policy=permissive)
    # Conventional bond ALWAYS NASIYAH regardless of policy
    assert RibaType.NASIYAH in a.riba_types


# --- Combined: base + operator flags merge ----------------------------------


def test_conventional_future_with_margin_combines_types():
    """Pin: base EMBEDDED_FINANCING + flag-driven LEVERAGE_INTEREST both fire."""
    inputs = RibaInputs(
        instrument_id="ES_LEVERED",
        instrument_class=InstrumentClass.CONVENTIONAL_FUTURE,
        uses_margin=True,
    )
    a = assess_riba(inputs)
    assert RibaType.EMBEDDED_FINANCING in a.riba_types
    assert RibaType.LEVERAGE_INTEREST in a.riba_types


# --- Assessment validation ---------------------------------------------------


def test_assessment_immutable():
    a = assess_riba(RibaInputs(instrument_id="X", instrument_class=InstrumentClass.SPOT_EQUITY))
    with pytest.raises(Exception):
        a.riba_types = frozenset({RibaType.NASIYAH})  # type: ignore[misc]


def test_assessment_empty_id_rejected():
    with pytest.raises(ValueError, match="instrument_id"):
        RibaAssessment(
            instrument_id="",
            instrument_class=InstrumentClass.SPOT_EQUITY,
            riba_types=frozenset(),
        )


def test_is_clean_property():
    clean = RibaAssessment(
        instrument_id="X",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        riba_types=frozenset(),
    )
    assert clean.is_clean is True
    not_clean = RibaAssessment(
        instrument_id="X",
        instrument_class=InstrumentClass.SPOT_EQUITY,
        riba_types=frozenset({RibaType.NASIYAH}),
    )
    assert not_clean.is_clean is False


# --- Batch + filter -----------------------------------------------------------


def test_batch_sorted_by_id():
    inputs = [
        RibaInputs(instrument_id="ZZZ", instrument_class=InstrumentClass.SPOT_EQUITY),
        RibaInputs(instrument_id="AAA", instrument_class=InstrumentClass.SPOT_EQUITY),
    ]
    result = assess_batch(inputs)
    assert [a.instrument_id for a in result] == ["AAA", "ZZZ"]


def test_batch_empty():
    assert assess_batch([]) == ()


def test_filter_blocked_returns_only_dirty():
    clean = assess_riba(
        RibaInputs(instrument_id="AAPL", instrument_class=InstrumentClass.SPOT_EQUITY)
    )
    dirty = assess_riba(
        RibaInputs(instrument_id="UST", instrument_class=InstrumentClass.CONVENTIONAL_BOND)
    )
    blocked = filter_blocked([clean, dirty])
    assert len(blocked) == 1
    assert blocked[0].instrument_id == "UST"


# --- Render -------------------------------------------------------------------


def test_render_clean_shows_check():
    a = assess_riba(RibaInputs(instrument_id="AAPL", instrument_class=InstrumentClass.SPOT_EQUITY))
    out = render_assessment(a)
    assert "✅" in out
    assert "AAPL" in out
    assert "spot equity" in out
    assert "no riba detected" in out


def test_render_dirty_shows_x_and_riba_labels():
    a = assess_riba(
        RibaInputs(instrument_id="UST", instrument_class=InstrumentClass.CONVENTIONAL_BOND)
    )
    out = render_assessment(a)
    assert "❌" in out
    assert "conventional bond" in out
    assert "riba al-nasiyah" in out


def test_render_includes_class_label():
    a = assess_riba(RibaInputs(instrument_id="X", instrument_class=InstrumentClass.WAAD_FORWARD))
    out = render_assessment(a)
    assert "wa'd forward" in out


def test_render_riba_types_sorted():
    """Pin: render labels alphabetically sorted."""
    a = assess_riba(
        RibaInputs(
            instrument_id="CFD_AAPL",
            instrument_class=InstrumentClass.CFD,
        )
    )
    out = render_assessment(a)
    # "embedded financing" < "leverage interest" alphabetically
    assert out.index("embedded financing") < out.index("leverage interest")


def test_render_no_secret_leak():
    """Pin: render output never includes broker financing schedules / counterparty terms."""
    a = assess_riba(RibaInputs(instrument_id="CFD_AAPL", instrument_class=InstrumentClass.CFD))
    out = render_assessment(a)
    forbidden = [
        "financing_schedule",
        "counterparty_id",
        "/api/",
        "Authorization",
        "Bearer",
        "bps",
    ]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_classic_halal_portfolio_all_clean():
    """A halal portfolio: spot equities + sukuk + physical gold + wa'd forward."""
    inputs = [
        RibaInputs(instrument_id="AAPL", instrument_class=InstrumentClass.SPOT_EQUITY),
        RibaInputs(instrument_id="MSFT", instrument_class=InstrumentClass.SPOT_EQUITY),
        RibaInputs(instrument_id="SAUDI_SUKUK_2030", instrument_class=InstrumentClass.SUKUK),
        RibaInputs(instrument_id="GOLD_VAULT", instrument_class=InstrumentClass.PHYSICAL_COMMODITY),
        RibaInputs(instrument_id="WAAD_AAPL_30D", instrument_class=InstrumentClass.WAAD_FORWARD),
    ]
    results = assess_batch(inputs)
    for a in results:
        assert a.is_clean
    assert len(filter_blocked(results)) == 0


def test_e2e_classic_non_halal_portfolio_all_blocked():
    """Conventional broker bundle: stocks-on-margin + bonds + CFDs + futures."""
    inputs = [
        RibaInputs(
            instrument_id="AAPL_MARGIN",
            instrument_class=InstrumentClass.SPOT_EQUITY,
            uses_margin=True,
        ),
        RibaInputs(instrument_id="UST_10Y", instrument_class=InstrumentClass.CONVENTIONAL_BOND),
        RibaInputs(instrument_id="CFD_AAPL", instrument_class=InstrumentClass.CFD),
        RibaInputs(
            instrument_id="ES_FUTURE",
            instrument_class=InstrumentClass.CONVENTIONAL_FUTURE,
        ),
        RibaInputs(instrument_id="TQQQ", instrument_class=InstrumentClass.LEVERAGED_ETF),
    ]
    results = assess_batch(inputs)
    for a in results:
        assert not a.is_clean
    assert len(filter_blocked(results)) == 5


def test_e2e_replay_consistency():
    """Pin: same inputs → equal assessment."""
    inputs = RibaInputs(
        instrument_id="X",
        instrument_class=InstrumentClass.CFD,
        uses_margin=True,
    )
    a1 = assess_riba(inputs)
    a2 = assess_riba(inputs)
    assert a1 == a2


def test_e2e_aaoifi_void_contract_pin():
    """AAOIFI Standard 21 + Standard 30 reference: an interest-rate swap
    is the canonical riba al-nasiyah instrument; render it
    appropriately."""
    inputs = RibaInputs(
        instrument_id="USD_LIBOR_SWAP_5Y",
        instrument_class=InstrumentClass.INTEREST_RATE_SWAP,
        pays_or_receives_fixed_interest=True,  # redundantly true
    )
    a = assess_riba(inputs)
    assert RibaType.NASIYAH in a.riba_types
    assert not a.is_clean
    # Render should clearly identify the riba violation
    out = render_assessment(a)
    assert "interest rate swap" in out
    assert "riba al-nasiyah" in out
