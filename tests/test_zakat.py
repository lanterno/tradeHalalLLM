"""Tests for the multi-currency Zakat calculator."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.halal.zakat import (
    DEFAULT_GOLD_NISAB_GRAMS,
    DEFAULT_SILVER_NISAB_GRAMS,
    DEFAULT_ZAKAT_RATE,
    LUNAR_YEAR_DAYS,
    FxRates,
    NisabBasis,
    ZakatCalculation,
    ZakatInputs,
    ZakatPolicy,
    calculate_zakat,
    days_until_hawl,
    render_calculation,
)


def _usd_fx(**overrides) -> FxRates:
    """Default FX rates: USD base, with SAR/EUR/GBP rates + metal prices."""
    base = {
        "base_currency": "USD",
        "rates": {
            "SAR": 0.27,  # 1 SAR = $0.27
            "EUR": 1.10,  # 1 EUR = $1.10
            "GBP": 1.25,  # 1 GBP = $1.25
            "AED": 0.27,  # 1 AED = $0.27
        },
        "gold_price_per_gram": 75.0,  # ~$75/g (~$2330/oz)
        "silver_price_per_gram": 0.95,  # ~$0.95/g
    }
    base.update(overrides)
    return FxRates(**base)


# --- Module constants ---------------------------------------------------------


def test_default_constants_pin():
    assert DEFAULT_GOLD_NISAB_GRAMS == 87.48
    assert DEFAULT_SILVER_NISAB_GRAMS == 612.36
    assert DEFAULT_ZAKAT_RATE == 0.025
    assert LUNAR_YEAR_DAYS == 354


# --- NisabBasis enum ----------------------------------------------------------


def test_nisab_basis_string_values():
    assert NisabBasis.GOLD.value == "gold"
    assert NisabBasis.SILVER.value == "silver"


# --- ZakatPolicy validation ---------------------------------------------------


def test_default_policy_pins():
    p = ZakatPolicy()
    assert p.zakat_rate == 0.025
    assert p.gold_nisab_grams == 87.48
    assert p.silver_nisab_grams == 612.36
    assert p.lunar_year_days == 354


def test_zero_zakat_rate_rejected():
    with pytest.raises(ValueError, match="zakat_rate"):
        ZakatPolicy(zakat_rate=0)


def test_zakat_rate_above_one_rejected():
    with pytest.raises(ValueError, match="zakat_rate"):
        ZakatPolicy(zakat_rate=1.1)


def test_zakat_rate_at_one_allowed():
    """Pin: rate=1.0 (100%) is the upper bound (Khums is 0.20)."""
    p = ZakatPolicy(zakat_rate=1.0)
    assert p.zakat_rate == 1.0


def test_zero_gold_nisab_rejected():
    with pytest.raises(ValueError, match="gold_nisab_grams"):
        ZakatPolicy(gold_nisab_grams=0)


def test_zero_silver_nisab_rejected():
    with pytest.raises(ValueError, match="silver_nisab_grams"):
        ZakatPolicy(silver_nisab_grams=0)


def test_zero_lunar_year_days_rejected():
    with pytest.raises(ValueError, match="lunar_year_days"):
        ZakatPolicy(lunar_year_days=0)


def test_policy_immutable():
    p = ZakatPolicy()
    with pytest.raises(Exception):
        p.zakat_rate = 0.05  # type: ignore[misc]


# --- FxRates validation -------------------------------------------------------


def test_fx_rates_basic():
    fx = _usd_fx()
    assert fx.base_currency == "USD"
    assert fx.rates["SAR"] == 0.27


def test_empty_base_currency_rejected():
    with pytest.raises(ValueError, match="base_currency"):
        FxRates(
            base_currency="",
            rates={},
            gold_price_per_gram=75.0,
            silver_price_per_gram=0.95,
        )


def test_zero_gold_price_rejected():
    with pytest.raises(ValueError, match="gold_price_per_gram"):
        _usd_fx(gold_price_per_gram=0)


def test_zero_silver_price_rejected():
    with pytest.raises(ValueError, match="silver_price_per_gram"):
        _usd_fx(silver_price_per_gram=0)


def test_zero_rate_rejected():
    with pytest.raises(ValueError, match="rate for"):
        _usd_fx(rates={"SAR": 0})


def test_negative_rate_rejected():
    with pytest.raises(ValueError, match="rate for"):
        _usd_fx(rates={"SAR": -0.5})


def test_empty_currency_code_rejected():
    with pytest.raises(ValueError, match="currency code"):
        _usd_fx(rates={"": 0.5})


def test_to_base_returns_amount_for_base():
    fx = _usd_fx()
    assert fx.to_base(100, "USD") == 100


def test_to_base_converts_other_currency():
    fx = _usd_fx()
    assert fx.to_base(100, "SAR") == pytest.approx(27.0)


def test_to_base_unknown_currency_raises():
    fx = _usd_fx()
    with pytest.raises(KeyError):
        fx.to_base(100, "XYZ")


# --- ZakatInputs validation ---------------------------------------------------


def test_inputs_default_empty():
    i = ZakatInputs()
    assert i.cash_by_currency == {}
    assert i.gold_grams == 0.0
    assert i.reporting_currency == "USD"


def test_empty_reporting_currency_rejected():
    with pytest.raises(ValueError, match="reporting_currency"):
        ZakatInputs(reporting_currency="")


def test_negative_gold_rejected():
    with pytest.raises(ValueError, match="gold_grams"):
        ZakatInputs(gold_grams=-1)


def test_negative_silver_rejected():
    with pytest.raises(ValueError, match="silver_grams"):
        ZakatInputs(silver_grams=-1)


def test_negative_cash_amount_rejected():
    with pytest.raises(ValueError, match="cash_by_currency"):
        ZakatInputs(cash_by_currency={"USD": -100})


def test_empty_currency_in_cash_rejected():
    with pytest.raises(ValueError, match="cash_by_currency"):
        ZakatInputs(cash_by_currency={"": 100})


def test_negative_investment_rejected():
    with pytest.raises(ValueError, match="investments_by_currency"):
        ZakatInputs(investments_by_currency={"USD": -100})


def test_negative_debts_owed_rejected():
    with pytest.raises(ValueError, match="debts_owed_by_user"):
        ZakatInputs(debts_owed_by_user_by_currency={"USD": -100})


def test_inputs_immutable():
    i = ZakatInputs()
    with pytest.raises(Exception):
        i.gold_grams = 99  # type: ignore[misc]


# --- ZakatCalculation validation ---------------------------------------------


def test_calculation_negative_zakat_rejected():
    with pytest.raises(ValueError, match="zakat_owed"):
        ZakatCalculation(
            net_assets=1000,
            nisab_value=500,
            meets_nisab=True,
            zakat_owed=-1,
            basis_used=NisabBasis.GOLD,
            reporting_currency="USD",
            hawl_due_date=None,
        )


def test_calculation_zero_nisab_rejected():
    with pytest.raises(ValueError, match="nisab_value"):
        ZakatCalculation(
            net_assets=1000,
            nisab_value=0,
            meets_nisab=False,
            zakat_owed=0,
            basis_used=NisabBasis.GOLD,
            reporting_currency="USD",
            hawl_due_date=None,
        )


def test_calculation_meets_nisab_with_zero_zakat_inconsistent():
    """Pin: meets_nisab=True with zakat_owed=0 (and assets>0) is invalid."""
    with pytest.raises(ValueError, match="inconsistent"):
        ZakatCalculation(
            net_assets=10000,
            nisab_value=500,
            meets_nisab=True,
            zakat_owed=0,
            basis_used=NisabBasis.GOLD,
            reporting_currency="USD",
            hawl_due_date=None,
        )


def test_calculation_below_nisab_with_zakat_inconsistent():
    """Pin: meets_nisab=False with zakat_owed>0 is invalid."""
    with pytest.raises(ValueError, match="inconsistent"):
        ZakatCalculation(
            net_assets=100,
            nisab_value=500,
            meets_nisab=False,
            zakat_owed=10,
            basis_used=NisabBasis.GOLD,
            reporting_currency="USD",
            hawl_due_date=None,
        )


def test_calculation_immutable():
    c = ZakatCalculation(
        net_assets=10000,
        nisab_value=500,
        meets_nisab=True,
        zakat_owed=250,
        basis_used=NisabBasis.GOLD,
        reporting_currency="USD",
        hawl_due_date=None,
    )
    with pytest.raises(Exception):
        c.zakat_owed = 999  # type: ignore[misc]


# --- calculate_zakat: simple paths --------------------------------------------


def test_zero_assets_below_nisab():
    inputs = ZakatInputs()
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.net_assets == 0
    assert c.meets_nisab is False
    assert c.zakat_owed == 0


def test_simple_cash_above_silver_nisab():
    """$10000 cash, silver-nisab ~ 612g * $0.95 = $581.74 → meets nisab."""
    inputs = ZakatInputs(cash_by_currency={"USD": 10000})
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    assert c.meets_nisab is True
    assert c.net_assets == 10000
    assert c.nisab_value == pytest.approx(612.36 * 0.95)
    assert c.zakat_owed == pytest.approx(250.0)


def test_simple_cash_above_gold_nisab():
    """$10000 cash, gold-nisab ~ 87.48g * $75 = $6561 → meets nisab."""
    inputs = ZakatInputs(cash_by_currency={"USD": 10000})
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.GOLD)
    assert c.meets_nisab is True
    assert c.zakat_owed == pytest.approx(250.0)


def test_below_silver_nisab():
    """$500 cash falls below silver nisab (~$582)."""
    inputs = ZakatInputs(cash_by_currency={"USD": 500})
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    assert c.meets_nisab is False
    assert c.zakat_owed == 0


def test_default_basis_is_silver():
    """Pin: default basis is SILVER (more conservative)."""
    inputs = ZakatInputs(cash_by_currency={"USD": 1000})
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)  # no basis kwarg
    assert c.basis_used is NisabBasis.SILVER


def test_silver_more_conservative_than_gold():
    """At default prices, silver-nisab is below gold-nisab."""
    inputs = ZakatInputs(cash_by_currency={"USD": 5000})
    fx = _usd_fx()
    silver_c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    gold_c = calculate_zakat(inputs, fx, basis=NisabBasis.GOLD)
    # $5000 > $582 silver-nisab → meets
    assert silver_c.meets_nisab is True
    # $5000 < $6561 gold-nisab → doesn't meet
    assert gold_c.meets_nisab is False


# --- Multi-currency netting --------------------------------------------------


def test_multi_currency_summing():
    """USD 1000 + SAR 5000 (=$1350) + EUR 500 (=$550) = $2900."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 1000, "SAR": 5000, "EUR": 500},
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    assert c.net_assets == pytest.approx(1000 + 5000 * 0.27 + 500 * 1.10)
    assert c.zakat_owed == pytest.approx(c.net_assets * 0.025)


def test_investments_added():
    inputs = ZakatInputs(
        cash_by_currency={"USD": 1000},
        investments_by_currency={"USD": 5000},
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.net_assets == 6000


def test_gold_holdings_priced():
    """100g gold * $75/g = $7500."""
    inputs = ZakatInputs(gold_grams=100)
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    assert c.net_assets == pytest.approx(7500.0)
    assert c.zakat_owed == pytest.approx(187.5)  # 7500 * 0.025


def test_silver_holdings_priced():
    inputs = ZakatInputs(silver_grams=1000)
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    assert c.net_assets == pytest.approx(950.0)


def test_loans_to_user_added():
    """Money you've lent to others counts as an asset."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 1000},
        debts_owed_to_user_by_currency={"USD": 500},
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.net_assets == 1500


def test_debts_owed_subtracted():
    """Money you owe is subtracted from net assets."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000},
        debts_owed_by_user_by_currency={"USD": 3000},
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.net_assets == 7000


def test_debts_can_drop_below_nisab():
    """Pin: high debts can pull net below nisab → no zakat owed."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 800},
        debts_owed_by_user_by_currency={"USD": 500},
    )
    fx = _usd_fx()
    # Net = 300; below silver nisab ~$582
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    assert c.net_assets == 300
    assert c.meets_nisab is False
    assert c.zakat_owed == 0


def test_debts_can_make_net_negative():
    """Net negative is allowed; zakat owed is 0."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 1000},
        debts_owed_by_user_by_currency={"USD": 5000},
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.net_assets == -4000
    assert c.meets_nisab is False
    assert c.zakat_owed == 0


# --- Reporting-currency / FX validation --------------------------------------


def test_reporting_currency_must_match_fx_base():
    """Pin: inputs.reporting_currency must match fx.base_currency."""
    inputs = ZakatInputs(
        reporting_currency="EUR",
        cash_by_currency={"USD": 1000},
    )
    fx = _usd_fx()  # USD base
    with pytest.raises(ValueError, match="base_currency"):
        calculate_zakat(inputs, fx)


def test_unknown_currency_raises_keyerror():
    inputs = ZakatInputs(cash_by_currency={"XYZ": 1000})
    fx = _usd_fx()
    with pytest.raises(KeyError):
        calculate_zakat(inputs, fx)


# --- Hawl date computation ---------------------------------------------------


def test_hawl_due_date_354_days_after_start():
    """Pin: lunar year is 354 days from hawl_start_date."""
    start = date(2026, 1, 1)
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000},
        hawl_start_date=start,
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.hawl_due_date == date(2026, 12, 21)  # 2026-01-01 + 354 days


def test_hawl_due_date_none_without_start():
    inputs = ZakatInputs(cash_by_currency={"USD": 10000})
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx)
    assert c.hawl_due_date is None


def test_custom_lunar_year_days():
    """Operator can override (e.g., 365 for solar year)."""
    start = date(2026, 1, 1)
    policy = ZakatPolicy(lunar_year_days=365)
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000},
        hawl_start_date=start,
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, policy=policy)
    assert c.hawl_due_date == date(2027, 1, 1)


# --- Custom policy (Khums-like) -----------------------------------------------


def test_custom_zakat_rate():
    """Khums = 20% on certain categories."""
    policy = ZakatPolicy(zakat_rate=0.20)
    inputs = ZakatInputs(cash_by_currency={"USD": 10000})
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, policy=policy)
    assert c.zakat_owed == pytest.approx(2000.0)


# --- days_until_hawl ---------------------------------------------------------


def test_days_until_hawl_returns_none_without_due_date():
    c = calculate_zakat(ZakatInputs(cash_by_currency={"USD": 10000}), _usd_fx())
    assert days_until_hawl(c, today=date(2026, 6, 1)) is None


def test_days_until_hawl_positive_before_due():
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000},
        hawl_start_date=date(2026, 1, 1),
    )
    c = calculate_zakat(inputs, _usd_fx())
    # Due 2026-12-21. Today 2026-06-01 → 203 days
    assert days_until_hawl(c, today=date(2026, 6, 1)) == 203


def test_days_until_hawl_negative_past_due():
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000},
        hawl_start_date=date(2026, 1, 1),
    )
    c = calculate_zakat(inputs, _usd_fx())
    # Due 2026-12-21; today 2027-01-01 → -11
    assert days_until_hawl(c, today=date(2027, 1, 1)) == -11


# --- Render -------------------------------------------------------------------


def test_render_meets_nisab_shows_owed():
    c = calculate_zakat(ZakatInputs(cash_by_currency={"USD": 10000}), _usd_fx())
    out = render_calculation(c)
    assert "💰" in out
    assert "OWED" in out
    assert "250.00" in out
    assert "USD" in out


def test_render_below_nisab_shows_clean():
    c = calculate_zakat(ZakatInputs(cash_by_currency={"USD": 100}), _usd_fx())
    out = render_calculation(c)
    assert "✅" in out
    assert "BELOW NISAB" in out


def test_render_includes_due_date_when_set():
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000},
        hawl_start_date=date(2026, 1, 1),
    )
    c = calculate_zakat(inputs, _usd_fx())
    out = render_calculation(c)
    assert "2026-12-21" in out


def test_render_no_secret_leak():
    """Pin: render output never includes per-account balances or alt-data."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 10000, "SAR": 5000, "EUR": 500},
        investments_by_currency={"USD": 50000},
    )
    c = calculate_zakat(inputs, _usd_fx())
    out = render_calculation(c)
    forbidden = ["SAR", "EUR", "5000", "50000", "Authorization", "account_id"]
    for word in forbidden:
        assert word not in out


# --- E2E flows ----------------------------------------------------------------


def test_e2e_saudi_diaspora_user():
    """A user with cash in USD/SAR + investments + small gold holding."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 5000, "SAR": 20000},
        investments_by_currency={"USD": 50000},
        gold_grams=50,  # small gold collection
        debts_owed_by_user_by_currency={"USD": 2000},  # credit card
        hawl_start_date=date(2026, 3, 15),
        reporting_currency="USD",
    )
    fx = _usd_fx()
    c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    # net = 5000 + 20000*0.27 + 50000 + 50*75 - 2000 = 62150
    assert c.net_assets == pytest.approx(5000 + 5400 + 50000 + 3750 - 2000)
    assert c.meets_nisab is True
    assert c.zakat_owed == pytest.approx(c.net_assets * 0.025)
    # 2026-03-15 + 354 = 2027-03-04
    assert c.hawl_due_date == date(2027, 3, 4)


def test_e2e_replay_consistency():
    """Pin: same inputs → equal calculation."""
    inputs = ZakatInputs(
        cash_by_currency={"USD": 1000, "SAR": 5000},
        gold_grams=10,
        hawl_start_date=date(2026, 1, 1),
    )
    fx = _usd_fx()
    a = calculate_zakat(inputs, fx)
    b = calculate_zakat(inputs, fx)
    assert a == b


def test_e2e_gold_basis_lets_a_smaller_holder_skip():
    """Pin: gold-basis user with $5000 net might fall below gold nisab
    (~$6561) and owe nothing, while silver-basis user pays."""
    inputs = ZakatInputs(cash_by_currency={"USD": 5000})
    fx = _usd_fx()
    silver_c = calculate_zakat(inputs, fx, basis=NisabBasis.SILVER)
    gold_c = calculate_zakat(inputs, fx, basis=NisabBasis.GOLD)
    assert silver_c.zakat_owed == pytest.approx(125.0)
    assert gold_c.zakat_owed == 0
