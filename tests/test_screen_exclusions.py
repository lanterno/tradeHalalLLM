"""Activities a SIC code hides: the CIK deny-list, and SIC 2080's allow-list."""

from __future__ import annotations

import pytest

from halal_trader.compliance.aaoifi import Fundamentals, prohibited_activity, screen
from halal_trader.compliance.exclusions import DENIED, NON_ALCOHOLIC_2080


def _f(**kw: object) -> Fundamentals:
    base: dict[str, object] = dict(
        symbol="CO",
        sic=6798,
        shares_outstanding=1_000.0,
        price=100.0,
        interest_bearing_debt=10_000.0,
        cash_and_securities=20_000.0,
        interest_income=100.0,
        revenue=50_000.0,
    )
    base.update(kw)
    return Fundamentals(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("cik", "sic", "fragment"),
    [
        (1070750, 6798, "hotel REIT"),  # HST
        (1040829, 6798, "Gaylord"),  # RHP
        (1090425, 6798, "billboard advertising"),  # LAMR
        (16918, 2080, "beer, wine and spirits"),  # STZ
        (1705696, 6798, "casino"),  # VICI
        (1428336, 7389, "HSA"),  # HQY
    ],
)
def test_named_companies_fail_on_activity_whatever_their_ratios(
    cik: int, sic: int, fragment: str
) -> None:
    r = screen(_f(cik=cik, sic=sic))
    assert r.verdict == "not_halal"
    assert r.reasons[0].startswith("business activity") and fragment in r.reasons[0]
    assert DENIED[cik][0] in r.reasons[0]  # the filer's name is on the record


def test_the_deny_list_wins_even_without_an_industry_code() -> None:
    assert screen(_f(cik=1070750, sic=None)).verdict == "not_halal"


@pytest.mark.parametrize("cik", sorted(NON_ALCOHOLIC_2080))
def test_named_soft_drink_makers_under_sic_2080_face_only_the_ratios(cik: int) -> None:
    assert prohibited_activity(2080, cik) is None
    assert screen(_f(cik=cik, sic=2080)).verdict == "halal"


def test_any_other_sic_2080_filer_is_excluded_as_possibly_alcoholic() -> None:
    r = screen(_f(cik=835403, sic=2080))  # Diageo files under 2080 too
    assert r.verdict == "not_halal" and "2080" in r.reasons[0]
    assert screen(_f(cik=None, sic=2080)).verdict == "not_halal"
    assert prohibited_activity(2086) is None  # bottled & canned soft drinks: a code of its own


def test_all_tobacco_manufacturing_codes_are_excluded() -> None:
    for sic in (2100, 2111, 2121, 2131, 2141):
        assert prohibited_activity(sic) == "tobacco products"


def test_every_entry_names_its_company_and_reason_and_the_lists_do_not_overlap() -> None:
    assert all(name and reason for name, reason in DENIED.values())
    assert not set(DENIED) & set(NON_ALCOHOLIC_2080)
