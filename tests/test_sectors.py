"""Sector names from SIC descriptions: keywords match at the start of a word."""

from __future__ import annotations

import pytest

from halal_trader.compliance.sectors import sector_of


@pytest.mark.parametrize(
    ("description", "sector"),
    [
        # Each of these read as the wrong sector through a substring.
        ("Heating Equip, Except Elec & Warm Air; & Plumbing Fixtures", "Industrials"),  # "eating"
        ("Newspapers: Publishing or  Publishing & Printing", "Communications & media"),  # "paper"
        ("Electronic & Other Electrical Equipment (No Computer Equip)", "Industrials"),
        ("Medicinal Chemicals & Botanical Products", "Healthcare"),  # "chemical"
        ("Services-Offices & Clinics of  Doctors of  Medicine", "Healthcare"),  # "services-"
        ("Services-Skilled Nursing Care Facilities", "Healthcare"),
        ("Special Industry Machinery (No Metalworking Machinery)", "Industrials"),  # "metal"
        ("Calculating & Accounting Machines (No Electronic Computers)", "Computer hardware"),
        # And these must not move.
        ("Retail-Eating  Places", "Retail"),
        ("Converted Paper & Paperboard Prods (No Contaners/Boxes)", "Materials & mining"),
        ("Primary Production of  Aluminum", "Materials & mining"),
        ("Services-Prepackaged Software", "Software & internet"),
        ("Electronic Computers", "Computer hardware"),
        ("Biological Products, (No Diagnostic Substances)", "Healthcare"),
        ("Air-Cond & Warm Air Heatg Equip & Comm & Indl Refrig Equip", "Industrials"),
    ],
)
def test_sector_of(description: str, sector: str) -> None:
    assert sector_of(description) == sector
