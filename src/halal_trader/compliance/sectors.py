"""Plain sector names from the SEC's industry (SIC) descriptions.

The screen keeps each filer's SIC description ("Services-Prepackaged
Software"), which is precise and unreadable on a phone. This folds the
roughly 300 descriptions seen in the universe into about fifteen sectors a
person would name. Order matters: the first rule with a keyword at the
start of a word wins, so specific rules ("semiconductor") sit above broad
ones ("electronic"). A "(No ...)" clause is ignored: it says what the code
excludes. Display only; nothing trades on it.
"""

from __future__ import annotations

import re

OTHER = "Other"

# fmt: off
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Electrical-equipment makers (GE Vernova, Emerson's peers) would read as
    # computer hardware through "electronic" below.
    ("Industrials", (
        "electronic & other electrical",
    )),
    ("Retail", (
        "retail", "eating", "catalog", "mail-order", "wholesale",
    )),
    ("Semiconductors", (
        "semiconductor", "printed circuit",
    )),
    ("Software & internet", (
        "software", "computer programming", "data processing", "computer processing",
        "computer integrated", "information retrieval", "business services", "computer related",
    )),
    ("Computer hardware", (
        "computer", "electronic", "communications equipment", "telephone & telegraph apparatus",
        "photographic", "office machines", "instruments for meas", "measuring",
        "laboratory analytical", "optical", "calculating",
    )),
    ("Healthcare", (
        "pharmaceutical", "biological", "medical", "medicinal", "surgical", "dental", "health",
        "hospital", "diagnostic", "ophthalmic", "x-ray", "electromedical", "in vitro",
        "doctors", "clinics", "nursing",
    )),
    ("Autos", (
        "motor vehicle", "auto dealers", "tires",
    )),
    ("Energy", (
        "petroleum", "crude", "natural gas", "oil & gas", "oil and gas", "coal", "drilling",
        "pipe lines", "gas transmission", "royalty",
    )),
    ("Utilities", (
        "electric services", "electric & other", "water supply", "gas distribution",
        "cogeneration", "sanitary", "gas & other services",
    )),
    ("Materials & mining", (
        "mining", " ores", "metal", "steel", "chemical", "plastic", "paper", "glass", "cement",
        "lumber", "wood", "nonmetallic", "fertilizer", "paints", "rolling", "smelting",
        "industrial inorganic", "industrial organic", "quarrying", "aluminum", "concrete",
    )),
    ("Real estate", (
        "real estate", "operative builders", "lessors", "land subdividers",
        "nonresidential buildings",
    )),
    ("Financials", (
        "bank", "insurance", "finance", "credit", "security", "investment", "savings",
        "brokers", "loan", "trust", "holding offices", "patent owners",
    )),
    ("Communications & media", (
        "telephone communications", "radiotelephone", "cable", "television", "broadcasting",
        "communications services", "motion picture", "newspapers", "periodicals", "books",
        "advertising", "publishing",
    )),
    ("Transport", (
        "air transportation", "air courier", "airports", "railroad", "trucking",
        "water transportation", "freight", "courier", "transportation services", "deep sea",
        "pipe line",
    )),
    ("Consumer goods", (
        "food", "beverage", "bottled", "soft drinks", "malt", "wines", "candy", "dairy", "meat",
        "cosmetic", "perfume", "soap", "apparel", "garment", "furnishings", "footwear", "shoes",
        "household", "furniture", "toys", "games", "sporting", "jewelry", "tobacco", "hotels",
        "motels", "amusement", "recreation", "personal services", "educational", "cigarettes",
        "textile", "knitting", "watches", "cutlery", "home", "grain mill", "fats & oils",
        "poultry", "canned", "sugar", "confectionery", "agricultural prod", "leather",
        "carpets", "cleaning", "motorcycles",
    )),
    ("Industrials", (
        "machinery", "equipment", "engines", "construction", "contractors", "industrial",
        "electrical", "pumps", "valves", "tools", "conveyors", "elevators", "ball & roller",
        "refrigeration", "heating", "engineering", "guided missiles", "ordnance", "fabricated",
        "management services", "staffing", "help supply", "facilities support", "detective",
        "services-", "miscellaneous manufacturing", "aeronautical", "aircraft", "ship & boat",
        "navigation", "air-cond", "refrig", "refuse", "waste", "switchgear",
        "motors & generators", "wire", "auto controls",
    )),
)
# fmt: on


def _pattern(keyword: str) -> re.Pattern[str]:
    # A keyword matches at the start of a word only: "eating" must not find
    # "Heating", nor "paper" "Newspapers". It may run on ("paper" finds
    # "Paperboard", "metal" "Metals").
    return re.compile(r"(?<![a-z])" + re.escape(keyword.strip()))


_PATTERNS = tuple((sector, tuple(_pattern(k) for k in keywords)) for sector, keywords in _RULES)
# "(No Computer Equip)", "(No Diagnostic Substances)": what a code excludes
# is not what the company does.
_NEGATED = re.compile(r"\(\s*no\b[^)]*\)")


def sector_of(sic_description: str | None) -> str:
    text = _NEGATED.sub(" ", (sic_description or "").lower())
    for sector, patterns in _PATTERNS:
        if any(p.search(text) for p in patterns):
            return sector
    return OTHER
