"""Companies whose prohibited activity their SIC code does not show.

The activity screen reads a company's SIC code (aaoifi.PROHIBITED_SIC), and
a SIC code is coarse: hotel and casino landlords file as REITs (6798), a
billboard company as a REIT, Constellation Brands' beer and wine as plain
"Beverages" (2080). Above the index veto's size floor (about $19B in 2026)
an index Shariah board catches these; below it nothing did. This is the
hand-kept list for the rest, keyed by CIK (tickers change, CIKs do not),
each entry naming the company and its reason.

What belongs here: a company whose main business is impermissible under
the strict option (AAOIFI plus S&P Shariah's exclusions: alcohol, tobacco,
pork, gambling, conventional finance and insurance, hotels and casinos as
SIC 7011 already treats them, advertising and media, entertainment), stated
in its own filings, and a SIC code that hides it. Adding one is a line
here and a test is not needed per entry; the name must be the filer's.

Restaurants that sell alcohol (Texas Roadhouse, Brinker, Cheesecake
Factory), and grocers or drugstores selling alcohol and tobacco, are not
listed one by one: since screen v12 a pass in those sectors needs a Shariah
index's inclusion (aaoifi.MIXED_ACTIVITY_SIC, index_veto.require_board).

``NON_ALCOHOLIC_2080`` is the other side: SIC 2080 ("Beverages") is now an
excluded code because it holds Constellation Brands and Diageo beside
Coca-Cola, and these filers under it make only non-alcoholic drinks.
"""

from __future__ import annotations

_HOTEL_REIT = (
    "hotel REIT: owns and leases hotels (lodging with alcohol service and "
    "gaming-adjacent resorts; SIC 7011's exclusion, filed under 6798)"
)

# CIK -> (company, reason)
DENIED: dict[int, tuple[str, str]] = {
    # Hotels and casinos filed as REITs (SIC 6798).
    1070750: ("Host Hotels & Resorts, Inc.", _HOTEL_REIT),
    1040829: (
        "Ryman Hospitality Properties, Inc.",
        "hotel REIT (Gaylord convention resorts) and Opry Entertainment "
        "(S&P: entertainment), filed under 6798",
    ),
    1617406: ("Park Hotels & Resorts Inc.", _HOTEL_REIT),
    1418121: ("Apple Hospitality REIT, Inc.", _HOTEL_REIT),
    1474098: ("Pebblebrook Hotel Trust", _HOTEL_REIT),
    1295810: ("Sunstone Hotel Investors, Inc.", _HOTEL_REIT),
    1298946: ("DiamondRock Hospitality Co", _HOTEL_REIT),
    1511337: ("RLJ Lodging Trust", _HOTEL_REIT),
    1616000: ("Xenia Hotels & Resorts, Inc.", _HOTEL_REIT),
    1497645: ("Summit Hotel Properties, Inc.", _HOTEL_REIT),
    1476045: ("Chatham Lodging Trust", _HOTEL_REIT),
    1232582: ("Ashford Hospitality Trust Inc", _HOTEL_REIT),
    1574085: ("Braemar Hotels & Resorts Inc.", _HOTEL_REIT),
    945394: (
        "Service Properties Trust",
        "REIT whose largest segment is hotels (lodging with alcohol service)",
    ),
    1705696: (
        "VICI Properties Inc.",
        "casino REIT: owns Caesars Palace, MGM Grand and other casino resorts (gambling)",
    ),
    1575965: ("Gaming & Leisure Properties, Inc.", "casino REIT: owns casinos (gambling)"),
    1045450: (
        "EPR Properties",
        "experiential REIT: movie theatres, casinos and amusement venues "
        "(S&P: entertainment; gambling)",
    ),
    # Advertising filed as a REIT.
    1090425: (
        "Lamar Advertising Co",
        "billboard advertising (S&P: advertising and media), filed under 6798",
    ),
    1579877: (
        "OUTFRONT Media Inc.",
        "billboard advertising (S&P: advertising and media), filed under 6798",
    ),
    # Alcohol and tobacco under codes that do not say so.
    16918: ("Constellation Brands, Inc.", "beer, wine and spirits (alcohol), filed as Beverages"),
    1731348: ("Tilray Brands, Inc.", "cannabis and craft beer"),
    102037: ("Universal Corp", "leaf tobacco merchant, filed as farm-product wholesale"),
    1290677: ("Turning Point Brands, Inc.", "tobacco: chewing tobacco, papers, nicotine products"),
    # Gambling under codes that do not say so.
    750004: ("Light & Wonder, Inc.", "gaming machines and casino systems (gambling)"),
    1793659: ("Rush Street Interactive, Inc.", "online casino and sports betting (gambling)"),
    1071255: ("Golden Entertainment, Inc.", "casinos and slot-route operations (gambling)"),
    907242: ("Monarch Casino & Resort Inc", "casinos (gambling)"),
    # Cruise lines (SIC 4400, shared with shipping): onboard casinos and bars.
    884887: (
        "ROYAL CARIBBEAN CRUISES LTD",
        "cruise line: onboard revenue includes casino operations (gambling) and drinks",
    ),
    1513761: (
        "Norwegian Cruise Line Holdings Ltd.",
        "cruise line: onboard revenue includes casino operations (gambling) and drinks",
    ),
    815097: (
        "Carnival Corp Ltd.",
        "cruise line: bars and lounges across the fleet; onboard and other revenue, "
        "including drinks, is 35% of the total (alcohol)",
    ),
    # Conventional finance under business-services codes (SIC 7389).
    1428336: (
        "HealthEquity, Inc.",
        "custodial revenue (over a third of revenue) is interest earned placing HSA "
        "members' cash with banks and insurers",
    ),
    1662991: (
        "Sezzle Inc.",
        "consumer credit (buy now, pay later): about half its revenue is fees on "
        "its loans to shoppers, filed as RevenueNotFromContractWithCustomer",
    ),
}

# SIC 2080 filers that make no alcoholic drinks.
NON_ALCOHOLIC_2080: dict[int, str] = {
    21344: "The Coca-Cola Company",
    77476: "PepsiCo, Inc.",
    1418135: "Keurig Dr Pepper Inc.",
    1482981: "The Vita Coco Company, Inc.",
    2042694: "Primo Brands Corp (bottled water)",
}


def denied(cik: int | None) -> str | None:
    """The reason a company is excluded by name, or None."""
    if cik is None or cik not in DENIED:
        return None
    company, reason = DENIED[cik]
    return f"{reason} ({company}, excluded by name)"
