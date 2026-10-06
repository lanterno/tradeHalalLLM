"""Companies that moved their listing to a new SEC registrant.

A holding-company reorganisation gives a ticker a new CIK, and the new
registrant has none of the old one's filings: no annual revenue or
interest income until its first 10-K. Screened on the new CIK alone such
a company is "not computable" (doubtful) for a year, and, because the
screen's history is built from today's ticker map, for every past date as
well. Where the successor has no value for a fact, the screen reads the
predecessor's.

Kept by hand: SEC links neither registrant to the other. The weekly
SPUS/HLAL check lists every large company that is doubtful only for
missing data (``validate.missing_data``), which is how a new case shows up.
"""

from __future__ import annotations

# successor CIK -> predecessor CIK
PREDECESSOR: dict[int, int] = {
    # ExxonMobil Holdings Corp took over XOM in July 2026 (Exxon Mobil Corp
    # filed its 25-NSE on 2 Jul 2026); the first annual report under the new
    # CIK is the FY2026 10-K.
    2115436: 34088,
}


def lineage(cik: int) -> tuple[int, ...]:
    """``cik`` and its predecessors, newest first."""
    out = [cik]
    while out[-1] in PREDECESSOR and PREDECESSOR[out[-1]] not in out:
        out.append(PREDECESSOR[out[-1]])
    return tuple(out)
