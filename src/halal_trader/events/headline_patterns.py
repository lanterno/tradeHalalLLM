"""Benzinga headline templates shared by the story builder, the alias learner and the taxonomy.

One copy of each pattern: the analyst-action template (who acts on which
company), the earnings and guidance templates' company slot, and the
correction prefix a re-issued wire carries. The news engine pins their
source text in its pre-registration, so a change here is a new trial.
"""

from __future__ import annotations

import re
from typing import Final

# An analyst action. Case-sensitive: Benzinga capitalises the verb in its
# rating wires ("Morgan Stanley Downgrades Apple to Equal-Weight").
ANALYST_ACTION: Final = re.compile(
    r"\b(?:Downgrades?|Upgrades?|Maintains|Reiterates|Initiates|Assumes|Resumes|Reinstates)\b"
)

# The company an analyst wire is about: "<Firm> Downgrades <Co> to ...",
# "<Firm> Maintains <Rating> on <Co>, Lowers Price Target to $X" (or "On
# <Co>", the 2016-2020 "UPDATE:" wires), "<Firm> Initiates <Co> With Buy".
# A bare "Initiates" names a company only before "With" and a rating: "Lilly
# Initiates Phase 3 Trial ... With ..." and "Initiates Coverage With ..." are
# no slot.
ANALYST_SLOT: Final = re.compile(
    r"\b(?:Downgrades|Upgrades|Initiates Coverage On|Assumes|Resumes|Reinstates)\s+(?P<a>.+?)"
    r"(?:\s+(?:to|To|With|with|At|at)\b|,|$)"
    r"|\b(?:Maintains|Reiterates)\s+.{1,40}?\s+(?:on|On)\s+(?P<b>.+?)(?:,|$)"
    r"|\bInitiates\s+(?!Coverage\b)(?P<c>.+?)\s+(?:With|with)\s+(?:An?\s+)?['\"]?"
    r"(?:(?:Strong|Speculative)\s+)?(?:Buy|Sell|Hold|Neutral|Accumulate|Reduce|Overweight|"
    r"Underweight|Equal[- ]?[Ww]eight|In-?[Ll]ine|(?:Market|Sector) Weight|"
    r"(?:Market |Sector |Peer )?(?:Outperform|Underperform|Perform))\b"
)

# The company slot of an earnings wire ("Apple Q3 Adj. EPS $1.40 Beats ...")
# and of a guidance wire ("Apple Sees Q4 Sales ...").
EARN_CO: Final = re.compile(
    r"^(?P<co>[A-Z][^;:]{1,50}?)\s+(?:Q[1-4]|FY|H[12])\S*\s+(?:\d{2,4}\s+)?"
    r"(?:Adj\.?\s+|Adjusted\s+|GAAP\s+|Core\s+)?(?:EPS|Sales|Revenue|Loss)"
)
GUIDE_CO: Final = re.compile(
    r"^(?P<co>[A-Z][^;:]{1,50}?)\s+(?:Sees|Raises|Lowers|Affirms|Reaffirms|Cuts|Maintains|Reiterates)"
    r"\s+(?:Q[1-4]|FY|H[12])"
)

# A re-issued wire's prefix; stripped before comparing texts.
REISSUE_PREFIX: Final = re.compile(
    r"^\s*(?:CORRECTION|CORRECTED|UPDATE[D]?|REPORTED EARLIER|EARLIER|BREAKING)\s*[:,\-]\s*", re.I
)
# A correction supersedes the facts of the wire it corrects.
CORRECTION: Final = re.compile(r"^\s*(?:CORRECTION|CORRECTED)\b", re.I)


def sources() -> dict[str, str]:
    """Every pattern's source text, for the pre-registration's pins."""
    return {
        name: rx.pattern
        for name, rx in (
            ("ANALYST_ACTION", ANALYST_ACTION),
            ("ANALYST_SLOT", ANALYST_SLOT),
            ("EARN_CO", EARN_CO),
            ("GUIDE_CO", GUIDE_CO),
            ("REISSUE_PREFIX", REISSUE_PREFIX),
            ("CORRECTION", CORRECTION),
        )
    }
