"""Earnings results and guidance, read from Benzinga's headlines.

Benzinga states the consensus in the headline itself:

    Micron Technology Q4 Adj. EPS $33.42 Beats $31.45 Estimate,
        Sales $54.229B Beat $50.751B Estimate
    Micron Technology Sees Q1 Adj EPS $37.15-$39.15 vs $35.07 Est;
        Sees Sales $60.000B-$63.000B vs $56.553B Est
    Carnival Raises FY2026 Adj EPS Guidance from $2.22 to $2.24 vs $2.22 Est

so a free regex yields what a paid consensus feed would: actual vs
estimate for EPS and sales, and guidance vs estimate. Surprises are
relative to the estimate's magnitude; price-scaled versions are built in
research, where the price is known.

Only what the headline states is extracted. A headline that matches
nothing returns no facts, never a guess. Year-over-year comparisons
("Up From $0.02 YoY") are not estimates and carry no surprise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_NUM = r"\(?-?\$?\(?-?[\d,]*\.?\d+\)?[KMB]?"
_PERIOD = r"(?P<period>(?:Q[1-4]|H[12])(?:\s+(?:FY\s?)?\d{2,4})?|FY\s?\d{2,4}|FY)"
_BASIS = r"(?P<basis>Adj\.?|Ajd\.?|Adjusted|GAAP|Core|Non-GAAP|Operating)?\s?"

_RESULT = re.compile(
    rf"\b{_PERIOD}\s+{_BASIS}(?:EPS\s+)?(?P<eps>{_NUM})\s+"
    rf"(?P<verdict>Beats|Misses|Meets|In-Line With|Inline With|Up From|Down From|vs\.?)\s+"
    rf"(?P<ref>{_NUM})\s+(?P<ref_kind>Estimate|Est\b\.?|YoY)",
    re.IGNORECASE,
)
# "Q3 EPS $0.22, Inline": the estimate equalled, stated without its number.
_INLINE = re.compile(
    rf"\b{_PERIOD}\s+{_BASIS}EPS\s+(?P<eps>{_NUM}),?\s+(?:In-?Line|Inline)\b",
    re.IGNORECASE,
)
_SALES = re.compile(
    rf"\b(?:Sales|Revenue)\s+(?P<sales>{_NUM})\s+"
    rf"(?P<verdict>Beats?|Miss(?:es)?|Meets?|Inline|In-Line|Up From|Down From|vs\.?)\s+"
    rf"(?P<ref>{_NUM})\s+(?P<ref_kind>Estimate|Est\b\.?|YoY)",
    re.IGNORECASE,
)
_GUIDE_RANGE = re.compile(
    rf"\b(?P<action>Sees|Raises|Lowers|Affirms|Reaffirms|Cuts|Maintains|Reiterates)\s+"
    rf"(?:{_PERIOD}\s+)?{_BASIS}(?P<metric>EPS|Sales|Revenue)\s+(?:Guidance\s+)?(?:of\s+)?"
    rf"(?:from\s+(?P<old>{_NUM}(?:\s?-\s?{_NUM})?)\s+to\s+)?"
    rf"(?:(?P<qualifier>At Least|At Most|Approximately|About|Around|Up To)\s+)?~?"
    rf"(?P<low>{_NUM})(?:\s?-\s?(?P<high>{_NUM}))?\s+vs\.?\s+(?P<est>{_NUM})\s+Est",
    re.IGNORECASE,
)


def money(text: str) -> float | None:
    """'$1.244B' -> 1.244e9, '$(1.26)' -> -1.26, '$756.000M' -> 7.56e8."""
    t = text.replace("$", "").replace(",", "").strip()
    negative = "(" in t or t.startswith("-")
    t = t.replace("(", "").replace(")", "").lstrip("-")
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(t[-1:].upper(), 1.0)
    if t[-1:].upper() in "KMB":
        t = t[:-1]
    try:
        value = float(t) * scale
    except ValueError:
        return None
    return -value if negative else value


def _surprise(actual: float | None, estimate: float | None) -> float | None:
    if actual is None or estimate is None:
        return None
    return (actual - estimate) / max(abs(estimate), 0.01)


@dataclass(frozen=True, slots=True)
class EarningsFacts:
    kind: str  # result | guidance
    fields: dict[str, object] = field(default_factory=dict)


def parse_headline(headline: str) -> list[EarningsFacts]:
    """Every result or guidance statement in one headline (segments split on ';')."""
    out: list[EarningsFacts] = []
    for segment in headline.split(";"):
        # Guidance first: "Sees Q4 Adj EPS $0.20 vs $0.26 Est" is a forecast,
        # though its tail reads like a result. Results carry no action verb.
        g = _GUIDE_RANGE.search(segment)
        if g is None and (m := _RESULT.search(segment)) is not None:
            eps, ref = money(m["eps"]), money(m["ref"])
            estimated = m["ref_kind"].lower().startswith("est")
            facts: dict[str, object] = {
                "period": m["period"].replace(" ", "").upper(),
                "basis": (m["basis"] or "GAAP").rstrip(".").lower().replace("ajd", "adj"),
                "eps": eps,
                "eps_estimate": ref if estimated else None,
                "eps_surprise": _surprise(eps, ref) if estimated else None,
                "eps_verdict": m["verdict"].lower(),
            }
            if (s := _SALES.search(segment, m.end())) is not None:
                sales, sref = money(s["sales"]), money(s["ref"])
                sest = s["ref_kind"].lower().startswith("est")
                facts |= {
                    "sales": sales,
                    "sales_estimate": sref if sest else None,
                    "sales_surprise": _surprise(sales, sref) if sest else None,
                    "sales_verdict": s["verdict"].lower(),
                }
            out.append(EarningsFacts("result", facts))
            continue
        if (m := _INLINE.search(segment)) is not None:
            eps = money(m["eps"])
            out.append(
                EarningsFacts(
                    "result",
                    {
                        "period": m["period"].replace(" ", "").upper(),
                        "basis": (m["basis"] or "GAAP").rstrip(".").lower(),
                        "eps": eps,
                        "eps_estimate": eps,
                        "eps_surprise": 0.0,
                        "eps_verdict": "inline",
                    },
                )
            )
            continue
        if g is not None:
            low, high, est = money(g["low"]), money(g["high"] or g["low"]), money(g["est"])
            mid = (low + high) / 2 if low is not None and high is not None else None
            period = g["period"]
            if period is None:  # "Sees Sales $X vs $Y Est" after a period-bearing segment
                prior = [f for f in out if f.kind == "guidance"]
                period = str(prior[-1].fields["period"]) if prior else None
            out.append(
                EarningsFacts(
                    "guidance",
                    {
                        "action": g["action"].lower(),
                        "period": period.replace(" ", "").upper() if period else None,
                        "basis": (g["basis"] or "GAAP").rstrip(".").lower(),
                        "metric": "eps" if g["metric"].upper() == "EPS" else "sales",
                        "low": low,
                        "high": high,
                        "mid": mid,
                        "estimate": est,
                        "surprise": _surprise(mid, est),
                        "previous": g["old"],
                        "qualifier": (g["qualifier"] or "").lower() or None,
                    },
                )
            )
    return out


# v2: the 2016-2019 formats ("EPS $1.27 vs $1.12 Est.", "EPS $0.22, Inline",
# "Ajd. EPS"), found validating the parser on the full history.
# v3: a year after the quarter ("Q4 2023 Adj EPS"), "Adj $0.65 Beats", "Revenue
# ... Misses" -- the commonest 2023-2026 misses.
EXTRACTOR = "benzinga-earnings-v3"


async def extract_all(engine: Any, *, batch: int = 5000) -> int:
    """Parse every news event this extractor version has not seen; returns facts stored.

    An event with no earnings statement is marked with a ``none`` fact so it
    is not re-read, and so the parse rate can be measured.
    """
    import json

    from sqlalchemy import text

    stored = 0
    while True:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT e.id, e.payload->>'headline' AS h FROM events e "
                        "WHERE e.kind = 'news' AND NOT EXISTS (SELECT 1 FROM event_facts f "
                        "WHERE f.event_id = e.id AND f.extractor = :x) ORDER BY e.id LIMIT :n"
                    ),
                    {"x": EXTRACTOR, "n": batch},
                )
            ).all()
        if not rows:
            return stored
        facts = []
        for r in rows:
            parsed = parse_headline(r.h or "")
            facts += [
                {"e": r.id, "x": EXTRACTOR, "k": f.kind, "f": json.dumps(f.fields)} for f in parsed
            ] or [{"e": r.id, "x": EXTRACTOR, "k": "none", "f": "{}"}]
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO event_facts (event_id, extractor, kind, fields) "
                    "VALUES (:e, :x, :k, CAST(:f AS JSONB))"
                ),
                facts,
            )
        stored += sum(1 for f in facts if f["k"] != "none")
