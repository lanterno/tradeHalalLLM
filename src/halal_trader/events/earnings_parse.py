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
("Up From $0.02 YoY") are not estimates and carry no surprise, and neither
does a statement Benzinga flags as not comparable ("May Not Compare To"):
the fact's ``not_comparable`` field is set and that statement's surprise is
None. The flag belongs to the statement whose own text carries it, so in
"EPS $(1.49) Beats $(1.52) Estimate, Sales $517.00K May Not Compare To
$3.54M Estimate" the EPS beat keeps its surprise.

A headline is read segment by segment (split on ``;``). The first template
of ``PARSE_ORDER`` that matches a segment reads it, so one segment gives at
most one fact; a result fact also carries the sales figure stated after the
EPS in the same segment.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

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

# v4 templates (found among the headlines v3 left unread). A quarter's sales
# alone: "Bloom Energy Q4 Sales $213.8M Miss $270.07M Estimate".
SALES_ONLY: Final = re.compile(
    rf"\b{_PERIOD}\s+(?:Adj\.?\s+)?(?:Net\s+)?(?:Sales|Revenues?)\s+(?P<sales>{_NUM})\s+"
    rf"(?P<verdict>Beats?|Miss(?:es)?|Meets?|In-?Line)\s+(?P<ref>{_NUM})\s+Estimate",
    re.I,
)
# The estimate before its number: "TripAdvisor Reports Q4 EPS $0.16 vs. Est. $0.31".
RESULT_EST_FIRST: Final = re.compile(
    rf"\b{_PERIOD}\s+{_BASIS}EPS\s+(?P<eps>{_NUM})\s+vs\.?\s+Est\.?\s+(?P<ref>{_NUM})", re.I
)
# Guidance in other words: "Kennametal Expects Q1 Sales Of $465M-$485M Vs $485.8M Est",
# "Abbott Expects ... Versus Consensus Of $1.25", "Amgen Narrows FY2026 GAAP EPS
# Guidance ... vs $15.44 Est", "Workiva Expects ... Revenue Of $718M-$722M (Est $731.727M)".
GUIDE_V4: Final = re.compile(
    r"\b(?P<action>Sees|Expects|Guides|Forecasts|Projects|Narrows|Revises|Updates|Issues|"
    r"Initiates|Raises|Lowers|Cuts|Affirms|Reaffirms|Maintains|Reiterates)\s+"
    rf"(?:{_PERIOD}\s+)?.{{0,40}}?(?P<metric>EPS|Sales|Revenues?)\b.{{0,40}}?"
    rf"(?P<low>{_NUM})(?:\s?(?:-|to)\s?(?P<high>{_NUM}))?\s*"
    rf"(?:vs\.?|Versus(?:\s+Consensus\s+Of)?|\(Est\.?)\s+(?P<est>{_NUM})",
    re.I,
)
# Benzinga's flag that the figure is not on the estimate's basis: no surprise.
NOT_COMPARABLE: Final = re.compile(r"May Not Compare To", re.I)
# Guidance against consensus that no template reads: an earnings item whose
# guidance is unknown (the taxonomy's ``guidance_unparsed``), never a fact.
GUIDE_UNPARSED: Final = re.compile(
    r"\b(?:Sees|Expects|Guides|Forecasts|Projects|Narrows|Revises|Updates|Issues|Initiates|"
    r"Reaffirms|Affirms|Reiterates|Maintains)\b.{0,60}"
    r"\b(?:EPS|Sales|Revenues?|Guidance|Outlook)\b.{0,80}"
    r"(?:\bEst|Estimate|Consensus|May Not Compare)",
    re.I,
)

# Each segment is read by the first of these that matches it.
PARSE_ORDER: Final = (
    "_GUIDE_RANGE",
    "GUIDE_V4",
    "_RESULT",
    "RESULT_EST_FIRST",
    "_INLINE",
    "SALES_ONLY",
)

# The basis words a guidance wire carries before its metric (GUIDE_V4 has no group).
_BASIS_WORD = re.compile(r"\b(?:Non-GAAP|GAAP|Adjusted|Adj|Ajd|Core|Operating)\b", re.I)
# GUIDE_V4's lazy gap can step over the guided level to its tolerance: in
# "Sees Q1 Revenue $7.1B +/- $300M Vs $6.995B Est" the figure it reads is the
# $300M, a -96% "surprise" that would type the story a guidance cut. Such a
# read is dropped and the later templates are tried; a segment none of them
# reads stays unread (GUIDE_UNPARSED: guidance unknown).
_TOLERANCE_GAP = re.compile(r"\+/-|±")


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


def _basis(raw: str | None) -> str:
    """'Adj.' -> 'adj', 'Ajd.' -> 'adj', None -> 'gaap'; the other words lowercased."""
    return (raw or "GAAP").rstrip(".").lower().replace("ajd", "adj")


def _period(raw: str) -> str:
    return raw.replace(" ", "").upper()


@dataclass(frozen=True, slots=True)
class EarningsFacts:
    kind: str  # result | guidance
    fields: dict[str, object] = field(default_factory=dict)


def _comparable(
    fields: dict[str, object], segment: str, m: re.Match[str], surprise: str
) -> dict[str, object]:
    """``fields``, or with ``surprise`` None and ``not_comparable`` set when the
    statement ``m`` read says its figure may not compare to the estimate."""
    if NOT_COMPARABLE.search(segment, m.start(), m.end()) is None:
        return fields
    return fields | {surprise: None, "not_comparable": True}


def _with_sales(fields: dict[str, object], segment: str, after: int) -> dict[str, object]:
    """``fields`` plus the sales figure stated after position ``after``, if any."""
    s = _SALES.search(segment, after)
    if s is None:
        return fields
    sales, ref = money(s["sales"]), money(s["ref"])
    estimated = s["ref_kind"].lower().startswith("est")
    stated: dict[str, object] = {
        "sales": sales,
        "sales_estimate": ref if estimated else None,
        "sales_surprise": _surprise(sales, ref) if estimated else None,
        "sales_verdict": s["verdict"].lower(),
    }
    return _comparable(fields | stated, segment, s, "sales_surprise")


def _eps_result(m: re.Match[str], segment: str, *, verdict: str, estimated: bool) -> EarningsFacts:
    eps, ref = money(m["eps"]), money(m["ref"])
    fields: dict[str, object] = {
        "period": _period(m["period"]),
        "basis": _basis(m["basis"]),
        "eps": eps,
        "eps_estimate": ref if estimated else None,
        "eps_surprise": _surprise(eps, ref) if estimated else None,
        "eps_verdict": verdict,
    }
    fields = _comparable(fields, segment, m, "eps_surprise")
    return EarningsFacts("result", _with_sales(fields, segment, m.end()))


def _guidance(
    g: re.Match[str],
    segment: str,
    prior: Sequence[EarningsFacts],
    *,
    basis: str,
    previous: str | None,
) -> EarningsFacts:
    low, high, est = money(g["low"]), money(g["high"] or g["low"]), money(g["est"])
    mid = (low + high) / 2 if low is not None and high is not None else None
    period: str | None = g["period"]
    if period is None:  # "Sees Sales $X vs $Y Est" after a period-bearing segment
        earlier = [f for f in prior if f.kind == "guidance"]
        period = str(earlier[-1].fields["period"]) if earlier else None
    qualifier = g.groupdict().get("qualifier")
    fields: dict[str, object] = {
        "action": g["action"].lower(),
        "period": _period(period) if period else None,
        "basis": basis,
        "metric": "eps" if g["metric"].upper() == "EPS" else "sales",
        "low": low,
        "high": high,
        "mid": mid,
        "estimate": est,
        "surprise": _surprise(mid, est),
        "previous": previous,
        "qualifier": (qualifier or "").lower() or None,
    }
    return EarningsFacts("guidance", _comparable(fields, segment, g, "surprise"))


def _segment_fact(segment: str, prior: Sequence[EarningsFacts]) -> EarningsFacts | None:
    """The fact the first template of ``PARSE_ORDER`` reads from ``segment``, if any.

    Guidance comes first: "Sees Q4 Adj EPS $0.20 vs $0.26 Est" is a forecast,
    though its tail reads like a result. Results carry no action verb.
    """
    if (g := _GUIDE_RANGE.search(segment)) is not None:
        return _guidance(g, segment, prior, basis=_basis(g["basis"]), previous=g["old"])
    g = GUIDE_V4.search(segment)
    if g is not None and _TOLERANCE_GAP.search(segment, g.end("metric"), g.start("low")):
        g = None
    if g is not None:
        word = _BASIS_WORD.search(segment, g.end("action"), g.start("metric"))
        basis = _basis(word[0] if word else None)
        return _guidance(g, segment, prior, basis=basis, previous=None)
    if (m := _RESULT.search(segment)) is not None:
        estimated = m["ref_kind"].lower().startswith("est")
        return _eps_result(m, segment, verdict=m["verdict"].lower(), estimated=estimated)
    if (m := RESULT_EST_FIRST.search(segment)) is not None:
        return _eps_result(m, segment, verdict="vs", estimated=True)
    if (m := _INLINE.search(segment)) is not None:
        eps = money(m["eps"])
        fields: dict[str, object] = {
            "period": _period(m["period"]),
            "basis": _basis(m["basis"]),
            "eps": eps,
            "eps_estimate": eps,
            "eps_surprise": 0.0,
            "eps_verdict": "inline",
        }
        fields = _comparable(fields, segment, m, "eps_surprise")
        return EarningsFacts("result", _with_sales(fields, segment, m.end()))
    if (m := SALES_ONLY.search(segment)) is not None:
        sales, ref = money(m["sales"]), money(m["ref"])
        stated = segment[m.end("period") : m.start("sales")]
        fields = {
            "period": _period(m["period"]),
            "basis": "adj" if re.search(r"\bAdj\b", stated, re.I) else "gaap",
            "sales": sales,
            "sales_estimate": ref,
            "sales_surprise": _surprise(sales, ref),
            "sales_verdict": m["verdict"].lower(),
        }
        return EarningsFacts("result", _comparable(fields, segment, m, "sales_surprise"))
    return None


def parse_headline(headline: str) -> list[EarningsFacts]:
    """Every result or guidance statement in one headline (segments split on ';')."""
    out: list[EarningsFacts] = []
    for segment in headline.split(";"):
        if (fact := _segment_fact(segment, out)) is not None:
            out.append(fact)
    return out


# v2: the 2016-2019 formats ("EPS $1.27 vs $1.12 Est.", "EPS $0.22, Inline",
# "Ajd. EPS"), found validating the parser on the full history.
# v3: a year after the quarter ("Q4 2023 Adj EPS"), "Adj $0.65 Beats", "Revenue
# ... Misses" -- the commonest 2023-2026 misses.
# v4: sales-only results, "EPS $X vs. Est. $Y", guidance in other verbs
# ("Expects", "Narrows", "Versus Consensus Of", "(Est $Y)"), "May Not Compare
# To" figures without a surprise, and the sales stated after an inline EPS.
# The news engine reads v4; v3 rows stay in event_facts.
EXTRACTOR_V3: Final = "benzinga-earnings-v3"
EXTRACTOR: Final = "benzinga-earnings-v4"


def sources() -> dict[str, str]:
    """Every pattern's source text, the parse order and the extractor, for the pins."""
    patterns = {
        "_RESULT": _RESULT,
        "_INLINE": _INLINE,
        "_SALES": _SALES,
        "_GUIDE_RANGE": _GUIDE_RANGE,
        "SALES_ONLY": SALES_ONLY,
        "RESULT_EST_FIRST": RESULT_EST_FIRST,
        "GUIDE_V4": GUIDE_V4,
        "NOT_COMPARABLE": NOT_COMPARABLE,
        "GUIDE_UNPARSED": GUIDE_UNPARSED,
        "_BASIS_WORD": _BASIS_WORD,
        "_TOLERANCE_GAP": _TOLERANCE_GAP,
    }
    return {name: rx.pattern for name, rx in patterns.items()} | {
        "PARSE_ORDER": ",".join(PARSE_ORDER),
        "EXTRACTOR": EXTRACTOR,
    }


# The parser's pin in the news engine's pre-registration: a change to any
# pattern above is a new parser, so a new trial.
PARSER_SHA: Final = hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()[:12]


async def extract_all(engine: Any, *, batch: int = 5000) -> int:
    """Parse every news event this extractor version has not seen; returns facts stored.

    An event with no earnings statement is marked with a ``none`` fact so it
    is not re-read, and so the parse rate can be measured.
    """
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
