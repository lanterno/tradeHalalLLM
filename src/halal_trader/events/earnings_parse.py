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

**Not comparable.** A statement has no surprise, and its fact is flagged
``not_comparable`` (per metric on a result: ``eps_not_comparable``,
``sales_not_comparable``), when

* Benzinga says so ("May Not Compare", with or without "To") in the
  statement's own clause, which runs from the statement to the next metric
  it names; a flag closing the segment covers every statement in it. In
  "EPS $(1.49) Beats $(1.52) Estimate, Sales $517.00K May Not Compare To
  $3.54M Estimate" the EPS beat keeps its surprise;
* the figure and its estimate are not in the same units: one carries a
  K/M/B suffix and the other none ("EPS $5.54-$5.61 Vs $5.66B Est."), or
  one is more than :data:`UNIT_RATIO` times the other ("Q1 2024 Vs $149.45M
  Est.", the year read as the EPS), unless it is an EPS whose smaller side
  is under :data:`EPS_RATIO_FLOOR` ("EPS $(0.68) Misses $0.01 Estimate") and
  larger side under :data:`EPS_RATIO_CAP` ("Q1 2024 Vs $(0.03) Est." does
  not compare).

**Guidance figures.** ``GUIDE_V4``'s lazy gap can stop on the wrong number:
a fragment glued to the token before it (the "1" of "Q1", the "-$1.85" of
"$1.75-$1.85"), or the guidance being replaced ("To $7.00-$7.30 From
$6.80-$7.30 vs. $7.17 Est", "(Prior $8.00) Vs."). Its figure is re-read as
the last dollar figure between its metric and its estimate that is neither
glued nor introduced by From/Prior/Previous/Was ("From $80M To $75M" gives
the $75M; Prior and Previous may name it in a few words first, "Versus Prior
Guidance Of $13.20-$13.60") nor inside a parenthetical opening with Prior,
From or Previous ("(From $8.80 To $8.90)"). Guidance kept as it was (Sees,
Affirms, Reaffirms, Maintains, Reiterates) may state its range with a
"Guidance from" no "to" follows ("PPL Reaffirms FY2017 EPS Guidance from
$1.92-2.12 vs $2.16 Est"); any other "from" states a base ("Sees FY Sales Down
5% From $1.2B"). When none qualifies the fact keeps its action with no figure and no
surprise, so "Lowers" and "Cuts" still count as guidance down. A range
states its unit once: "$643-$684M" is $643M to $684M.

A headline is read segment by segment (split on ``;``). The first template
of ``PARSE_ORDER`` that matches a segment reads it, so one segment gives at
most one fact; a result fact also carries the sales figure stated after the
EPS in the same segment ("Sales $X Beat $Y Estimate", or the estimate first:
"Rev. $X vs. Est. $Y"). Results carry no forward-looking verb: a result
template never reads a statement that "Sees", "Expects", "Guides",
"Forecasts" or "Projects" precedes ("Celgene Sees Q4 Adj EPS $1.18 Vs Est
$1.30" is single-figure guidance no template reads).

Against v3 (every distinct earnings-like headline to 2026-10): v4 reads 15
headlines v3 read as results as guidance instead, pre-announcements such as
"Ashland Sees Prelim. Q1 Adj. EPS $0.97 vs $1.10 Est.", which ``GUIDE_V4``
reads before the result templates (and whose sales, stated after, a
guidance fact does not carry); 3 v3 "results" are forecasts it no longer
reads ("Globus Medical Sees FY17 EPS $1.27, Inline"); and it corrects v3's
own figures where the unit was stated once or the estimate did not compare.
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
# The sales after an estimate-first EPS: "..., Rev. $316M vs. Est. $326M".
SALES_EST_FIRST: Final = re.compile(
    rf"\b(?:Sales|Revenues?|Revs?\.?)\s+(?P<sales>{_NUM})\s+vs\.?\s+Est\.?\s+(?P<ref>{_NUM})", re.I
)
# Benzinga's flag that the figure is not on the estimate's basis: no surprise.
NOT_COMPARABLE: Final = re.compile(r"May Not Compare", re.I)
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
# A guided figure GUIDE_V4's read is re-read from: a dollar amount not glued
# to the token before it (the "$0.57)" of "($0.57)" is no figure, and the
# typo "$($0.57)" reads whole, as -0.57) nor stating a change ("By $0.05 To
# $5.60-$5.75", though "Group $3.2B" is a figure), alone or as a range ("$0.70
# - $0.76", "$1.23 to $1.27", "To 3.22-$3.31"). A "to" never joins a figure to
# the start of another range, and a lone figure is not the start of a range it
# cannot complete ("$6.70-0$7.50").
_FIGURE = re.compile(
    r"(?<![\w.,$(-])(?<!\bBy )(?<!\bUp )(?<!\bDown )"
    r"(?P<low>(?:\$\()?\(?-?\$\(?-?[\d,]*\.?\d+\)?[KMB]?"
    r"|(?<=\bTo )\d[\d,]*(?:\.\d+)?(?=\s?-\s?\$))(?![\w.%$])"
    rf"(?:(?P<join>\s?-\s?|\s+to\s+)(?P<high>{_NUM})(?![\w.%$])(?!\s?-\s?\(?-?\$?\d)|(?!\s?-))",
    re.I,
)
# The words that introduce the guidance a figure replaces ("From $6.80-$7.30",
# "(Prior View: $8.00)"); "Will Range From" introduces the guidance itself.
# Prior and Previous(ly) may name it in up to three words first ("Versus Prior
# Guidance Of $13.20-$13.60", "Previously-Issued Range $2.11-$2.16"), none a
# figure, a "To" or a "Now", and no clause break.
_OLD = re.compile(
    r"\b(?:(?<!Range )From|Was|"
    r"(?:Prior|Previous(?:ly)?)(?:[\s-]+(?!(?:To|Now)\b)[^\s$;(),]+){0,3})"
    r"\s*:?\s*\(?\s*~?\s*$",
    re.I,
)
_FROM = re.compile(r"\bFrom\s*:?\s*\(?\s*~?\s*$", re.I)
# How far before a figure _OLD looks for the words that introduce it.
OLD_WINDOW: Final = 40
# Guidance kept as it was: a "Guidance from" before its first figure
# introduces the range itself when no "to" follows ("PPL Reaffirms FY2017 EPS
# Guidance from $1.92-2.12 vs $2.16 Est"); after one, a "from" introduces the
# range replaced ("Atkore Sees FY Adj. EPS $1.37-$1.45 from $1.55-$1.65 vs
# $1.57 Est."). Any other "from" states a base ("Sees FY Sales Down 5% From
# $1.2B"), never the guidance.
KEEP_ACTIONS: Final = frozenset({"sees", "affirms", "reaffirms", "maintains", "reiterates"})
_GUIDANCE_FROM = re.compile(r"\bGuidance\s+From\s*:?\s*\(?\s*~?\s*$", re.I)
_TO = re.compile(r"\bto\b", re.I)
# A parenthetical stating the guidance replaced: no figure in it is the guidance
# ("Sees Adj. EPS To $9.00 (From $8.80 To $8.90) Vs $8.87 Est.").
_OLD_PAREN = re.compile(r"\(\s*(?:Prior|From|Previous(?:ly)?)\b", re.I)
# The verbs of a forecast: a result template never reads a statement they precede.
_FORWARD = re.compile(r"\b(?:Sees|Expects|Guides|Forecasts|Projects)\b", re.I)
# A figure stated in thousands, millions or billions.
_SUFFIXED = re.compile(r"[KMB]\)?$", re.I)
# A figure more than this many times its estimate (or less than its 1/50th)
# is in other units: Benzinga dropped or added a suffix.
UNIT_RATIO: Final = 50.0
# ... except an EPS whose smaller side is under this many dollars: a loss of
# $(0.01) against $(0.60) is a real comparison, not a dropped suffix ...
EPS_RATIO_FLOOR: Final = 0.10
# ... while its larger side is under this many: "Q1 2024 Vs $(0.03) Est." is
# the year read as the EPS (MicroStrategy's $32.52 against $(0.07) compares).
EPS_RATIO_CAP: Final = 100.0
# The metric a statement names: a statement's clause runs to the next one.
_METRIC_WORD = re.compile(r"\b(?:EPS|Sales|Revenues?|Revs?)\b", re.I)
# A "May Not Compare" closing its segment qualifies every statement in it.
_CLOSING_FLAG = re.compile(
    r"May Not Compare(?:\s+(?:To|With)\s+(?:Estimates?|Est\.?))?[\s.,:]*$", re.I
)
# A figure's K/M/B suffix.
SCALE: Final[dict[str, float]] = {"K": 1e3, "M": 1e6, "B": 1e9}
# A surprise is relative to the estimate's magnitude, but never to less than this.
SURPRISE_FLOOR: Final = 0.01
# What a headline's statements are split on.
SEGMENT_SEP: Final = ";"


def money(text: str) -> float | None:
    """'$1.244B' -> 1.244e9, '$(1.26)' -> -1.26, '$756.000M' -> 7.56e8."""
    t = text.replace("$", "").replace(",", "").strip()
    negative = "(" in t or t.startswith("-")
    t = t.replace("(", "").replace(")", "").lstrip("-")
    scale = SCALE.get(t[-1:].upper(), 1.0)
    if t[-1:].upper() in SCALE:
        t = t[:-1]
    try:
        value = float(t) * scale
    except ValueError:
        return None
    return -value if negative else value


def _surprise(actual: float | None, estimate: float | None) -> float | None:
    if actual is None or estimate is None:
        return None
    return (actual - estimate) / max(abs(estimate), SURPRISE_FLOOR)


def _basis(raw: str | None) -> str:
    """'Adj.' -> 'adj', 'Ajd.' -> 'adj', None -> 'gaap'; the other words lowercased."""
    return (raw or "GAAP").rstrip(".").lower().replace("ajd", "adj")


def _period(raw: str) -> str:
    return raw.replace(" ", "").upper()


@dataclass(frozen=True, slots=True)
class EarningsFacts:
    kind: str  # result | guidance
    fields: dict[str, object] = field(default_factory=dict)


def _same_units(
    figures: Sequence[str | None],
    estimate: str,
    value: float | None,
    est: float | None,
    *,
    per_share: bool,
) -> bool:
    """Whether a figure (stated as ``figures``) and its estimate are in the same
    units: all carry a K/M/B suffix or none does, and neither is more than
    :data:`UNIT_RATIO` times the other (a zero compares with anything, and so
    does an EPS whose smaller side is under :data:`EPS_RATIO_FLOOR` and larger
    side under :data:`EPS_RATIO_CAP`)."""
    stated = [t.strip() for t in (*figures, estimate) if t]
    if len({_SUFFIXED.search(t) is not None for t in stated}) > 1:
        return False
    if not value or not est:
        return True
    sides = abs(value), abs(est)
    if per_share and min(sides) < EPS_RATIO_FLOOR and max(sides) < EPS_RATIO_CAP:
        return True
    return 1 / UNIT_RATIO <= abs(value / est) <= UNIT_RATIO


def _flagged(segment: str, start: int, end: int) -> bool:
    """Whether Benzinga flags the statement at ``segment[start:end]`` as not
    comparable: in its clause, which runs to the next metric it names, or at
    the close of the segment."""
    if _CLOSING_FLAG.search(segment):
        return True
    following = _METRIC_WORD.search(segment, end)
    clause_end = following.start() if following else len(segment)
    return NOT_COMPARABLE.search(segment, start, clause_end) is not None


def _not_comparable(fields: dict[str, object], metric: str | None) -> dict[str, object]:
    """``fields`` with the statement's surprise removed and the flags set
    (``metric`` None: a guidance fact's one statement)."""
    if metric is None:
        return fields | {"surprise": None, "not_comparable": True}
    return fields | {
        f"{metric}_surprise": None,
        f"{metric}_not_comparable": True,
        "not_comparable": True,
    }


def _comparison(
    segment: str,
    m: re.Match[str],
    figures: Sequence[str | None],
    estimate: str,
    value: float | None,
    est: float | None,
    *,
    per_share: bool,
) -> bool:
    """Whether the statement ``m`` read compares its figure with its estimate."""
    return not _flagged(segment, m.start(), m.end()) and _same_units(
        figures, estimate, value, est, per_share=per_share
    )


def _sales_after(segment: str, after: int) -> tuple[re.Match[str], str, bool] | None:
    """The sales statement after position ``after`` (match, verdict, estimated),
    unless a forecast verb comes first: "..., Sees Q4 Sales $X vs $Y Est" is guidance."""
    if (s := _SALES.search(segment, after)) is not None:
        verdict, estimated = s["verdict"].lower(), s["ref_kind"].lower().startswith("est")
    elif (s := SALES_EST_FIRST.search(segment, after)) is not None:
        verdict, estimated = "vs", True
    else:
        return None
    if _FORWARD.search(segment, after, s.start()) is not None:
        return None
    return s, verdict, estimated


def _with_sales(fields: dict[str, object], segment: str, after: int) -> dict[str, object]:
    """``fields`` plus the sales figure stated after position ``after``, if any."""
    if (found := _sales_after(segment, after)) is None:
        return fields
    s, verdict, estimated = found
    sales, ref = money(s["sales"]), money(s["ref"])
    stated: dict[str, object] = fields | {
        "sales": sales,
        "sales_estimate": ref if estimated else None,
        "sales_surprise": _surprise(sales, ref) if estimated else None,
        "sales_verdict": verdict,
    }
    if estimated and not _comparison(
        segment, s, (s["sales"],), s["ref"], sales, ref, per_share=False
    ):
        stated = _not_comparable(stated, "sales")
    return stated


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
    if estimated and not _comparison(segment, m, (m["eps"],), m["ref"], eps, ref, per_share=True):
        fields = _not_comparable(fields, "eps")
    return EarningsFacts("result", _with_sales(fields, segment, m.end()))


def _in_old_paren(segment: str, at: int) -> bool:
    """Whether ``segment[at]`` sits in a parenthetical stating the guidance replaced."""
    opened = segment.rfind("(", 0, at)
    return (
        opened >= 0
        and ")" not in segment[opened:at]
        and _OLD_PAREN.match(segment, opened) is not None
    )


def _guided_figure(segment: str, g: re.Match[str]) -> tuple[str | None, str | None]:
    """The figure a ``GUIDE_V4`` read guides to, as (low, high) texts, high None
    for a single figure; (None, None) when no figure qualifies.

    It is the last dollar figure between the read's metric and its estimate
    that is not glued to the token before it and not introduced as the
    guidance being replaced, by the words before it (after the metric) or a
    parenthetical; of "From $80M To $75M" the "To" figure counts. Guidance
    kept as it was (:data:`KEEP_ACTIONS`) may state its range with a
    "Guidance from" that no "to" follows, as its first figure.
    """
    end = g.end("high") if g["high"] else g.end("low")
    kept = g["action"].lower() in KEEP_ACTIONS
    found: tuple[str | None, str | None] = (None, None)
    pos = floor = g.end("metric")
    while (f := _FIGURE.search(segment, pos, end)) is not None:
        before = max(floor, f.start() - OLD_WINDOW)
        if _in_old_paren(segment, f.start()):
            pos = f.end()
        elif _OLD.search(segment, before, f.start()) is None or (
            kept
            and found == (None, None)
            and _GUIDANCE_FROM.search(segment, before, f.start())
            and not _TO.search(segment, f.start(), end)
        ):
            found, pos = (f["low"], f["high"]), f.end()
        elif (
            f["high"]
            and f["join"].strip().lower() == "to"
            and _FROM.search(segment, before, f.start())
        ):
            pos = f.start("high")  # "From ~$0.31 To $0.52-$0.62": read on from the "To"
        else:
            pos = f.end()
    return found


def _guidance(
    g: re.Match[str],
    segment: str,
    prior: Sequence[EarningsFacts],
    *,
    basis: str,
    previous: str | None,
    figure: tuple[str | None, str | None],
) -> EarningsFacts:
    low_text, high_text = figure
    unit = _SUFFIXED.search(high_text.strip()) if high_text else None
    if low_text and unit and _SUFFIXED.search(low_text.strip()) is None:
        low_text += unit[0].rstrip(")")  # "$643-$684M": a range states its unit once
    low = money(low_text) if low_text else None
    high = money(high_text or low_text) if low_text else None
    est = money(g["est"])
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
    metric = fields["metric"]
    if not _comparison(
        segment, g, (low_text, high_text), g["est"], mid, est, per_share=metric == "eps"
    ):
        fields = _not_comparable(fields, None)
    return EarningsFacts("guidance", fields)


def _result_read(rx: re.Pattern[str], segment: str) -> re.Match[str] | None:
    """``rx``'s read of ``segment``, unless a forecast verb precedes it."""
    m = rx.search(segment)
    if m is None or _FORWARD.search(segment, 0, m.start()) is not None:
        return None
    return m


def _segment_fact(segment: str, prior: Sequence[EarningsFacts]) -> EarningsFacts | None:
    """The fact the first template of ``PARSE_ORDER`` reads from ``segment``, if any.

    Guidance comes first: "Sees Q4 Adj EPS $0.20 vs $0.26 Est" is a forecast,
    though its tail reads like a result. Results carry no forecast verb.
    """
    if (g := _GUIDE_RANGE.search(segment)) is not None:
        figure = (g["low"], g["high"])
        return _guidance(
            g, segment, prior, basis=_basis(g["basis"]), previous=g["old"], figure=figure
        )
    g = GUIDE_V4.search(segment)
    # Up to the read's first character: in "$3.8B +/-$300M" its figure starts
    # at the tolerance's own "-".
    if g is not None and _TOLERANCE_GAP.search(segment, g.end("metric"), g.start("low") + 1):
        g = None
    if g is not None:
        word = _BASIS_WORD.search(segment, g.end("action"), g.start("metric"))
        basis = _basis(word[0] if word else None)
        figure = _guided_figure(segment, g)
        return _guidance(g, segment, prior, basis=basis, previous=None, figure=figure)
    if (m := _result_read(_RESULT, segment)) is not None:
        estimated = m["ref_kind"].lower().startswith("est")
        return _eps_result(m, segment, verdict=m["verdict"].lower(), estimated=estimated)
    if (m := _result_read(RESULT_EST_FIRST, segment)) is not None:
        return _eps_result(m, segment, verdict="vs", estimated=True)
    if (m := _result_read(_INLINE, segment)) is not None:
        eps = money(m["eps"])
        fields: dict[str, object] = {
            "period": _period(m["period"]),
            "basis": _basis(m["basis"]),
            "eps": eps,
            "eps_estimate": eps,
            "eps_surprise": 0.0,
            "eps_verdict": "inline",
        }
        if _flagged(segment, m.start(), m.end()):
            fields = _not_comparable(fields, "eps")
        return EarningsFacts("result", _with_sales(fields, segment, m.end()))
    if (m := _result_read(SALES_ONLY, segment)) is not None:
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
        if not _comparison(segment, m, (m["sales"],), m["ref"], sales, ref, per_share=False):
            fields = _not_comparable(fields, "sales")
        return EarningsFacts("result", fields)
    return None


def parse_headline(headline: str) -> list[EarningsFacts]:
    """Every result or guidance statement in one headline (segments split on ';')."""
    out: list[EarningsFacts] = []
    for segment in headline.split(SEGMENT_SEP):
        if (fact := _segment_fact(segment, out)) is not None:
            out.append(fact)
    return out


# v2: the 2016-2019 formats ("EPS $1.27 vs $1.12 Est.", "EPS $0.22, Inline",
# "Ajd. EPS"), found validating the parser on the full history.
# v3: a year after the quarter ("Q4 2023 Adj EPS"), "Adj $0.65 Beats", "Revenue
# ... Misses" -- the commonest 2023-2026 misses.
# v4: sales-only results, "EPS $X vs. Est. $Y" (and its "Rev. $X vs. Est. $Y"),
# guidance in other verbs ("Expects", "Narrows", "Versus Consensus Of", "(Est
# $Y)") with its figure re-read, "May Not Compare" and unit mismatches without
# a surprise, no result read after a forecast verb, and the sales stated after
# an inline EPS. The news engine reads v4; v3 rows stay in event_facts.
EXTRACTOR_V3: Final = "benzinga-earnings-v3"
# v4's family: the stored label is this, "+", then the parser's pin (EXTRACTOR).
EXTRACTOR_V4: Final = "benzinga-earnings-v4"


def sources() -> dict[str, str]:
    """Every pattern's source text, the parse order, the constants that shape a
    read (the windows, ratios and floors, the verbs, the scale and the separator)
    and the extractor's family, for the pins."""
    patterns = {
        "_RESULT": _RESULT,
        "_INLINE": _INLINE,
        "_SALES": _SALES,
        "_GUIDE_RANGE": _GUIDE_RANGE,
        "SALES_ONLY": SALES_ONLY,
        "RESULT_EST_FIRST": RESULT_EST_FIRST,
        "SALES_EST_FIRST": SALES_EST_FIRST,
        "GUIDE_V4": GUIDE_V4,
        "NOT_COMPARABLE": NOT_COMPARABLE,
        "GUIDE_UNPARSED": GUIDE_UNPARSED,
        "_BASIS_WORD": _BASIS_WORD,
        "_TOLERANCE_GAP": _TOLERANCE_GAP,
        "_FIGURE": _FIGURE,
        "_OLD": _OLD,
        "_FROM": _FROM,
        "_GUIDANCE_FROM": _GUIDANCE_FROM,
        "_FORWARD": _FORWARD,
        "_SUFFIXED": _SUFFIXED,
        "_METRIC_WORD": _METRIC_WORD,
        "_CLOSING_FLAG": _CLOSING_FLAG,
        "_OLD_PAREN": _OLD_PAREN,
        "_TO": _TO,
    }
    return {name: rx.pattern for name, rx in patterns.items()} | {
        "PARSE_ORDER": ",".join(PARSE_ORDER),
        "UNIT_RATIO": repr(UNIT_RATIO),
        "EPS_RATIO_FLOOR": repr(EPS_RATIO_FLOOR),
        "EPS_RATIO_CAP": repr(EPS_RATIO_CAP),
        "OLD_WINDOW": repr(OLD_WINDOW),
        "KEEP_ACTIONS": ",".join(sorted(KEEP_ACTIONS)),
        "SCALE": json.dumps(SCALE, sort_keys=True),
        "SURPRISE_FLOOR": repr(SURPRISE_FLOOR),
        "SEGMENT_SEP": SEGMENT_SEP,
        "EXTRACTOR_V4": EXTRACTOR_V4,
    }


# The parser's pin in the news engine's pre-registration: a change to any
# pattern or constant above is a new parser, so a new trial.
PARSER_SHA: Final = hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()[:12]
# The label every fact this parser stores carries, and the one every reader
# selects: a parser change is a new label, so no reader sees a fact an older
# parser read, and extract_all re-reads every event under the new one.
EXTRACTOR: Final = f"{EXTRACTOR_V4}+{PARSER_SHA}"


async def extract_all(engine: Any, *, batch: int = 5000) -> int:
    """Parse every news event this parser (``EXTRACTOR``) has not read; returns
    facts stored.

    An event with no earnings statement is marked with a ``none`` fact so it
    is not re-read, and so the parse rate can be measured. Once every event is
    read, the facts of every older v4 label are deleted (:func:`drop_superseded`);
    v3's stay.
    """
    from sqlalchemy import text

    stored, after = 0, 0
    while True:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT e.id, e.payload->>'headline' AS h FROM events e "
                        "WHERE e.kind = 'news' AND e.id > :after AND NOT EXISTS ("
                        "SELECT 1 FROM event_facts f "
                        "WHERE f.event_id = e.id AND f.extractor = :x) ORDER BY e.id LIMIT :n"
                    ),
                    {"x": EXTRACTOR, "after": after, "n": batch},
                )
            ).all()
        if not rows:
            await drop_superseded(engine, batch=batch)
            return stored
        after = rows[-1].id
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


async def drop_superseded(engine: Any, *, batch: int = 5000) -> int:
    """Delete the facts of every v4 label but ``EXTRACTOR``, ``batch`` rows a
    transaction; returns rows deleted. The bare ``benzinga-earnings-v4`` of the
    parsers that stored no pin counts as superseded; v3's rows stay."""
    from sqlalchemy import text

    dropped, after = 0, 0
    while True:
        async with engine.begin() as conn:
            ids = (
                (
                    await conn.execute(
                        text(
                            "DELETE FROM event_facts WHERE id IN (SELECT id FROM event_facts "
                            "WHERE id > :after AND (extractor = :v4 OR extractor LIKE :older) "
                            "AND extractor <> :x ORDER BY id LIMIT :n) RETURNING id"
                        ),
                        {
                            "after": after,
                            "v4": EXTRACTOR_V4,
                            "older": f"{EXTRACTOR_V4}+%",
                            "x": EXTRACTOR,
                            "n": batch,
                        },
                    )
                )
                .scalars()
                .all()
            )
        if not ids:
            return dropped
        dropped += len(ids)
        after = max(ids)
