"""AAOIFI Standard 30 (Monetisation / Tawarruq) edge-case engine.

Round-5 Wave 1.C primitive. Standard 30 governs commodity Murabaha
(Tawarruq) and reverse-Murabaha — the load-bearing structures behind
any halal short-term liquidity product (Islamic credit cards, working-
capital sukuk, treasury yield substitutes). Without a clause-level
encoding, the bot can't tell the operator *why* a proposed Tawarruq
structure is compliant or impermissible.

Three load-bearing prohibitions Standard 30 codifies:

1. **Bay' al-Inah** — sale + immediate buy-back of the same commodity
   between the same two parties at different prices, manufactured to
   produce a riba-equivalent cash flow. Always impermissible
   (Standard 30, cl. 4/3).
2. **Constructive possession** — the commodity must be in the buyer's
   constructive possession (qabd ma'nawi) before resale. Direct
   pre-arranged tri-party loops without real possession are
   impermissible (Standard 30, cl. 4/5).
3. **Rate-cap guidance** — the markup may be benchmarked to a market
   rate but the deal must remain a *sale* (one execution, one
   settlement) rather than a perpetual rate-tracking instrument
   (Standard 30, cl. 5/2).

This module ships the structural encoding plus a screener that runs
candidate Tawarruq structures against the catalogue and returns the
specific violations.

Pinned semantics:

- **Closed-set TawarruqViolation ladder** — five enumerated violations.
- **Closed-set TawarruqStructure ladder** — six recognised structures
  (organised / reverse / direct / agent / commodity_sale / inah).
- **Inah is *always* a violation** — even in isolation, an Inah-shaped
  structure flips the assessment to non-compliant, mirroring the fiqh
  rule that intent + form together produce the haram outcome.
- **No-secret-leak pin** on render output.
- The catalogue is import-time frozen.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class TawarruqViolation(str, Enum):
    """Closed-set catalogue of Standard-30 violations."""

    BAY_AL_INAH = "bay_al_inah"
    NO_CONSTRUCTIVE_POSSESSION = "no_constructive_possession"
    PRE_ARRANGED_BUYBACK = "pre_arranged_buyback"
    RATE_TRACKING_PERPETUAL = "rate_tracking_perpetual"
    SAME_COUNTERPARTY_LOOP = "same_counterparty_loop"


class TawarruqStructure(str, Enum):
    """The structures Standard 30 names + screens."""

    ORGANISED_TAWARRUQ = "organised_tawarruq"
    REVERSE_MURABAHA = "reverse_murabaha"
    DIRECT_TAWARRUQ = "direct_tawarruq"
    AGENT_BASED_TAWARRUQ = "agent_based_tawarruq"
    COMMODITY_SALE = "commodity_sale"
    INAH = "inah"  # always non-compliant


_CLAUSE_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _clause_sort_key(cid: str) -> tuple[int, ...]:
    return tuple(int(seg) for seg in cid.split("."))


@dataclass(frozen=True)
class StandardClause:
    """A single operative clause from AAOIFI Standard 30."""

    clause_id: str
    title: str
    violation: TawarruqViolation
    summary: str

    def __post_init__(self) -> None:
        if not self.clause_id or not _CLAUSE_ID_RE.match(self.clause_id):
            raise ValueError(f"clause_id must be dotted-numeric; got {self.clause_id!r}")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")


_RAW_CLAUSES: tuple[StandardClause, ...] = (
    StandardClause(
        clause_id="4.3",
        title="Bay' al-Inah is impermissible",
        violation=TawarruqViolation.BAY_AL_INAH,
        summary="A sale followed by immediate buy-back of the same commodity between the "
        "same two parties at different prices is haram. The form is a sale; the "
        "substance is interest.",
    ),
    StandardClause(
        clause_id="4.5",
        title="Constructive possession (qabd ma'nawi) required",
        violation=TawarruqViolation.NO_CONSTRUCTIVE_POSSESSION,
        summary="The buyer must take constructive possession of the commodity (the right "
        "to dispose of it, with risk of loss transferred) before reselling. A flow "
        "where the commodity is sold by the buyer before any possession is impermissible.",
    ),
    StandardClause(
        clause_id="4.7",
        title="No pre-arranged buyback by the original seller",
        violation=TawarruqViolation.PRE_ARRANGED_BUYBACK,
        summary="The third-party buyer in organised Tawarruq must not be a pre-arranged "
        "agent of the original seller; a contractual loop reduces the structure to "
        "Inah.",
    ),
    StandardClause(
        clause_id="5.2",
        title="Rate may be benchmarked, transaction remains a sale",
        violation=TawarruqViolation.RATE_TRACKING_PERPETUAL,
        summary="The mark-up may reference a market rate for transparency but the "
        "structure must terminate as a single sale; perpetually rate-tracking "
        "evergreen rollovers tip into riba.",
    ),
    StandardClause(
        clause_id="5.5",
        title="Counterparty independence",
        violation=TawarruqViolation.SAME_COUNTERPARTY_LOOP,
        summary="The third-party purchaser must be independent of the original seller; "
        "shared ownership / control collapses the structure into Inah.",
    ),
)


CLAUSES: tuple[StandardClause, ...] = tuple(
    sorted(_RAW_CLAUSES, key=lambda c: _clause_sort_key(c.clause_id))
)


def clause_by_id(clause_id: str) -> StandardClause | None:
    for c in CLAUSES:
        if c.clause_id == clause_id:
            return c
    return None


def clauses_for_violation(v: TawarruqViolation) -> tuple[StandardClause, ...]:
    return tuple(c for c in CLAUSES if c.violation is v)


@dataclass(frozen=True)
class TawarruqInputs:
    """Inputs describing a candidate Tawarruq / Murabaha structure."""

    structure: TawarruqStructure
    same_counterparty_buyback: bool = False
    pre_arranged_third_party: bool = False
    constructive_possession_taken: bool = True
    independent_third_party: bool = True
    perpetual_rate_tracking: bool = False
    markup_bps: float = 0.0  # informational; cap policy elsewhere

    def __post_init__(self) -> None:
        if self.markup_bps < 0:
            raise ValueError("markup_bps must be non-negative")


@dataclass(frozen=True)
class TawarruqAssessment:
    """Result of running a structure against Standard 30."""

    structure: TawarruqStructure
    violations: frozenset[TawarruqViolation]
    is_compliant: bool

    def __post_init__(self) -> None:
        if self.is_compliant and self.violations:
            raise ValueError("is_compliant=True but violations non-empty")
        if (not self.is_compliant) and not self.violations:
            raise ValueError("is_compliant=False but violations empty")


def screen_tawarruq(inputs: TawarruqInputs) -> TawarruqAssessment:
    """Run the structure against Standard 30 and return the assessment."""
    violations: set[TawarruqViolation] = set()

    # INAH is always non-compliant — fiqh rule.
    if inputs.structure is TawarruqStructure.INAH:
        violations.add(TawarruqViolation.BAY_AL_INAH)

    if inputs.same_counterparty_buyback:
        violations.add(TawarruqViolation.BAY_AL_INAH)
    if not inputs.constructive_possession_taken:
        violations.add(TawarruqViolation.NO_CONSTRUCTIVE_POSSESSION)
    if inputs.pre_arranged_third_party:
        violations.add(TawarruqViolation.PRE_ARRANGED_BUYBACK)
    if not inputs.independent_third_party:
        violations.add(TawarruqViolation.SAME_COUNTERPARTY_LOOP)
    if inputs.perpetual_rate_tracking:
        violations.add(TawarruqViolation.RATE_TRACKING_PERPETUAL)

    return TawarruqAssessment(
        structure=inputs.structure,
        violations=frozenset(violations),
        is_compliant=len(violations) == 0,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub(text: str) -> str:
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(a: TawarruqAssessment) -> str:
    emoji = "✅" if a.is_compliant else "❌"
    lines = [f"{emoji} structure: {a.structure.value}"]
    for v in sorted(a.violations, key=lambda x: x.value):
        clauses = clauses_for_violation(v)
        cite = f"§{clauses[0].clause_id}" if clauses else "§?"
        lines.append(f"  • {cite} {v.value}")
    return _scrub("\n".join(lines))


def render_clause(c: StandardClause) -> str:
    return _scrub(f"§{c.clause_id} — {c.title} [{c.violation.value}]")


def render_coverage_matrix(violations_engaged: Iterable[TawarruqViolation] | None = None) -> str:
    engaged = (
        frozenset(violations_engaged)
        if violations_engaged is not None
        else frozenset(c.violation for c in CLAUSES)
    )
    engaged_clauses = sum(1 for c in CLAUSES if c.violation in engaged)
    header = (
        f"AAOIFI Standard 30 (Monetisation / Tawarruq) — "
        f"{engaged_clauses}/{len(CLAUSES)} clauses engaged"
    )
    lines = [header, "-" * len(header)]
    for c in CLAUSES:
        marker = "✅" if c.violation in engaged else "  "
        lines.append(f"{marker} {render_clause(c)}")
    return "\n".join(lines)
