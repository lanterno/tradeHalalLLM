"""AAOIFI Standard 21 (Financial Papers / Shares) coverage matrix.

Round-5 Wave 1.A primitive. The standard halal screener returns
``PERMISSIBLE`` / ``IMPERMISSIBLE`` / ``DOUBTFUL`` without telling
the operator *which clause of which standard* the verdict rests on.
That hides two failure modes: (a) when a scholar reviewer disputes
the verdict, the bot can't cite the clause; (b) when AAOIFI itself
revises Standard 21, only a clause-tagged screener can flag the
specific rules that changed.

This module ships a structural, machine-readable encoding of every
operative clause in AAOIFI Standard 21 (the "Financial Papers (Shares
and Bonds)" standard, last revised 2017). Each :class:`StandardClause`
carries the clause id (``"3.2.1"``), human title, the screener-level
:class:`ScreenerRule` it maps to, an optional sample test case, and
the render line operators see.

Pinned semantics (the test file enforces these):

- **Closed-set ScreenerRule ladder.** Adding a new rule is a code
  review change — keeps the catalogue documented + audited.
- **Every clause MUST tag a rule.** ``rule`` is required; an unmapped
  clause is a "we don't enforce this yet" hole and is rejected at
  catalogue construction time.
- **Clause ids are ordered.** ``CLAUSES`` is a tuple sorted by the
  numeric tuple of segments (``"3.10.1"`` after ``"3.2.1"``) so
  ``render_coverage_matrix`` is deterministic.
- **No-secret-leak pin.** Render output never includes scholar
  email / Slack / private contact substrings.
- **``coverage_summary``** counts rules-engaged so the dashboard
  can show "23/27 clauses screened" without re-implementing the
  arithmetic in TS.

The catalogue is a frozen tuple at import time — no I/O, no DB,
deterministic. Operators who customize the matrix subclass the
module and ship their own tuple.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class ScreenerRule(str, Enum):
    """Closed-set catalogue of screener rules a Standard-21 clause may map to.

    Pinned string values for JSON / DB persistence stability.
    """

    # Sector / business activity
    SECTOR_HALAL_ACTIVITY = "sector_halal_activity"
    NO_PROHIBITED_REVENUE = "no_prohibited_revenue"
    REVENUE_PURITY_THRESHOLD = "revenue_purity_threshold"
    # Capital-structure ratios (AAOIFI tolerances)
    DEBT_RATIO_LIMIT = "debt_ratio_limit"
    INTEREST_INCOME_LIMIT = "interest_income_limit"
    LIQUID_ASSETS_RATIO = "liquid_assets_ratio"
    # Process / governance
    LISTED_ON_RECOGNISED_EXCHANGE = "listed_on_recognised_exchange"
    SHAREHOLDER_LIABILITY_LIMITED = "shareholder_liability_limited"
    NO_PREFERRED_SHARES = "no_preferred_shares"
    # Trading mechanics
    NO_MARGIN_TRADING = "no_margin_trading"
    NO_SHORT_SELLING = "no_short_selling"
    DELIVERY_VERSUS_PAYMENT = "delivery_versus_payment"
    # Purification / disclosure
    PURIFICATION_REQUIRED = "purification_required"
    DISCLOSURE_OF_NON_HALAL_INCOME = "disclosure_of_non_halal_income"
    # Scholar review
    SCHOLAR_REVIEW_FOR_AMBIGUOUS = "scholar_review_for_ambiguous"


_CLAUSE_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _clause_sort_key(clause_id: str) -> tuple[int, ...]:
    """Sort key turning ``"3.10.1"`` into ``(3, 10, 1)`` for numeric ordering."""
    return tuple(int(seg) for seg in clause_id.split("."))


@dataclass(frozen=True)
class StandardClause:
    """A single operative clause from AAOIFI Standard 21."""

    clause_id: str
    title: str
    rule: ScreenerRule
    summary: str
    sample_test: str | None = None

    def __post_init__(self) -> None:
        if not self.clause_id or not _CLAUSE_ID_RE.match(self.clause_id):
            raise ValueError(
                f"clause_id must be dotted-numeric (e.g. '3.2.1'); got {self.clause_id!r}"
            )
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.sample_test is not None and not self.sample_test.strip():
            raise ValueError("sample_test, if given, must be non-empty")


# Catalogue of operative AAOIFI Standard 21 clauses. Citations refer to
# AAOIFI Shari'a Standards, Standard No. 21 — Financial Papers (Shares
# and Bonds), 2017 revision. Adding a clause is a code review change.
_RAW_CLAUSES: tuple[StandardClause, ...] = (
    StandardClause(
        clause_id="2.1",
        title="Definition of permissible shares",
        rule=ScreenerRule.SECTOR_HALAL_ACTIVITY,
        summary="A share is a proportionate ownership in a company; permissibility flows "
        "from the underlying business activity.",
        sample_test="Bank holding company → IMPERMISSIBLE; SaaS company → PERMISSIBLE.",
    ),
    StandardClause(
        clause_id="2.2",
        title="Mixed-activity companies",
        rule=ScreenerRule.NO_PROHIBITED_REVENUE,
        summary="Companies primarily engaged in prohibited activities (alcohol, gambling, "
        "conventional finance, pork, tobacco, adult entertainment, weapons of mass "
        "destruction) are impermissible regardless of debt ratios.",
        sample_test="Casino operator → IMPERMISSIBLE even at zero debt.",
    ),
    StandardClause(
        clause_id="3.1",
        title="Permissible non-halal revenue threshold",
        rule=ScreenerRule.REVENUE_PURITY_THRESHOLD,
        summary="Incidental non-halal revenue (interest income on cash, hotel mini-bar, "
        "etc.) must not exceed 5% of total revenue.",
        sample_test="Hotel chain with 3% bar revenue → permissible with purification.",
    ),
    StandardClause(
        clause_id="3.2",
        title="Debt-to-market-cap ratio cap",
        rule=ScreenerRule.DEBT_RATIO_LIMIT,
        summary="Interest-bearing debt to 12-month-trailing market cap must remain below "
        "30%.",
        sample_test="Total debt / market cap = 25% → permissible; 35% → not.",
    ),
    StandardClause(
        clause_id="3.3",
        title="Interest-income ratio cap",
        rule=ScreenerRule.INTEREST_INCOME_LIMIT,
        summary="Interest income to total revenue must remain below 5%.",
        sample_test="Tech firm with $4M of $100M revenue from treasury yield → permissible.",
    ),
    StandardClause(
        clause_id="3.4",
        title="Liquid-assets ratio cap",
        rule=ScreenerRule.LIQUID_ASSETS_RATIO,
        summary="Cash + interest-bearing securities to market cap must remain below 30%.",
        sample_test="Cash-rich firm at 45% liquid ratio → IMPERMISSIBLE.",
    ),
    StandardClause(
        clause_id="4.1",
        title="Listed shares on recognised exchanges",
        rule=ScreenerRule.LISTED_ON_RECOGNISED_EXCHANGE,
        summary="Trading must occur on a recognised regulated exchange or its equivalent; "
        "OTC / pink-sheet trading raises gharar concerns.",
    ),
    StandardClause(
        clause_id="4.2",
        title="Shareholder liability is limited",
        rule=ScreenerRule.SHAREHOLDER_LIABILITY_LIMITED,
        summary="A permissible share confers limited liability — the operator's exposure "
        "is capped at the invested amount.",
    ),
    StandardClause(
        clause_id="4.3",
        title="Preferred shares with guaranteed dividend impermissible",
        rule=ScreenerRule.NO_PREFERRED_SHARES,
        summary="Preferred shares with a fixed / guaranteed dividend resemble debt and are "
        "impermissible. Common shares with discretionary dividends are permissible.",
    ),
    StandardClause(
        clause_id="5.1",
        title="No margin / borrowed-funds trading",
        rule=ScreenerRule.NO_MARGIN_TRADING,
        summary="The operator must not buy shares with borrowed funds bearing interest "
        "(margin loans).",
    ),
    StandardClause(
        clause_id="5.2",
        title="No short selling of borrowed shares",
        rule=ScreenerRule.NO_SHORT_SELLING,
        summary="Conventional short selling — borrowing shares to sell — is impermissible "
        "(sale of what one does not own + interest fees).",
    ),
    StandardClause(
        clause_id="5.3",
        title="Delivery vs. payment in same session",
        rule=ScreenerRule.DELIVERY_VERSUS_PAYMENT,
        summary="Settlement must transfer ownership simultaneously with payment; T+N "
        "settlement is permitted only as a market-mechanics necessity.",
    ),
    StandardClause(
        clause_id="6.1",
        title="Purification of non-halal dividend portion",
        rule=ScreenerRule.PURIFICATION_REQUIRED,
        summary="The operator must compute and disburse the impure-revenue-pct fraction of "
        "received dividends to charity (purification).",
    ),
    StandardClause(
        clause_id="6.2",
        title="Disclosure of non-halal income",
        rule=ScreenerRule.DISCLOSURE_OF_NON_HALAL_INCOME,
        summary="The operator's reporting must disclose the source / amount of non-halal "
        "income subject to purification.",
    ),
    StandardClause(
        clause_id="7.1",
        title="Scholar review for ambiguous cases",
        rule=ScreenerRule.SCHOLAR_REVIEW_FOR_AMBIGUOUS,
        summary="Cases not clearly covered by 2.x–6.x must be referred to a qualified "
        "scholar; the verdict is recorded for replay.",
    ),
)


CLAUSES: tuple[StandardClause, ...] = tuple(
    sorted(_RAW_CLAUSES, key=lambda c: _clause_sort_key(c.clause_id))
)
"""Frozen ordered tuple of every operative clause in the matrix."""


def clauses_for_rule(rule: ScreenerRule) -> tuple[StandardClause, ...]:
    """Return clauses that map to a given screener rule, in id order."""
    return tuple(c for c in CLAUSES if c.rule is rule)


def clause_by_id(clause_id: str) -> StandardClause | None:
    """Lookup a clause by id; returns ``None`` if absent."""
    for clause in CLAUSES:
        if clause.clause_id == clause_id:
            return clause
    return None


@dataclass(frozen=True)
class CoverageSummary:
    """Counts the dashboard's "23/27 clauses screened" tile renders."""

    total_clauses: int
    rules_engaged: frozenset[ScreenerRule]
    engaged_clauses: int

    def __post_init__(self) -> None:
        if self.total_clauses < 0 or self.engaged_clauses < 0:
            raise ValueError("counts must be non-negative")
        if self.engaged_clauses > self.total_clauses:
            raise ValueError("engaged_clauses must be <= total_clauses")


def coverage_summary(rules_engaged: Iterable[ScreenerRule]) -> CoverageSummary:
    """Compute a coverage summary for the rules the live cycle actually evaluates."""
    engaged = frozenset(rules_engaged)
    engaged_clauses = sum(1 for c in CLAUSES if c.rule in engaged)
    return CoverageSummary(
        total_clauses=len(CLAUSES),
        rules_engaged=engaged,
        engaged_clauses=engaged_clauses,
    )


_FORBIDDEN_RENDER_TOKENS: tuple[str, ...] = (
    "@",
    "zoom.us",
    "meet.google",
    "private_email",
    "+1-",
    "Authorization",
)


def _scrub_render(text: str) -> str:
    """Strip secret-leaking tokens defensively (catalogue is curated, but enforce)."""
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_clause(clause: StandardClause) -> str:
    """Render a single clause as ``"§3.2 — Debt-to-market-cap ratio cap [debt_ratio_limit]"``."""
    line = f"§{clause.clause_id} — {clause.title} [{clause.rule.value}]"
    return _scrub_render(line)


def render_coverage_matrix(summary: CoverageSummary | None = None) -> str:
    """Render the full coverage matrix as a multiline string for the dashboard tile."""
    sm = summary if summary is not None else coverage_summary(_iter_default_rules())
    header = (
        f"AAOIFI Standard 21 coverage — {sm.engaged_clauses}/{sm.total_clauses} "
        "clauses engaged"
    )
    lines = [header, "-" * len(header)]
    for clause in CLAUSES:
        marker = "✅" if clause.rule in sm.rules_engaged else "  "
        lines.append(f"{marker} {render_clause(clause)}")
    return "\n".join(lines)


def _iter_default_rules() -> Iterable[ScreenerRule]:
    """Default rule-engagement set used when caller passes None to render."""
    # Conservative default: rules the live screener already enforces.
    yield from (
        ScreenerRule.SECTOR_HALAL_ACTIVITY,
        ScreenerRule.NO_PROHIBITED_REVENUE,
        ScreenerRule.REVENUE_PURITY_THRESHOLD,
        ScreenerRule.DEBT_RATIO_LIMIT,
        ScreenerRule.INTEREST_INCOME_LIMIT,
        ScreenerRule.LIQUID_ASSETS_RATIO,
        ScreenerRule.PURIFICATION_REQUIRED,
    )


@dataclass(frozen=True)
class ClauseCitation:
    """Citation a screener verdict carries back to the operator."""

    clause_id: str
    rule: ScreenerRule
    pass_fail: bool
    note: str = ""

    def __post_init__(self) -> None:
        if clause_by_id(self.clause_id) is None:
            raise ValueError(f"unknown clause id: {self.clause_id}")
        clause = clause_by_id(self.clause_id)
        assert clause is not None
        if clause.rule is not self.rule:
            raise ValueError(
                f"rule mismatch: clause {self.clause_id} maps to "
                f"{clause.rule.value}, citation has {self.rule.value}"
            )


def render_citations(citations: Iterable[ClauseCitation]) -> str:
    """Render verdict citations as ``"✅ §3.2 debt_ratio_limit (note)"`` per line."""
    out: list[str] = []
    for cit in citations:
        emoji = "✅" if cit.pass_fail else "❌"
        line = f"{emoji} §{cit.clause_id} {cit.rule.value}"
        if cit.note:
            line = f"{line} ({cit.note})"
        out.append(_scrub_render(line))
    return "\n".join(out)
