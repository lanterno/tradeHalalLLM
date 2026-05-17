"""AAOIFI Standard 17 (Investment Sukuk) coverage matrix.

Round-5 Wave 1.B primitive. The sukuk vertical (Round-5 Wave 3 — sukuk
universe ingestion, pricing, allocation) needs a clause-level encoding
of AAOIFI Standard 17 the screener can cite, mirror of the equity-side
``halal/aaoifi_standard_21.py``.

This module ships the structural encoding of every operative clause in
AAOIFI Standard 17 (Investment Sukuk, last revised 2017). Each
:class:`SukukClause` carries the clause id, title, the
:class:`SukukRule` it maps to, summary, and an optional sample test
case. ``CLAUSES`` is import-time frozen and sorted by clause id.

Pinned semantics (the test file enforces these):

- **Closed-set SukukRule ladder.**
- **Closed-set SukukType ladder** for the seven AAOIFI-recognised sukuk
  structures (Ijara / Mudarabah / Musharakah / Murabaha / Salam /
  Istisna / Wakalah-bil-istithmar). Adding a new structure is a code
  review change.
- **No tradability of debt-only sukuk.** ``is_tradable_in_secondary``
  returns False for pure-Murabaha sukuk (Standard 17 cl. 5.1.8).
- **No-secret-leak pin** on render output.
- **Tangible-asset-ratio gate.** Operators wanting tradable secondary-
  market sukuk must hold ≥51% in non-debt-backed instruments
  (Standard 17 cl. 5.1.8 supplemented by the 2008 AAOIFI clarification
  on tangibility).

The catalogue is a frozen tuple at import time — no I/O, no DB.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class SukukRule(str, Enum):
    """Closed-set screener rules a Standard-17 clause may map to."""

    UNDERLYING_ASSET_HALAL = "underlying_asset_halal"
    OWNERSHIP_TRANSFER_REAL = "ownership_transfer_real"
    NO_GUARANTEED_PRINCIPAL = "no_guaranteed_principal"
    PROFIT_FROM_REAL_ACTIVITY = "profit_from_real_activity"
    LOSS_SHARING_PROPORTIONAL = "loss_sharing_proportional"
    NO_INTEREST_RATE_LINK = "no_interest_rate_link"
    TANGIBLE_ASSET_RATIO = "tangible_asset_ratio"
    SECONDARY_MARKET_TRADABILITY = "secondary_market_tradability"
    PURPOSE_LISTED_HALAL = "purpose_listed_halal"
    SHARIA_BOARD_OPINION = "sharia_board_opinion"
    REDEMPTION_AT_FAIR_VALUE = "redemption_at_fair_value"
    PROCEEDS_USAGE_DISCLOSED = "proceeds_usage_disclosed"


class SukukType(str, Enum):
    """The seven AAOIFI-recognised sukuk structures.

    Pinned string values. Each carries a different secondary-market
    tradability profile under Standard 17.
    """

    IJARA = "ijara"
    MUDARABAH = "mudarabah"
    MUSHARAKAH = "musharakah"
    MURABAHA = "murabaha"
    SALAM = "salam"
    ISTISNA = "istisna"
    WAKALAH = "wakalah"


# Sukuk types whose secondary-market trading is permissible under
# Standard 17 cl. 5.1.8. Pure-Murabaha and Salam are debt-instruments
# and may only be transferred at face value (effectively assignment of
# debt, not trading).
TRADABLE_IN_SECONDARY: frozenset[SukukType] = frozenset(
    {
        SukukType.IJARA,
        SukukType.MUDARABAH,
        SukukType.MUSHARAKAH,
        SukukType.WAKALAH,
        SukukType.ISTISNA,
    }
)
"""Frozenset of sukuk types tradable on the secondary market.

Pinned closed-set: Ijara / Mudarabah / Musharakah / Wakalah / Istisna.
Murabaha + Salam excluded.
"""


_CLAUSE_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _clause_sort_key(cid: str) -> tuple[int, ...]:
    return tuple(int(seg) for seg in cid.split("."))


@dataclass(frozen=True)
class SukukClause:
    """A single operative clause from AAOIFI Standard 17."""

    clause_id: str
    title: str
    rule: SukukRule
    summary: str
    sample_test: str | None = None

    def __post_init__(self) -> None:
        if not self.clause_id or not _CLAUSE_ID_RE.match(self.clause_id):
            raise ValueError(
                f"clause_id must be dotted-numeric; got {self.clause_id!r}"
            )
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if self.sample_test is not None and not self.sample_test.strip():
            raise ValueError("sample_test, if given, must be non-empty")


_RAW_CLAUSES: tuple[SukukClause, ...] = (
    SukukClause(
        clause_id="2.1",
        title="Sukuk represent ownership in tangible assets / usufruct / services",
        rule=SukukRule.OWNERSHIP_TRANSFER_REAL,
        summary="Sukuk certificates of equal value represent undivided shares in the "
        "ownership of tangible assets, usufruct, or services — not debt obligations.",
        sample_test="Conventional bond mislabelled 'sukuk' → IMPERMISSIBLE.",
    ),
    SukukClause(
        clause_id="2.2",
        title="Underlying assets must be halal",
        rule=SukukRule.UNDERLYING_ASSET_HALAL,
        summary="The underlying business / asset must itself be Shariah-compliant. A "
        "sukuk whose proceeds finance a brewery is impermissible regardless of "
        "structure.",
    ),
    SukukClause(
        clause_id="2.3",
        title="Purpose / use of proceeds disclosed",
        rule=SukukRule.PROCEEDS_USAGE_DISCLOSED,
        summary="Issuer prospectus must disclose the purpose of the issuance.",
    ),
    SukukClause(
        clause_id="3.1",
        title="No guaranteed principal repayment",
        rule=SukukRule.NO_GUARANTEED_PRINCIPAL,
        summary="The issuer cannot guarantee return of principal at maturity (would "
        "convert the sukuk into an interest-bearing debt). Third-party guarantees are "
        "permitted only if they are unrelated and unrequited.",
    ),
    SukukClause(
        clause_id="3.2",
        title="Profit from real activity, not interest",
        rule=SukukRule.PROFIT_FROM_REAL_ACTIVITY,
        summary="Returns must come from rent (Ijara), profit (Mudarabah/Musharakah/"
        "Wakalah-investment), or sale margin — not from a fixed coupon resembling "
        "interest.",
    ),
    SukukClause(
        clause_id="3.3",
        title="No interest-rate benchmark linkage",
        rule=SukukRule.NO_INTEREST_RATE_LINK,
        summary="Profit-sharing rates may reference benchmarks for transparency but the "
        "actual payout must reflect realised profit, not a rate-derived calculation.",
    ),
    SukukClause(
        clause_id="3.4",
        title="Loss sharing proportional",
        rule=SukukRule.LOSS_SHARING_PROPORTIONAL,
        summary="Sukukholders bear loss in proportion to their ownership; capital-loss "
        "must not be shifted entirely to the issuer.",
    ),
    SukukClause(
        clause_id="4.1",
        title="Permissible structures enumerated",
        rule=SukukRule.OWNERSHIP_TRANSFER_REAL,
        summary="AAOIFI recognises seven primary structures: Ijara, Mudarabah, "
        "Musharakah, Murabaha, Salam, Istisna, Wakalah-bil-istithmar.",
    ),
    SukukClause(
        clause_id="4.2",
        title="Tangibility threshold for tradability",
        rule=SukukRule.TANGIBLE_ASSET_RATIO,
        summary="To be tradable on the secondary market, the underlying portfolio must "
        "include at least 51% tangible assets / usufruct (per the 2008 AAOIFI "
        "clarification).",
        sample_test="Sukuk backed 30% real estate + 70% Murabaha receivables → not "
        "tradable on secondary market.",
    ),
    SukukClause(
        clause_id="5.1.8",
        title="Murabaha + Salam sukuk: debt instruments — face-value transfer only",
        rule=SukukRule.SECONDARY_MARKET_TRADABILITY,
        summary="Pure Murabaha and Salam sukuk represent debt and may be transferred "
        "only at face value (assignment of debt, not trading).",
    ),
    SukukClause(
        clause_id="6.1",
        title="Listed halal purpose",
        rule=SukukRule.PURPOSE_LISTED_HALAL,
        summary="If the sukuk is publicly listed, the listing prospectus must declare "
        "the halal purpose and any non-halal income subject to purification.",
    ),
    SukukClause(
        clause_id="6.2",
        title="Shariah board opinion published",
        rule=SukukRule.SHARIA_BOARD_OPINION,
        summary="The issuance must include a Shariah Board opinion / fatwa published "
        "alongside the prospectus.",
    ),
    SukukClause(
        clause_id="7.1",
        title="Redemption at fair / market value, not original face",
        rule=SukukRule.REDEMPTION_AT_FAIR_VALUE,
        summary="At maturity, sukukholders are paid the fair / market value of the "
        "underlying asset (after profit / loss attribution), not a fixed face amount "
        "that would resemble principal repayment.",
    ),
)


CLAUSES: tuple[SukukClause, ...] = tuple(
    sorted(_RAW_CLAUSES, key=lambda c: _clause_sort_key(c.clause_id))
)


def clauses_for_rule(rule: SukukRule) -> tuple[SukukClause, ...]:
    return tuple(c for c in CLAUSES if c.rule is rule)


def clause_by_id(clause_id: str) -> SukukClause | None:
    for c in CLAUSES:
        if c.clause_id == clause_id:
            return c
    return None


def is_tradable_in_secondary(sukuk_type: SukukType) -> bool:
    """Cl. 5.1.8 — Murabaha + Salam are debt-only and not tradable on secondary."""
    return sukuk_type in TRADABLE_IN_SECONDARY


@dataclass(frozen=True)
class SukukIssuanceInputs:
    """Inputs for screening a candidate sukuk issuance against Standard 17."""

    issuer: str
    sukuk_type: SukukType
    underlying_purpose: str
    tangible_asset_ratio: float
    proceeds_usage_disclosed: bool
    sharia_board_opinion_published: bool
    purpose_is_halal: bool
    interest_rate_linked_payouts: bool = False
    principal_guaranteed_by_issuer: bool = False
    redemption_is_fair_value: bool = True

    def __post_init__(self) -> None:
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")
        if not self.underlying_purpose or not self.underlying_purpose.strip():
            raise ValueError("underlying_purpose must be non-empty")
        if not 0.0 <= self.tangible_asset_ratio <= 1.0:
            raise ValueError("tangible_asset_ratio must be in [0,1]")


@dataclass(frozen=True)
class SukukAssessment:
    """Result of running an issuance against the Standard-17 catalogue."""

    issuer: str
    sukuk_type: SukukType
    violated_rules: frozenset[SukukRule]
    is_compliant: bool
    secondary_tradable: bool

    def __post_init__(self) -> None:
        # Structural pin: is_compliant ⇔ violated_rules empty
        if self.is_compliant and self.violated_rules:
            raise ValueError("is_compliant=True but violated_rules non-empty")
        if (not self.is_compliant) and not self.violated_rules:
            raise ValueError("is_compliant=False but violated_rules empty")


def screen_sukuk(inputs: SukukIssuanceInputs) -> SukukAssessment:
    """Run an issuance against the Standard-17 catalogue and return an assessment."""
    violations: set[SukukRule] = set()
    if not inputs.purpose_is_halal:
        violations.add(SukukRule.UNDERLYING_ASSET_HALAL)
    if not inputs.proceeds_usage_disclosed:
        violations.add(SukukRule.PROCEEDS_USAGE_DISCLOSED)
    if inputs.principal_guaranteed_by_issuer:
        violations.add(SukukRule.NO_GUARANTEED_PRINCIPAL)
    if inputs.interest_rate_linked_payouts:
        violations.add(SukukRule.NO_INTEREST_RATE_LINK)
    if not inputs.sharia_board_opinion_published:
        violations.add(SukukRule.SHARIA_BOARD_OPINION)
    if not inputs.redemption_is_fair_value:
        violations.add(SukukRule.REDEMPTION_AT_FAIR_VALUE)

    secondary = is_tradable_in_secondary(inputs.sukuk_type) and (
        inputs.tangible_asset_ratio >= 0.51
    )

    return SukukAssessment(
        issuer=inputs.issuer,
        sukuk_type=inputs.sukuk_type,
        violated_rules=frozenset(violations),
        is_compliant=len(violations) == 0,
        secondary_tradable=secondary,
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
    for token in _FORBIDDEN_RENDER_TOKENS:
        if token in text:
            text = text.replace(token, "[redacted]")
    return text


def render_assessment(assessment: SukukAssessment) -> str:
    """Render a sukuk assessment as a short multi-line operator summary."""
    emoji = "✅" if assessment.is_compliant else "❌"
    secondary = "tradable on secondary" if assessment.secondary_tradable else "primary-only"
    lines = [
        f"{emoji} {assessment.issuer} — sukuk type: {assessment.sukuk_type.value} "
        f"({secondary})"
    ]
    if assessment.violated_rules:
        for r in sorted(assessment.violated_rules, key=lambda x: x.value):
            lines.append(f"  • violates {r.value}")
    return _scrub_render("\n".join(lines))


def render_clause(clause: SukukClause) -> str:
    return _scrub_render(
        f"§{clause.clause_id} — {clause.title} [{clause.rule.value}]"
    )


def render_coverage_matrix(rules_engaged: Iterable[SukukRule] | None = None) -> str:
    engaged = (
        frozenset(rules_engaged)
        if rules_engaged is not None
        else frozenset(c.rule for c in CLAUSES)
    )
    engaged_clauses = sum(1 for c in CLAUSES if c.rule in engaged)
    header = (
        f"AAOIFI Standard 17 (Investment Sukuk) — {engaged_clauses}/{len(CLAUSES)} "
        "clauses engaged"
    )
    lines = [header, "-" * len(header)]
    for c in CLAUSES:
        marker = "✅" if c.rule in engaged else "  "
        lines.append(f"{marker} {render_clause(c)}")
    return "\n".join(lines)
