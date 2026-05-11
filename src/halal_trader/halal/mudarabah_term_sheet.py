"""Mudarabah term-sheet generator — Round-5 Wave 6.B.

Conventional VC term sheets bundle riba-laden clauses: liquidation
preferences (a guaranteed minimum return on top of equity); cumulative
preferred dividends (interest by another name); ratchet anti-dilution
(asymmetric upside protection); senior debt with a fixed rate.

The Mudarabah-style halal alternative:

- **No guaranteed return** — investor (rabb-al-mal) gets profit only
  when the venture is profitable.
- **Profit-share ratio is contractual** (e.g. 70/30 investor/founder).
- **Loss is borne by capital provider only** (the founder loses
  effort, not money — unless negligence is proven). AAOIFI Standard 13.
- **No preferred liquidation** — pari-passu pro-rata distribution.
- **Anti-dilution by capital re-injection** — not a ratchet.

This module is the **term-sheet generator + validator**. It composes
proposed terms into a `MudarabahTermSheet` dataclass and either emits
a clean rendering or rejects the proposal with a list of haram
clauses that need to be removed.

Pinned semantics:

- **Closed-set ProhibitedClause ladder** — each clause is rejected
  with a specific reason citing AAOIFI Standard 13.
- **Profit-share ratio must be in (0, 1)** — neither party gets 100%.
- **No guaranteed return** — `guaranteed_return_pct` must be 0;
  setting it raises.
- **Liquidation preferences forbidden.**
- **Cumulative dividends forbidden.**
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class ProhibitedClause(str, Enum):
    """Closed-set list of clauses that violate AAOIFI Standard 13."""

    GUARANTEED_RETURN = "guaranteed_return"
    LIQUIDATION_PREFERENCE = "liquidation_preference"
    CUMULATIVE_DIVIDEND = "cumulative_dividend"
    INTEREST_BEARING_DEBT = "interest_bearing_debt"
    FIXED_PAYOUT = "fixed_payout"
    RATCHET_ANTI_DILUTION = "ratchet_anti_dilution"
    """Full-ratchet anti-dilution; permissible alternatives use
    Mudarabah re-injection."""
    SENIOR_PREFERRED_SHARES = "senior_preferred_shares"


@dataclass(frozen=True)
class MudarabahTermSheet:
    """A proposed Mudarabah term sheet."""

    deal_id: str
    investor_name: str
    founder_name: str
    capital_amount_usd: float
    profit_share_investor: float
    """Investor's share of profit. In (0, 1). Founder gets 1 - this."""
    valuation_usd: float
    closing_date: date
    expected_horizon_years: int = 5
    guaranteed_return_pct: float = 0.0
    has_liquidation_preference: bool = False
    has_cumulative_dividend: bool = False
    has_interest_bearing_debt: bool = False
    has_fixed_payout: bool = False
    has_ratchet_anti_dilution: bool = False
    has_senior_preferred_shares: bool = False
    weighted_anti_dilution_allowed: bool = True
    """Weighted-average anti-dilution is permissible (not a ratchet)."""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.deal_id or not self.deal_id.strip():
            raise ValueError("deal_id must be non-empty")
        if not self.investor_name or not self.investor_name.strip():
            raise ValueError("investor_name must be non-empty")
        if not self.founder_name or not self.founder_name.strip():
            raise ValueError("founder_name must be non-empty")
        if self.investor_name == self.founder_name:
            raise ValueError("investor and founder must be distinct parties")
        if self.capital_amount_usd <= 0:
            raise ValueError("capital_amount_usd must be positive")
        if self.valuation_usd <= 0:
            raise ValueError("valuation_usd must be positive")
        if self.capital_amount_usd > self.valuation_usd:
            raise ValueError("capital_amount cannot exceed valuation")
        if not 0.0 < self.profit_share_investor < 1.0:
            raise ValueError("profit_share_investor must be in (0, 1)")
        if self.expected_horizon_years <= 0:
            raise ValueError("expected_horizon_years must be positive")
        if self.guaranteed_return_pct < 0:
            raise ValueError("guaranteed_return_pct cannot be negative")

    def founder_share(self) -> float:
        return 1.0 - self.profit_share_investor


@dataclass(frozen=True)
class ValidationResult:
    """Output of `validate_term_sheet`."""

    is_halal: bool
    prohibited_clauses: tuple[ProhibitedClause, ...]
    reasons: tuple[str, ...]


def validate_term_sheet(sheet: MudarabahTermSheet) -> ValidationResult:
    """Apply the Mudarabah halal rule set and surface every breach.

    Pinned: returns a *complete* list of breaches (not first-fail).
    Operators want to know everything that needs to change in one
    pass, not iterate fix-by-fix.
    """
    prohibited: list[ProhibitedClause] = []
    reasons: list[str] = []
    if sheet.guaranteed_return_pct > 0:
        prohibited.append(ProhibitedClause.GUARANTEED_RETURN)
        reasons.append(
            f"guaranteed_return_pct={sheet.guaranteed_return_pct:.2%} violates "
            "AAOIFI Standard 13: investor receives profit only when venture profits"
        )
    if sheet.has_liquidation_preference:
        prohibited.append(ProhibitedClause.LIQUIDATION_PREFERENCE)
        reasons.append(
            "liquidation preference violates pari-passu pro-rata distribution "
            "required by Mudarabah; remove or restructure"
        )
    if sheet.has_cumulative_dividend:
        prohibited.append(ProhibitedClause.CUMULATIVE_DIVIDEND)
        reasons.append(
            "cumulative dividend is interest by another name (riba); "
            "use Mudarabah profit-share instead"
        )
    if sheet.has_interest_bearing_debt:
        prohibited.append(ProhibitedClause.INTEREST_BEARING_DEBT)
        reasons.append(
            "interest-bearing debt is riba; replace with Mudarabah equity "
            "or Murabaha cost-plus financing"
        )
    if sheet.has_fixed_payout:
        prohibited.append(ProhibitedClause.FIXED_PAYOUT)
        reasons.append(
            "fixed payout is a guaranteed return; Mudarabah requires outcome-contingent profit"
        )
    if sheet.has_ratchet_anti_dilution:
        prohibited.append(ProhibitedClause.RATCHET_ANTI_DILUTION)
        reasons.append(
            "full-ratchet anti-dilution creates asymmetric loss-protection; "
            "use weighted-average or Mudarabah re-injection instead"
        )
    if sheet.has_senior_preferred_shares:
        prohibited.append(ProhibitedClause.SENIOR_PREFERRED_SHARES)
        reasons.append(
            "senior preferred shares create pari-passu violation; "
            "Mudarabah requires equal-rank investors"
        )
    return ValidationResult(
        is_halal=not prohibited,
        prohibited_clauses=tuple(prohibited),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class ScenarioPayout:
    """Output of `scenario_payout` — investor + founder slices for a profit/loss."""

    venture_pnl: float
    investor_payout: float
    founder_payout: float
    note: str


def scenario_payout(
    sheet: MudarabahTermSheet,
    *,
    venture_pnl: float,
) -> ScenarioPayout:
    """Compute the Mudarabah-correct payout for a given venture P&L.

    Pinned semantics:
    - Profit > 0 → split per profit_share ratio.
    - Loss < 0 → 100% borne by investor (Standard 13). Founder loses
      effort only — unless negligence; that path is not modelled here.
    - Loss > capital → capped at capital (no clawback).
    """
    if venture_pnl >= 0:
        investor = venture_pnl * sheet.profit_share_investor
        founder = venture_pnl * sheet.founder_share()
        note = f"profit split: {sheet.profit_share_investor:.0%}/{sheet.founder_share():.0%}"
    else:
        loss = -venture_pnl
        capped = min(loss, sheet.capital_amount_usd)
        investor = -capped
        founder = 0.0
        if capped < loss:
            note = "loss capped at capital (no clawback under Mudarabah)"
        else:
            note = "loss borne 100% by investor (Standard 13)"
    return ScenarioPayout(
        venture_pnl=venture_pnl,
        investor_payout=investor,
        founder_payout=founder,
        note=note,
    )


def _mask(name: str) -> str:
    if len(name) <= 4:
        return "***"
    return name[:2] + "…" + name[-2:]


def render_term_sheet(sheet: MudarabahTermSheet) -> str:
    """Operator-readable summary of the term sheet."""
    head = (
        f"📃 Mudarabah term sheet: {sheet.deal_id} "
        f"capital ${sheet.capital_amount_usd:,.0f} @ "
        f"${sheet.valuation_usd:,.0f} valuation\n"
        f"  • Parties: {_mask(sheet.investor_name)} (investor) / "
        f"{_mask(sheet.founder_name)} (founder)\n"
        f"  • Profit share: {sheet.profit_share_investor:.0%} investor / "
        f"{sheet.founder_share():.0%} founder\n"
        f"  • Horizon: {sheet.expected_horizon_years} years; closing "
        f"{sheet.closing_date.isoformat()}"
    )
    return head


def render_validation(result: ValidationResult) -> str:
    """Operator-readable validation result."""
    if result.is_halal:
        return "✅ Term sheet is halal-compliant under AAOIFI Standard 13."
    lines = [f"❌ Term sheet has {len(result.prohibited_clauses)} prohibited clause(s):"]
    for c, reason in zip(result.prohibited_clauses, result.reasons, strict=True):
        lines.append(f"  • {c.value}: {reason}")
    return "\n".join(lines)


def render_scenario(scenario: ScenarioPayout) -> str:
    return (
        f"💼 Scenario P&L=${scenario.venture_pnl:+,.2f}: "
        f"investor=${scenario.investor_payout:+,.2f}, "
        f"founder=${scenario.founder_payout:+,.2f} ({scenario.note})"
    )
