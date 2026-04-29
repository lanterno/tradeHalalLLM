"""Sharia-compliance explainer.

Wave L — every trade is gated by a halal screener whose reasoning
ends up as a JSONB criteria blob nobody looks at. This module turns
the criteria back into operator-readable Markdown with citations to
the project's jurisprudence reference (``docs/halal_jurisprudence.md``).

The output is a stable Markdown string the dashboard can render and
the CLI can print. Tests pin the exact text for two representative
cases (clear pass, doubtful with override) so a future scholar
challenge can be answered with the same chain-of-reasoning the bot
saw at trade time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Explanation:
    """Rendered explanation for one halal-screening decision."""

    decision: str  # "halal" | "not_halal" | "doubtful"
    body_md: str
    sources: list[str]


def explain_screening(receipt: dict[str, Any]) -> Explanation:
    """Render an operator-friendly Markdown explanation.

    ``receipt`` is the dict returned by the audit module
    (``halal/audit.py:export_receipt``) — at minimum it has:

    * ``screening.decision`` — "halal" / "not_halal" / "doubtful"
    * ``screening.source`` — "zoya" / "coingecko_rules" / "override" / "cache"
    * ``screening.criteria`` — the JSONB criteria dict the screener wrote
    * ``trade.symbol`` — what we traded
    """
    screening = receipt.get("screening") or {}
    trade = receipt.get("trade") or {}

    symbol = trade.get("symbol") or trade.get("pair") or "?"
    decision = (screening.get("decision") or "doubtful").lower()
    source = screening.get("source") or "unknown"
    criteria = screening.get("criteria") or {}

    lines: list[str] = []
    sources: list[str] = []

    lines.append(f"# Halal compliance: `{symbol}` — **{decision.upper()}**")
    lines.append("")
    lines.append(f"_Source:_ `{source}`")
    if screening.get("cache_hit"):
        lines.append("_Cache hit — see audit row for the original decision row._")
    lines.append("")

    if decision == "halal":
        lines.append("## Why this passed")
        lines.append(_render_pass_criteria(criteria))
        sources.append("docs/halal_jurisprudence.md#section-1-utility-tokens")
    elif decision == "not_halal":
        lines.append("## Why this failed")
        lines.append(_render_fail_criteria(criteria))
        sources.append("docs/halal_jurisprudence.md#section-3-prohibited-activities")
    else:  # doubtful
        lines.append("## Why this is doubtful")
        lines.append(_render_doubt_criteria(criteria))
        lines.append("")
        lines.append(
            "Operator may override via the Sharia exception queue; "
            "see `/api/insights/exceptions` for pending entries."
        )
        sources.append("docs/halal_jurisprudence.md#section-4-doubtful-and-overrides")

    if criteria.get("notes"):
        lines.append("")
        lines.append(f"> {criteria['notes']}")

    return Explanation(decision=decision, body_md="\n".join(lines).strip(), sources=sources)


def _render_pass_criteria(criteria: dict[str, Any]) -> str:
    parts = []
    if cat := criteria.get("category"):
        parts.append(f"- Asset category: **{cat}**")
    if cap := criteria.get("market_cap"):
        try:
            parts.append(f"- Market cap: **${float(cap):,.0f}**")
        except TypeError, ValueError:
            pass
    if criteria.get("interest_bearing") is False:
        parts.append("- Confirmed not interest-bearing.")
    if criteria.get("haram_revenue_pct") is not None:
        ratio = float(criteria["haram_revenue_pct"])
        parts.append(f"- Haram revenue ratio: **{ratio:.1%}** (within 5% threshold).")
    if not parts:
        parts.append("- All standard halal-cache criteria met (no specific flags raised).")
    return "\n".join(parts)


def _render_fail_criteria(criteria: dict[str, Any]) -> str:
    parts = []
    failures = criteria.get("failures") or []
    if isinstance(failures, list):
        for f in failures:
            parts.append(f"- ❌ {f}")
    if criteria.get("interest_bearing") is True:
        parts.append("- ❌ Asset is interest-bearing (riba).")
    if (haram := criteria.get("haram_revenue_pct")) is not None:
        try:
            haram_f = float(haram)
            if haram_f > 0.05:
                parts.append(f"- ❌ Haram revenue ratio **{haram_f:.1%}** exceeds 5% threshold.")
        except TypeError, ValueError:
            pass
    if not parts:
        parts.append("- Failed standard halal-cache screening (no specific flags surfaced).")
    return "\n".join(parts)


def _render_doubt_criteria(criteria: dict[str, Any]) -> str:
    parts = []
    if reason := criteria.get("reason"):
        parts.append(f"- {reason}")
    for k, v in criteria.items():
        if k in ("reason", "notes"):
            continue
        parts.append(f"- {k}: {v}")
    if not parts:
        parts.append("- No specific criteria recorded.")
    return "\n".join(parts)
