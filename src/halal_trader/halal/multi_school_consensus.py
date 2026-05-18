"""Multi-school Shariah consensus engine.

Round-5 Wave 1.D primitive. The standard halal screener returns a
single PERMISSIBLE/IMPERMISSIBLE verdict using one school's
methodology (typically Hanafi defaults). Many trades are halal in
one school but disputed in another — a Saudi user (often
Hanbali/Maliki) and a Pakistani user (often Hanafi) might reach
different verdicts on the same name. This module aggregates the
four Sunni schools' positions (Hanafi / Shafi'i / Maliki /
Hanbali) plus optional Ja'fari for Shia inclusivity, surfacing
consensus level (unanimous / majority / split) so the operator
can apply their preferred strictness.

Picked an explicit consensus aggregator over hard-coding one
school's verdict because (a) the platform's user base is
multi-school by definition (a global halal trading platform must
serve users following different fiqh schools); (b) operators in
some jurisdictions (Saudi/UAE) prefer unanimous Sunni consensus
as the strictness gate; in others (Pakistan/India) majority is
acceptable; encoding the strictness as an operator-tunable mode
keeps the engine policy-free; (c) the consensus output feeds the
dashboard "school disagreement" tile which surfaces edge cases
for the scholar review queue (Wave 2.F).

Pinned semantics:
- **Closed-set School catalogue.** HANAFI / SHAFII / MALIKI /
  HANBALI / JAFARI. Adding a school is a code review change —
  the catalogue is documentation that scholars + operators read.
- **Closed-set SchoolVerdict ladder.** PERMISSIBLE / IMPERMISSIBLE
  / ABSTAIN. ABSTAIN means the school hasn't opined (e.g., a
  brand-new fintech instrument the school hasn't reviewed).
- **No duplicate schools.** A `ConsensusReport` cannot have two
  positions from the same school — pinned via test.
- **ConsensusMode determines tradability.** UNANIMOUS = every
  non-abstain school must agree PERMISSIBLE; MAJORITY = strict
  permissible-count > impermissible-count; ANY = at least one
  PERMISSIBLE position. Operator picks based on jurisdictional
  norms.
- **Render output never includes scholar contact emails or
  internal verdict transcripts.** Only the school + verdict +
  brief reasoning; the full scholar correspondence lives in
  `halal/scholar_review.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class School(str, Enum):
    """Fiqh school catalogue.

    Pinned string values for JSON / DB persistence stability.
    Four Sunni majority schools + Ja'fari (Shia) for inclusivity.
    """

    HANAFI = "hanafi"
    SHAFII = "shafii"
    MALIKI = "maliki"
    HANBALI = "hanbali"
    JAFARI = "jafari"


# Module-level canonical Sunni-only set. Used for the `is_sunni_consensus`
# predicate. Frozen at module level — adding a Sunni school is a code
# review change.
SUNNI_SCHOOLS: frozenset[School] = frozenset(
    {School.HANAFI, School.SHAFII, School.MALIKI, School.HANBALI}
)


class SchoolVerdict(str, Enum):
    """Per-school verdict on a screening question.

    Pinned string values. PERMISSIBLE = halal under this school;
    IMPERMISSIBLE = haram under this school; ABSTAIN = the school
    hasn't formally opined (often the case for novel fintech).
    """

    PERMISSIBLE = "permissible"
    IMPERMISSIBLE = "impermissible"
    ABSTAIN = "abstain"


class ConsensusMode(str, Enum):
    """Operator-tunable strictness for the consensus gate.

    Pinned string values. UNANIMOUS = strict (any disagreement
    blocks); MAJORITY = permissible-count must exceed
    impermissible-count strictly; ANY = at least one permissible
    position is enough.
    """

    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    ANY = "any"


@dataclass(frozen=True)
class SchoolPosition:
    """One school's position on the question.

    `reasoning` is a brief operator-facing explanation (e.g.,
    "permitted under Salam construct per Standard 7"); the full
    scholar transcript goes in `halal/scholar_review.py`.
    `scholar_handle` optionally cites the specific scholar
    (e.g., "mufti_taqi_usmani") whose verdict this represents.
    """

    school: School
    verdict: SchoolVerdict
    reasoning: str
    scholar_handle: str | None = None

    def __post_init__(self) -> None:
        if not self.reasoning or not self.reasoning.strip():
            raise ValueError("reasoning must be non-empty")
        if self.scholar_handle is not None and not self.scholar_handle.strip():
            raise ValueError("scholar_handle if provided must be non-empty")


@dataclass(frozen=True)
class ConsensusReport:
    """Aggregated multi-school consensus."""

    positions: tuple[SchoolPosition, ...]
    permissible_count: int
    impermissible_count: int
    abstain_count: int

    def __post_init__(self) -> None:
        if self.permissible_count < 0:
            raise ValueError("permissible_count must be non-negative")
        if self.impermissible_count < 0:
            raise ValueError("impermissible_count must be non-negative")
        if self.abstain_count < 0:
            raise ValueError("abstain_count must be non-negative")
        total = self.permissible_count + self.impermissible_count + self.abstain_count
        if total != len(self.positions):
            raise ValueError(f"counts {total} != positions length {len(self.positions)}")
        # No duplicate schools
        seen: set[School] = set()
        for p in self.positions:
            if p.school in seen:
                raise ValueError(f"duplicate position for school {p.school.value}")
            seen.add(p.school)

    @property
    def total_engaged(self) -> int:
        """Schools that took a position (excludes ABSTAIN)."""

        return self.permissible_count + self.impermissible_count

    @property
    def is_unanimous_permissible(self) -> bool:
        """Every engaged school says PERMISSIBLE."""

        return self.total_engaged > 0 and self.impermissible_count == 0

    @property
    def is_unanimous_impermissible(self) -> bool:
        """Every engaged school says IMPERMISSIBLE."""

        return self.total_engaged > 0 and self.permissible_count == 0

    @property
    def is_majority_permissible(self) -> bool:
        """Strict majority (permissible > impermissible)."""

        return self.permissible_count > self.impermissible_count

    @property
    def is_split(self) -> bool:
        """At least one PERMISSIBLE and one IMPERMISSIBLE."""

        return self.permissible_count > 0 and self.impermissible_count > 0

    @property
    def is_sunni_consensus(self) -> bool:
        """All four Sunni schools have opined PERMISSIBLE."""

        sunni_engaged = {p.school for p in self.positions if p.school in SUNNI_SCHOOLS}
        if sunni_engaged != SUNNI_SCHOOLS:
            return False
        return all(
            p.verdict is SchoolVerdict.PERMISSIBLE
            for p in self.positions
            if p.school in SUNNI_SCHOOLS
        )


def build_report(positions: Iterable[SchoolPosition]) -> ConsensusReport:
    """Aggregate a list of per-school positions into a report.

    Positions are sorted by school name for deterministic ordering
    (school enum order: HANAFI, SHAFII, MALIKI, HANBALI, JAFARI).
    """

    # Sort by school enum value for stable, deterministic output
    sorted_positions = sorted(positions, key=lambda p: list(School).index(p.school))

    permissible = sum(1 for p in sorted_positions if p.verdict is SchoolVerdict.PERMISSIBLE)
    impermissible = sum(1 for p in sorted_positions if p.verdict is SchoolVerdict.IMPERMISSIBLE)
    abstain = sum(1 for p in sorted_positions if p.verdict is SchoolVerdict.ABSTAIN)

    return ConsensusReport(
        positions=tuple(sorted_positions),
        permissible_count=permissible,
        impermissible_count=impermissible,
        abstain_count=abstain,
    )


def tradable_under_consensus(
    report: ConsensusReport,
    *,
    mode: ConsensusMode = ConsensusMode.MAJORITY,
) -> bool:
    """Whether the report passes the strictness gate.

    UNANIMOUS: every engaged (non-abstain) school says PERMISSIBLE
    AND at least one school is engaged.
    MAJORITY: strict permissible > impermissible.
    ANY: at least one PERMISSIBLE position.
    """

    if mode is ConsensusMode.UNANIMOUS:
        return report.is_unanimous_permissible
    if mode is ConsensusMode.MAJORITY:
        return report.is_majority_permissible
    # ANY
    return report.permissible_count > 0


def disagreement_summary(report: ConsensusReport) -> tuple[School, ...]:
    """Schools with verdicts diverging from the majority position.

    If majority says PERMISSIBLE, returns schools voting
    IMPERMISSIBLE (and vice versa). If split (no majority),
    returns the minority side. Operators surface these in the
    "schools to consult further" tile.
    """

    # Determine majority verdict
    if report.permissible_count > report.impermissible_count:
        minority_verdict = SchoolVerdict.IMPERMISSIBLE
    elif report.impermissible_count > report.permissible_count:
        minority_verdict = SchoolVerdict.PERMISSIBLE
    else:
        # Tied — pick IMPERMISSIBLE side as "minority" by convention
        # (the conservative read: any impermissible position is the
        # operator-facing concern)
        minority_verdict = SchoolVerdict.IMPERMISSIBLE

    return tuple(p.school for p in report.positions if p.verdict is minority_verdict)


_SCHOOL_LABEL: dict[School, str] = {
    School.HANAFI: "Hanafi",
    School.SHAFII: "Shafi'i",
    School.MALIKI: "Maliki",
    School.HANBALI: "Hanbali",
    School.JAFARI: "Ja'fari",
}


_VERDICT_EMOJI: dict[SchoolVerdict, str] = {
    SchoolVerdict.PERMISSIBLE: "✅",
    SchoolVerdict.IMPERMISSIBLE: "❌",
    SchoolVerdict.ABSTAIN: "❔",
}


def render_position(position: SchoolPosition) -> str:
    """Format one school position for ops display.

    No-secret-leak: shows only the school label + verdict emoji
    + reasoning + optional scholar handle. The full transcript
    lives in the scholar review module.
    """

    emoji = _VERDICT_EMOJI[position.verdict]
    label = _SCHOOL_LABEL[position.school]
    parts = [f"{emoji} {label}: {position.verdict.value}", f"— {position.reasoning}"]
    if position.scholar_handle is not None:
        parts.append(f"[via {position.scholar_handle}]")
    return " ".join(parts)


def render_report(
    report: ConsensusReport,
    *,
    mode: ConsensusMode = ConsensusMode.MAJORITY,
) -> str:
    """Format the consensus report for ops display.

    Top-line summary + per-school details. The strictness mode is
    consulted for the verdict line.
    """

    if tradable_under_consensus(report, mode=mode):
        verdict = "✅ TRADABLE"
    else:
        verdict = "❌ NOT TRADABLE"

    lines = [
        f"{verdict} (mode: {mode.value})",
        f"  permissible: {report.permissible_count} | "
        f"impermissible: {report.impermissible_count} | "
        f"abstain: {report.abstain_count}",
    ]
    if report.is_split:
        lines.append("  ⚠️ schools disagree — operator review recommended")
    elif report.is_unanimous_permissible:
        lines.append("  unanimous PERMISSIBLE among engaged schools")
    elif report.is_unanimous_impermissible:
        lines.append("  unanimous IMPERMISSIBLE among engaged schools")
    if report.positions:
        lines.append("")
        for p in report.positions:
            lines.append(f"  {render_position(p)}")
    return "\n".join(lines)


__all__ = [
    "SUNNI_SCHOOLS",
    "ConsensusMode",
    "ConsensusReport",
    "School",
    "SchoolPosition",
    "SchoolVerdict",
    "build_report",
    "disagreement_summary",
    "render_position",
    "render_report",
    "tradable_under_consensus",
]
