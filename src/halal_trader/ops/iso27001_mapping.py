"""ISO/IEC 27001 control mapping — Round-5 Wave 19.F.

ISO 27001:2022 Annex A defines 93 controls organised across 4 themes
(Organisational, People, Physical, Technological). This module is the
**control-to-platform-evidence mapping + gap reporter**:

1. Operator maps platform-internal control implementations to ISO
   Annex A control IDs.
2. Evidence artefacts are linked to the mapping.
3. Gaps are surfaced: which Annex-A controls have no mapping; which
   have a mapping but no evidence; which have evidence but are stale.

This module is pure-Python; the deployment owns persistence + actual
artefact storage. It composes with `ops/soc2_evidence.py` for the
artefact dataclass.

Pinned semantics:

- **Closed-set Annex A theme ladder** — ORGANISATIONAL / PEOPLE /
  PHYSICAL / TECHNOLOGICAL.
- **Closed-set MappingStatus ladder** — NOT_APPLICABLE / NOT_MAPPED /
  MAPPED / IMPLEMENTED / EVIDENCED. Each step requires the prior.
- **Stale threshold default 365 days** — evidence older than this is
  flagged STALE.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — mapping rationale is truncated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum


class AnnexATheme(str, Enum):
    """Closed-set ISO 27001:2022 Annex A theme ladder."""

    ORGANISATIONAL = "organisational"  # A.5 (37 controls)
    PEOPLE = "people"  # A.6 (8 controls)
    PHYSICAL = "physical"  # A.7 (14 controls)
    TECHNOLOGICAL = "technological"  # A.8 (34 controls)


class MappingStatus(str, Enum):
    """Closed-set mapping status ladder."""

    NOT_APPLICABLE = "not_applicable"
    """Operator has documented why this control doesn't apply (e.g.
    no physical office → physical controls N/A)."""
    NOT_MAPPED = "not_mapped"
    """No platform control identified yet."""
    MAPPED = "mapped"
    """Platform control identified but not implemented."""
    IMPLEMENTED = "implemented"
    """Implemented; no evidence collected yet."""
    EVIDENCED = "evidenced"
    """Implemented + at least one fresh evidence artefact."""


_STATUS_ORDER: dict[MappingStatus, int] = {
    MappingStatus.NOT_APPLICABLE: -1,
    MappingStatus.NOT_MAPPED: 0,
    MappingStatus.MAPPED: 1,
    MappingStatus.IMPLEMENTED: 2,
    MappingStatus.EVIDENCED: 3,
}


@dataclass(frozen=True)
class AnnexAControl:
    """One ISO 27001 Annex A control entry."""

    control_id: str
    """Format A.X.Y (e.g. 'A.5.1')."""
    theme: AnnexATheme
    title: str

    def __post_init__(self) -> None:
        if not self.control_id or not self.control_id.strip():
            raise ValueError("control_id must be non-empty")
        if not self.control_id.startswith("A."):
            raise ValueError("control_id must start with 'A.'")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")


@dataclass(frozen=True)
class ControlMapping:
    """A mapping of a platform control to an Annex A control."""

    mapping_id: str
    annex_control_id: str
    platform_control_ref: str
    """Reference to the platform's internal control catalogue."""
    rationale: str
    """Short text describing why this platform control covers the
    Annex A control."""
    status: MappingStatus = MappingStatus.MAPPED
    last_evidenced_on: date | None = None
    """When the latest fresh evidence was collected. None when the
    mapping isn't EVIDENCED."""

    def __post_init__(self) -> None:
        if not self.mapping_id or not self.mapping_id.strip():
            raise ValueError("mapping_id must be non-empty")
        if not self.annex_control_id or not self.annex_control_id.strip():
            raise ValueError("annex_control_id must be non-empty")
        if not self.platform_control_ref.strip():
            raise ValueError("platform_control_ref must be non-empty")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty")
        if len(self.rationale) > 1000:
            raise ValueError("rationale must be ≤ 1000 chars")
        # NOT_APPLICABLE doesn't require a platform ref or evidence.
        if self.status is MappingStatus.EVIDENCED and self.last_evidenced_on is None:
            raise ValueError("EVIDENCED status requires last_evidenced_on")
        if self.status is not MappingStatus.EVIDENCED and self.last_evidenced_on is not None:
            raise ValueError("last_evidenced_on can only be set on EVIDENCED mappings")


@dataclass(frozen=True)
class MappingGap:
    """One gap surfaced by `find_gaps`."""

    annex_control_id: str
    theme: AnnexATheme
    gap_kind: str
    """One of: 'unmapped', 'not_implemented', 'no_evidence', 'stale'."""


_VALID_GAP_KINDS = {"unmapped", "not_implemented", "no_evidence", "stale"}


def find_gaps(
    catalog: Sequence[AnnexAControl],
    mappings: Sequence[ControlMapping],
    *,
    as_of: date,
    stale_days: int = 365,
) -> tuple[MappingGap, ...]:
    """Compare catalog vs mappings; emit one gap per shortfall.

    Pinned:
    - Each catalog entry yields ≤ 1 gap. Higher-severity wins (unmapped
      > not_implemented > no_evidence > stale).
    - NOT_APPLICABLE mappings are silently OK (operator decision).
    """
    if stale_days <= 0:
        raise ValueError("stale_days must be positive")
    by_annex: dict[str, list[ControlMapping]] = {}
    for m in mappings:
        by_annex.setdefault(m.annex_control_id, []).append(m)
    out: list[MappingGap] = []
    for c in catalog:
        ms = by_annex.get(c.control_id, [])
        if not ms:
            out.append(
                MappingGap(
                    annex_control_id=c.control_id,
                    theme=c.theme,
                    gap_kind="unmapped",
                )
            )
            continue
        # Pick the highest-status mapping per control.
        best = max(ms, key=lambda m: _STATUS_ORDER[m.status])
        if best.status is MappingStatus.NOT_APPLICABLE:
            continue
        if best.status in (MappingStatus.NOT_MAPPED, MappingStatus.MAPPED):
            out.append(
                MappingGap(
                    annex_control_id=c.control_id,
                    theme=c.theme,
                    gap_kind="not_implemented",
                )
            )
            continue
        if best.status is MappingStatus.IMPLEMENTED:
            out.append(
                MappingGap(
                    annex_control_id=c.control_id,
                    theme=c.theme,
                    gap_kind="no_evidence",
                )
            )
            continue
        # EVIDENCED — check staleness.
        if best.last_evidenced_on is None:
            # Defensive — should not reach here per dataclass invariant.
            out.append(
                MappingGap(
                    annex_control_id=c.control_id,
                    theme=c.theme,
                    gap_kind="no_evidence",
                )
            )
            continue
        age = (as_of - best.last_evidenced_on).days
        if age > stale_days:
            out.append(
                MappingGap(
                    annex_control_id=c.control_id,
                    theme=c.theme,
                    gap_kind="stale",
                )
            )
    return tuple(out)


@dataclass(frozen=True)
class CatalogSummary:
    """Aggregate coverage summary."""

    n_controls: int
    n_not_applicable: int
    n_mapped: int
    n_implemented: int
    n_evidenced: int
    n_unmapped: int
    n_stale: int


def summarise_catalog(
    catalog: Sequence[AnnexAControl],
    mappings: Sequence[ControlMapping],
    *,
    as_of: date,
    stale_days: int = 365,
) -> CatalogSummary:
    """One-shot summary."""
    by_annex: dict[str, list[ControlMapping]] = {}
    for m in mappings:
        by_annex.setdefault(m.annex_control_id, []).append(m)
    n_na = n_unmapped = n_mapped = n_impl = n_evid = n_stale = 0
    for c in catalog:
        ms = by_annex.get(c.control_id, [])
        if not ms:
            n_unmapped += 1
            continue
        best = max(ms, key=lambda m: _STATUS_ORDER[m.status])
        if best.status is MappingStatus.NOT_APPLICABLE:
            n_na += 1
        elif best.status is MappingStatus.NOT_MAPPED:
            n_unmapped += 1
        elif best.status is MappingStatus.MAPPED:
            n_mapped += 1
        elif best.status is MappingStatus.IMPLEMENTED:
            n_impl += 1
        else:  # EVIDENCED
            if (
                best.last_evidenced_on is not None
                and (as_of - best.last_evidenced_on).days > stale_days
            ):
                n_stale += 1
            else:
                n_evid += 1
    return CatalogSummary(
        n_controls=len(catalog),
        n_not_applicable=n_na,
        n_mapped=n_mapped,
        n_implemented=n_impl,
        n_evidenced=n_evid,
        n_unmapped=n_unmapped,
        n_stale=n_stale,
    )


def evidence_completeness(summary: CatalogSummary) -> float:
    """Ratio of evidenced controls to (controls − N/A). 0 when N/A == all."""
    denom = summary.n_controls - summary.n_not_applicable
    if denom <= 0:
        return 0.0
    return summary.n_evidenced / denom


_GAP_EMOJI: dict[str, str] = {
    "unmapped": "🔴",
    "not_implemented": "🟠",
    "no_evidence": "🟡",
    "stale": "🟤",
}


def render_gap(gap: MappingGap) -> str:
    emoji = _GAP_EMOJI.get(gap.gap_kind, "•")
    return f"{emoji} {gap.annex_control_id} [{gap.theme.value}]: {gap.gap_kind}"


def render_summary(summary: CatalogSummary) -> str:
    return (
        f"🛡️ ISO 27001 ({summary.n_controls} controls): "
        f"evidenced={summary.n_evidenced}, "
        f"implemented={summary.n_implemented}, "
        f"mapped={summary.n_mapped}, "
        f"N/A={summary.n_not_applicable}, "
        f"unmapped={summary.n_unmapped}, "
        f"stale={summary.n_stale}"
    )
