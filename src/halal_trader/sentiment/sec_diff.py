"""SEC 8-K / 10-K diff engine — Round-5 Wave 11.E.

Detect material changes between two consecutive filings of the same
type from the same issuer. The bot then alerts when a 10-K shows new
risk-factor disclosures or an 8-K signals a material event the
sentiment layer hasn't caught.

This module ships the **structural diff primitive** — section-keyed
diff with material-change classification. The fetcher (EDGAR client)
+ section parser sit above; this module operates on the
post-extracted section text.

Pinned semantics:

- **Closed-set Section ladder** — the major sections of 10-K + 8-K.
- **Closed-set ChangeKind ladder** (ADDED / REMOVED / MODIFIED).
- **Materiality threshold** — operator-tunable; defaults to a
  sentence-overlap ratio of 0.30. A section that retains <30% of
  the previous filing's sentences is MATERIAL.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class FilingSection(str, Enum):
    """Closed-set 10-K / 8-K sections."""

    BUSINESS = "business"
    RISK_FACTORS = "risk_factors"
    LEGAL_PROCEEDINGS = "legal_proceedings"
    MD_A = "md_a"
    FINANCIAL_STATEMENTS = "financial_statements"
    CONTROLS = "controls"
    EXECUTIVE_COMP = "executive_comp"
    SUBSEQUENT_EVENTS = "subsequent_events"
    ITEM_1_01 = "item_1_01"  # 8-K material agreement
    ITEM_2_02 = "item_2_02"  # 8-K results of operations
    ITEM_5_02 = "item_5_02"  # 8-K officers / directors


class ChangeKind(str, Enum):
    """Closed-set diff change types."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class DiffPolicy:
    """Operator-tunable diff policy."""

    material_overlap_threshold: float = 0.30
    min_section_chars: int = 50

    def __post_init__(self) -> None:
        if not 0.0 < self.material_overlap_threshold < 1.0:
            raise ValueError("material_overlap_threshold must be in (0, 1)")
        if self.min_section_chars <= 0:
            raise ValueError("min_section_chars must be positive")


@dataclass(frozen=True)
class SectionDiff:
    """Diff for a single section."""

    section: FilingSection
    change_kind: ChangeKind
    overlap_ratio: float
    is_material: bool
    sentences_added: int
    sentences_removed: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap_ratio <= 1.0:
            raise ValueError("overlap_ratio must be in [0, 1]")
        if self.sentences_added < 0 or self.sentences_removed < 0:
            raise ValueError("sentence counts must be non-negative")


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$")


def _sentences(text: str) -> tuple[str, ...]:
    if not text or not text.strip():
        return ()
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return tuple(p.strip() for p in parts if p.strip())


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def diff_section(
    section: FilingSection,
    previous_text: str,
    current_text: str,
    *,
    policy: DiffPolicy | None = None,
) -> SectionDiff:
    """Diff a single section, classifying the change."""
    pol = policy if policy is not None else DiffPolicy()

    prev_chars = len(previous_text.strip())
    curr_chars = len(current_text.strip())

    if prev_chars < pol.min_section_chars and curr_chars >= pol.min_section_chars:
        new_count = len(_sentences(current_text))
        return SectionDiff(
            section=section,
            change_kind=ChangeKind.ADDED,
            overlap_ratio=0.0,
            is_material=True,
            sentences_added=new_count,
            sentences_removed=0,
        )
    if prev_chars >= pol.min_section_chars and curr_chars < pol.min_section_chars:
        old_count = len(_sentences(previous_text))
        return SectionDiff(
            section=section,
            change_kind=ChangeKind.REMOVED,
            overlap_ratio=0.0,
            is_material=True,
            sentences_added=0,
            sentences_removed=old_count,
        )
    if prev_chars < pol.min_section_chars and curr_chars < pol.min_section_chars:
        return SectionDiff(
            section=section,
            change_kind=ChangeKind.UNCHANGED,
            overlap_ratio=1.0,
            is_material=False,
            sentences_added=0,
            sentences_removed=0,
        )

    prev_sents = {_normalise(s) for s in _sentences(previous_text)}
    curr_sents = {_normalise(s) for s in _sentences(current_text)}
    common = prev_sents & curr_sents
    union = prev_sents | curr_sents
    overlap = len(common) / len(union) if union else 1.0
    added = len(curr_sents - prev_sents)
    removed = len(prev_sents - curr_sents)

    if added == 0 and removed == 0:
        return SectionDiff(
            section=section,
            change_kind=ChangeKind.UNCHANGED,
            overlap_ratio=overlap,
            is_material=False,
            sentences_added=0,
            sentences_removed=0,
        )

    is_material = overlap < pol.material_overlap_threshold
    return SectionDiff(
        section=section,
        change_kind=ChangeKind.MODIFIED,
        overlap_ratio=overlap,
        is_material=is_material,
        sentences_added=added,
        sentences_removed=removed,
    )


@dataclass(frozen=True)
class FilingDiff:
    """Diff across all sections of a filing."""

    section_diffs: tuple[SectionDiff, ...]

    def material_changes(self) -> tuple[SectionDiff, ...]:
        return tuple(d for d in self.section_diffs if d.is_material)

    def has_material_changes(self) -> bool:
        return bool(self.material_changes())


def diff_filing(
    previous_sections: Mapping[FilingSection, str],
    current_sections: Mapping[FilingSection, str],
    *,
    policy: DiffPolicy | None = None,
) -> FilingDiff:
    """Diff every section in either filing."""
    sections = sorted(set(previous_sections) | set(current_sections), key=lambda s: s.value)
    diffs: list[SectionDiff] = []
    for s in sections:
        prev = previous_sections.get(s, "")
        curr = current_sections.get(s, "")
        diffs.append(diff_section(s, prev, curr, policy=policy))
    return FilingDiff(section_diffs=tuple(diffs))


def render_diff(filing_diff: FilingDiff) -> str:
    if not filing_diff.section_diffs:
        return "Filing diff: no sections"
    head = f"Filing diff: {len(filing_diff.section_diffs)} sections, "
    head += f"{len(filing_diff.material_changes())} material"
    lines = [head]
    for d in filing_diff.section_diffs:
        marker = "‼" if d.is_material else "·"
        lines.append(
            f"  {marker} {d.section.value}: {d.change_kind.value} "
            f"(overlap={d.overlap_ratio:.2f}, +{d.sentences_added}/-{d.sentences_removed})"
        )
    return "\n".join(lines)
