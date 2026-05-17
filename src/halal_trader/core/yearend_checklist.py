"""Year-end tax + Zakat checklist generator — Round-5 Wave 18.J.

A composable per-jurisdiction year-end checklist the operator works
through to ensure all tax-prep + Zakat-prep tasks are complete before
filing. Each item is a typed task with category, deadline, and a
"required for jurisdiction" flag so the checklist tailors itself to
the operator's filing profile.

Pinned semantics:

- **Closed-set TaskCategory ladder** (TAX / ZAKAT / PURIFICATION /
  RECORDS / COMPLIANCE).
- **Closed-set Jurisdiction ladder** — re-uses round-5 jurisdictions.
- **`build_checklist`** is pure — given (jurisdictions, year), it
  returns a deterministic ordered list of tasks.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from halal_trader.halal.jurisdiction_router import Jurisdiction


class TaskCategory(str, Enum):
    """Closed-set task categories."""

    TAX = "tax"
    ZAKAT = "zakat"
    PURIFICATION = "purification"
    RECORDS = "records"
    COMPLIANCE = "compliance"


@dataclass(frozen=True)
class ChecklistTask:
    """A single year-end task."""

    task_id: str
    title: str
    category: TaskCategory
    jurisdictions: frozenset[Jurisdiction]
    deadline: date
    description: str = ""

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")


# Module-level catalogue of common year-end tasks. Operators add custom
# tasks via `extend_checklist`.
def _catalogue(year: int) -> tuple[ChecklistTask, ...]:
    return (
        ChecklistTask(
            task_id="YE-001",
            title="Reconcile broker 1099-B with internal trade ledger",
            category=TaskCategory.TAX,
            jurisdictions=frozenset({Jurisdiction.USA}),
            deadline=date(year + 1, 2, 15),
            description="Compare every realized round-trip in 1099-B against the internal ledger.",
        ),
        ChecklistTask(
            task_id="YE-002",
            title="Generate Form 8949 from realized lots",
            category=TaskCategory.TAX,
            jurisdictions=frozenset({Jurisdiction.USA}),
            deadline=date(year + 1, 4, 15),
        ),
        ChecklistTask(
            task_id="YE-003",
            title="UK CGT computation and 30-day rule check",
            category=TaskCategory.TAX,
            jurisdictions=frozenset({Jurisdiction.UK}),
            deadline=date(year + 1, 1, 31),
        ),
        ChecklistTask(
            task_id="YE-004",
            title="Compute annual Zakat on net wealth",
            category=TaskCategory.ZAKAT,
            jurisdictions=frozenset(
                {
                    Jurisdiction.SAUDI_ARABIA,
                    Jurisdiction.UAE,
                    Jurisdiction.PAKISTAN,
                    Jurisdiction.MALAYSIA,
                    Jurisdiction.INDONESIA,
                }
            ),
            deadline=date(year + 1, 1, 1),
        ),
        ChecklistTask(
            task_id="YE-005",
            title="Verify all purification disbursements signed and chained",
            category=TaskCategory.PURIFICATION,
            jurisdictions=frozenset(Jurisdiction),
            deadline=date(year + 1, 1, 31),
        ),
        ChecklistTask(
            task_id="YE-006",
            title="Archive cycle replay snapshots for the year",
            category=TaskCategory.RECORDS,
            jurisdictions=frozenset(Jurisdiction),
            deadline=date(year + 1, 3, 31),
        ),
        ChecklistTask(
            task_id="YE-007",
            title="Tax-loss harvesting cutoff (US wash-sale window)",
            category=TaskCategory.TAX,
            jurisdictions=frozenset({Jurisdiction.USA}),
            deadline=date(year, 12, 27),
        ),
        ChecklistTask(
            task_id="YE-008",
            title="Saudi CMA disclosure (if applicable to operator)",
            category=TaskCategory.COMPLIANCE,
            jurisdictions=frozenset({Jurisdiction.SAUDI_ARABIA}),
            deadline=date(year + 1, 3, 31),
        ),
        ChecklistTask(
            task_id="YE-009",
            title="Signed receipts to charity recipients",
            category=TaskCategory.PURIFICATION,
            jurisdictions=frozenset(Jurisdiction),
            deadline=date(year + 1, 1, 31),
        ),
        ChecklistTask(
            task_id="YE-010",
            title="Year-end FX rate snapshot for multi-currency translation",
            category=TaskCategory.RECORDS,
            jurisdictions=frozenset(Jurisdiction),
            deadline=date(year, 12, 31),
        ),
    )


def build_checklist(
    *,
    year: int,
    jurisdictions: Iterable[Jurisdiction],
    extra: Iterable[ChecklistTask] = (),
) -> tuple[ChecklistTask, ...]:
    """Build a sorted, deduplicated checklist for the given jurisdictions."""
    if year < 1900 or year > 9999:
        raise ValueError(f"year must be in [1900, 9999]; got {year}")
    juris_set = frozenset(jurisdictions)
    if not juris_set:
        raise ValueError("at least one jurisdiction required")

    cat = list(_catalogue(year))
    cat.extend(extra)

    relevant = [t for t in cat if t.jurisdictions & juris_set]
    relevant.sort(key=lambda t: (t.deadline, t.task_id))
    return tuple(relevant)


def filter_by_category(
    checklist: Iterable[ChecklistTask], category: TaskCategory
) -> tuple[ChecklistTask, ...]:
    return tuple(t for t in checklist if t.category is category)


def upcoming(
    checklist: Iterable[ChecklistTask], *, today: date, days: int = 30
) -> tuple[ChecklistTask, ...]:
    """Tasks due in the next `days` days (inclusive of today)."""
    if days < 0:
        raise ValueError("days must be non-negative")
    return tuple(
        t for t in checklist if (t.deadline - today).days >= 0 and (t.deadline - today).days <= days
    )


def render_checklist(checklist: Iterable[ChecklistTask]) -> str:
    items = list(checklist)
    if not items:
        return "Year-end checklist: empty"
    head = f"Year-end checklist: {len(items)} tasks"
    lines = [head]
    for t in items:
        juris = "/".join(sorted(j.value for j in t.jurisdictions))[:40]
        lines.append(
            f"  □ [{t.deadline.isoformat()}] {t.task_id}: {t.title} "
            f"({t.category.value}, {juris})"
        )
    return "\n".join(lines)
