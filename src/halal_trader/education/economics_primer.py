"""Halal economics primer — Round-5 Wave 20.E.

A contextual glossary the dashboard surfaces from any contextual entry
point: tap a term in a chart → primer explains the concept. The
**concept graph** captures cross-references so the user can navigate
related ideas (Mudarabah → Musharakah → profit-share rules → ...).

Pinned semantics:

- **Closed-set ConceptCategory ladder** — STRUCTURE / PROHIBITION /
  CONTRACT / INSTRUMENT / RATIO / OPERATIONAL.
- **Closed-set DifficultyLevel ladder** — BEGINNER / INTERMEDIATE /
  ADVANCED. Primer can be filtered by user's qualified tier (Wave
  20.A).
- **Each concept has a deterministic `concept_id`** + canonical
  title + summary (≤ 500 chars) + body (≤ 5000 chars) + tuple of
  related concept IDs.
- **Graph integrity**: related_ids must resolve to existing concepts
  in the catalogue, else load fails.
- **Pure-Python deterministic.**
- **No-secret-leak pin** — concept content is operator-provided
  static text; no user data referenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConceptCategory(str, Enum):
    """Closed-set category ladder."""

    STRUCTURE = "structure"
    PROHIBITION = "prohibition"
    CONTRACT = "contract"
    INSTRUMENT = "instrument"
    RATIO = "ratio"
    OPERATIONAL = "operational"


class DifficultyLevel(str, Enum):
    """Closed-set difficulty ladder."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


_DIFFICULTY_ORDER: dict[DifficultyLevel, int] = {
    DifficultyLevel.BEGINNER: 0,
    DifficultyLevel.INTERMEDIATE: 1,
    DifficultyLevel.ADVANCED: 2,
}


@dataclass(frozen=True)
class Concept:
    """One primer entry."""

    concept_id: str
    title: str
    category: ConceptCategory
    difficulty: DifficultyLevel
    summary: str
    body: str
    related_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    """Alternative search keys — e.g. ('musharakah', 'partnership-fund')."""

    def __post_init__(self) -> None:
        if not self.concept_id or not self.concept_id.strip():
            raise ValueError("concept_id must be non-empty")
        if not self.title or not self.title.strip():
            raise ValueError("title must be non-empty")
        if len(self.title) > 200:
            raise ValueError("title must be ≤ 200 chars")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if len(self.summary) > 500:
            raise ValueError("summary must be ≤ 500 chars")
        if not self.body or not self.body.strip():
            raise ValueError("body must be non-empty")
        if len(self.body) > 5000:
            raise ValueError("body must be ≤ 5000 chars")
        # Aliases must be unique + non-empty.
        seen: set[str] = set()
        for a in self.aliases:
            if not a or not a.strip():
                raise ValueError("alias must be non-empty")
            if a in seen:
                raise ValueError(f"duplicate alias {a}")
            seen.add(a)
        # related_ids unique (graph nodes can't self-loop).
        if len(set(self.related_ids)) != len(self.related_ids):
            raise ValueError("related_ids must be unique")
        if self.concept_id in self.related_ids:
            raise ValueError("concept cannot relate to itself")


@dataclass(frozen=True)
class PrimerCatalog:
    """A frozen catalogue of concepts with graph integrity."""

    concepts: tuple[Concept, ...]

    def __post_init__(self) -> None:
        if not self.concepts:
            raise ValueError("catalogue must be non-empty")
        ids = [c.concept_id for c in self.concepts]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate concept_id")
        id_set = set(ids)
        # Every related_id must resolve to a real concept.
        for c in self.concepts:
            for r in c.related_ids:
                if r not in id_set:
                    raise ValueError(f"concept {c.concept_id} references unknown {r}")
        # Aliases must be globally unique (no two concepts share an alias).
        all_aliases: set[str] = set()
        for c in self.concepts:
            for a in c.aliases:
                if a in all_aliases or a in id_set:
                    raise ValueError(f"alias {a!r} duplicates an alias or concept_id")
                all_aliases.add(a)

    def by_id(self, concept_id: str) -> Concept | None:
        for c in self.concepts:
            if c.concept_id == concept_id:
                return c
        return None

    def by_alias(self, alias: str) -> Concept | None:
        for c in self.concepts:
            if alias in c.aliases:
                return c
        return None

    def lookup(self, key: str) -> Concept | None:
        """Resolve by concept_id OR alias."""
        hit = self.by_id(key)
        if hit is not None:
            return hit
        return self.by_alias(key)

    def by_category(self, category: ConceptCategory) -> tuple[Concept, ...]:
        return tuple(c for c in self.concepts if c.category is category)

    def for_difficulty(self, max_difficulty: DifficultyLevel) -> tuple[Concept, ...]:
        max_order = _DIFFICULTY_ORDER[max_difficulty]
        return tuple(c for c in self.concepts if _DIFFICULTY_ORDER[c.difficulty] <= max_order)


def related_walk(
    catalog: PrimerCatalog, *, start_id: str, max_depth: int = 2
) -> tuple[Concept, ...]:
    """Return concepts within `max_depth` graph hops from `start_id`.

    Pinned: BFS; deterministic ordering by (depth, concept_id);
    `start_id` is at depth 0 and included.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be ≥ 0")
    start = catalog.by_id(start_id)
    if start is None:
        raise ValueError(f"unknown concept_id {start_id}")
    visited: dict[str, int] = {start_id: 0}
    queue: list[tuple[str, int]] = [(start_id, 0)]
    out: list[Concept] = [start]
    head = 0
    while head < len(queue):
        cid, depth = queue[head]
        head += 1
        if depth >= max_depth:
            continue
        c = catalog.by_id(cid)
        assert c is not None
        for r in c.related_ids:
            if r in visited:
                continue
            visited[r] = depth + 1
            queue.append((r, depth + 1))
            child = catalog.by_id(r)
            if child is not None:
                out.append(child)
    # Sort by (depth, concept_id) for deterministic ordering.
    out.sort(key=lambda c: (visited[c.concept_id], c.concept_id))
    return tuple(out)


def search(
    catalog: PrimerCatalog,
    query: str,
    *,
    max_results: int = 20,
) -> tuple[Concept, ...]:
    """Case-insensitive substring search over title / aliases / summary.

    Pinned ranking:
    1. Exact alias / id match → top.
    2. Title substring match → next.
    3. Summary substring match → next.
    Tie-breaks by concept_id.
    """
    if not query or not query.strip():
        return ()
    if max_results <= 0:
        raise ValueError("max_results must be positive")
    q = query.strip().lower()
    exact: list[Concept] = []
    title_hits: list[Concept] = []
    summary_hits: list[Concept] = []
    for c in catalog.concepts:
        if c.concept_id.lower() == q or q in (a.lower() for a in c.aliases):
            exact.append(c)
        elif q in c.title.lower():
            title_hits.append(c)
        elif q in c.summary.lower():
            summary_hits.append(c)
    exact.sort(key=lambda c: c.concept_id)
    title_hits.sort(key=lambda c: c.concept_id)
    summary_hits.sort(key=lambda c: c.concept_id)
    return tuple((exact + title_hits + summary_hits)[:max_results])


def default_catalog() -> PrimerCatalog:
    """A small seed catalogue covering the core halal-finance concepts.

    Operators replace with their own production catalogue; this is
    enough to exercise the primer + serve as a documentation pin.
    """
    return PrimerCatalog(
        concepts=(
            Concept(
                concept_id="riba",
                title="Riba (Interest)",
                category=ConceptCategory.PROHIBITION,
                difficulty=DifficultyLevel.BEGINNER,
                summary=(
                    "Riba is any unjustified increment on a loan or "
                    "debt — the foundational prohibition in Islamic finance."
                ),
                body=(
                    "Riba covers both riba al-fadl (excess in a like-for-like "
                    "exchange) and riba al-nasi'ah (interest on deferred "
                    "payment). Conventional bonds, savings interest, and "
                    "margin loans all involve riba."
                ),
                related_ids=("gharar", "mudarabah", "sukuk"),
                aliases=("interest",),
            ),
            Concept(
                concept_id="gharar",
                title="Gharar (Excessive Uncertainty)",
                category=ConceptCategory.PROHIBITION,
                difficulty=DifficultyLevel.BEGINNER,
                summary=(
                    "Gharar is excessive uncertainty in a contract that "
                    "creates an unfair information asymmetry."
                ),
                body=(
                    "Classical fiqh distinguishes minor gharar (acceptable, "
                    "e.g. fruit on a tree before harvest) from major "
                    "gharar (forbidden, e.g. selling a fish in the sea). "
                    "Options + futures often fall in the major-gharar "
                    "bucket without specific structuring."
                ),
                related_ids=("riba", "maysir"),
                aliases=("uncertainty",),
            ),
            Concept(
                concept_id="maysir",
                title="Maysir (Gambling)",
                category=ConceptCategory.PROHIBITION,
                difficulty=DifficultyLevel.BEGINNER,
                summary=(
                    "Maysir is gambling or any zero-sum game where one "
                    "party's gain comes entirely from another's loss."
                ),
                body=(
                    "Casino games, sports betting, and binary options are "
                    "maysir. Speculation on tradable assets is not, as "
                    "long as the underlying represents real economic value."
                ),
                related_ids=("gharar",),
            ),
            Concept(
                concept_id="mudarabah",
                title="Mudarabah (Profit-Sharing)",
                category=ConceptCategory.CONTRACT,
                difficulty=DifficultyLevel.INTERMEDIATE,
                summary=(
                    "Mudarabah is a profit-sharing partnership where one "
                    "party provides capital (rabb-al-mal) and another "
                    "provides labour / expertise (mudarib)."
                ),
                body=(
                    "Profit is shared per a pre-agreed ratio; loss is borne "
                    "by the capital provider only (unless negligence). "
                    "Mudarabah is the structural basis for halal investment "
                    "funds + DeFi vaults."
                ),
                related_ids=("musharakah", "riba", "sukuk"),
            ),
            Concept(
                concept_id="musharakah",
                title="Musharakah (Joint Venture)",
                category=ConceptCategory.CONTRACT,
                difficulty=DifficultyLevel.INTERMEDIATE,
                summary=(
                    "Musharakah is a partnership where all parties "
                    "contribute capital and share profit + loss in "
                    "proportion to capital + agreed split."
                ),
                body=(
                    "Distinguished from Mudarabah by the loss rule: in "
                    "Musharakah, all capital contributors absorb loss; "
                    "in Mudarabah, only the rabb-al-mal does."
                ),
                related_ids=("mudarabah",),
            ),
            Concept(
                concept_id="sukuk",
                title="Sukuk (Islamic Bonds)",
                category=ConceptCategory.INSTRUMENT,
                difficulty=DifficultyLevel.INTERMEDIATE,
                summary=(
                    "Sukuk are asset-backed certificates representing "
                    "ownership in tangible assets or business projects."
                ),
                body=(
                    "Sukuk pay returns from the underlying asset's cash "
                    "flow (rent, profit) rather than interest. AAOIFI "
                    "recognises seven structures: Ijara, Mudarabah, "
                    "Musharakah, Murabaha, Salam, Istisna, Wakalah."
                ),
                related_ids=("mudarabah", "riba"),
                aliases=("islamic-bonds",),
            ),
        )
    )


def render_concept(concept: Concept) -> str:
    """Operator-readable summary."""
    aliases = f"\n  Aliases: {', '.join(concept.aliases)}" if concept.aliases else ""
    related = f"\n  Related: {', '.join(concept.related_ids)}" if concept.related_ids else ""
    return (
        f"📚 {concept.title} [{concept.category.value}/{concept.difficulty.value}]\n"
        f"  {concept.summary}{aliases}{related}"
    )
