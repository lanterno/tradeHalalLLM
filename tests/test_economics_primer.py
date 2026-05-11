"""Tests for education/economics_primer.py — Round-5 Wave 20.E."""

from __future__ import annotations

import pytest

from halal_trader.education.economics_primer import (
    Concept,
    ConceptCategory,
    DifficultyLevel,
    PrimerCatalog,
    default_catalog,
    related_walk,
    render_concept,
    search,
)


def _concept(
    concept_id: str = "c1",
    title: str = "Concept One",
    category: ConceptCategory = ConceptCategory.STRUCTURE,
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
    summary: str = "Short summary",
    body: str = "Detailed body text",
    related_ids: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
) -> Concept:
    return Concept(
        concept_id=concept_id,
        title=title,
        category=category,
        difficulty=difficulty,
        summary=summary,
        body=body,
        related_ids=related_ids,
        aliases=aliases,
    )


# --- Concept validation ---------------------


def test_concept_valid():
    c = _concept()
    assert c.concept_id == "c1"


def test_concept_empty_id_rejected():
    with pytest.raises(ValueError):
        _concept(concept_id="")


def test_concept_empty_title_rejected():
    with pytest.raises(ValueError):
        _concept(title=" ")


def test_concept_long_title_rejected():
    with pytest.raises(ValueError):
        _concept(title="x" * 300)


def test_concept_long_summary_rejected():
    with pytest.raises(ValueError):
        _concept(summary="x" * 600)


def test_concept_long_body_rejected():
    with pytest.raises(ValueError):
        _concept(body="x" * 6000)


def test_concept_duplicate_alias_rejected():
    with pytest.raises(ValueError):
        _concept(aliases=("a", "a"))


def test_concept_empty_alias_rejected():
    with pytest.raises(ValueError):
        _concept(aliases=("a", " "))


def test_concept_duplicate_related_rejected():
    with pytest.raises(ValueError):
        _concept(related_ids=("c2", "c2"))


def test_concept_self_relation_rejected():
    with pytest.raises(ValueError):
        _concept(concept_id="c1", related_ids=("c1",))


def test_concept_immutable():
    c = _concept()
    with pytest.raises(AttributeError):
        c.title = "x"  # type: ignore[misc]


# --- PrimerCatalog validation ----------------


def test_catalog_basic():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a"),))
    assert cat.by_id("a") is not None


def test_catalog_empty_rejected():
    with pytest.raises(ValueError):
        PrimerCatalog(concepts=())


def test_catalog_duplicate_id_rejected():
    with pytest.raises(ValueError):
        PrimerCatalog(
            concepts=(
                _concept(concept_id="a"),
                _concept(concept_id="a", title="other"),
            )
        )


def test_catalog_orphan_related_id_rejected():
    with pytest.raises(ValueError):
        PrimerCatalog(concepts=(_concept(concept_id="a", related_ids=("nonexistent",)),))


def test_catalog_alias_collision_with_concept_id_rejected():
    with pytest.raises(ValueError):
        PrimerCatalog(
            concepts=(
                _concept(concept_id="a"),
                _concept(concept_id="b", aliases=("a",)),
            )
        )


def test_catalog_duplicate_aliases_across_concepts_rejected():
    with pytest.raises(ValueError):
        PrimerCatalog(
            concepts=(
                _concept(concept_id="a", aliases=("shared",)),
                _concept(concept_id="b", aliases=("shared",)),
            )
        )


# --- by_id / by_alias / lookup ----------------


def test_by_id():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a"),))
    assert cat.by_id("a") is not None
    assert cat.by_id("missing") is None


def test_by_alias():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a", aliases=("alpha",)),))
    assert cat.by_alias("alpha") is not None
    assert cat.by_alias("missing") is None


def test_lookup_id_first():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", aliases=("x",)),
            _concept(concept_id="b", aliases=("a-alt",)),
        )
    )
    # "a" resolves as id (exact match), not as someone else's alias.
    hit = cat.lookup("a")
    assert hit is not None
    assert hit.concept_id == "a"


def test_lookup_alias_fallback():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a", aliases=("alpha",)),))
    hit = cat.lookup("alpha")
    assert hit is not None
    assert hit.concept_id == "a"


# --- by_category / for_difficulty -------------


def test_by_category():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", category=ConceptCategory.STRUCTURE),
            _concept(concept_id="b", category=ConceptCategory.PROHIBITION),
        )
    )
    structures = cat.by_category(ConceptCategory.STRUCTURE)
    assert len(structures) == 1


def test_for_difficulty_includes_lower_levels():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", difficulty=DifficultyLevel.BEGINNER),
            _concept(concept_id="b", difficulty=DifficultyLevel.INTERMEDIATE),
            _concept(concept_id="c", difficulty=DifficultyLevel.ADVANCED),
        )
    )
    intermediate = cat.for_difficulty(DifficultyLevel.INTERMEDIATE)
    assert len(intermediate) == 2
    ids = {c.concept_id for c in intermediate}
    assert "c" not in ids


# --- related_walk -----------------------------


def test_related_walk_includes_self():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a"),))
    walk = related_walk(cat, start_id="a")
    assert walk == (cat.by_id("a"),)


def test_related_walk_depth_one():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", related_ids=("b",)),
            _concept(concept_id="b"),
        )
    )
    walk = related_walk(cat, start_id="a", max_depth=1)
    ids = [c.concept_id for c in walk]
    assert ids == ["a", "b"]


def test_related_walk_depth_two_chain():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", related_ids=("b",)),
            _concept(concept_id="b", related_ids=("c",)),
            _concept(concept_id="c"),
        )
    )
    walk = related_walk(cat, start_id="a", max_depth=2)
    assert len(walk) == 3


def test_related_walk_depth_zero_self_only():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", related_ids=("b",)),
            _concept(concept_id="b"),
        )
    )
    walk = related_walk(cat, start_id="a", max_depth=0)
    assert len(walk) == 1


def test_related_walk_handles_cycles():
    cat = PrimerCatalog(
        concepts=(
            _concept(concept_id="a", related_ids=("b",)),
            _concept(concept_id="b", related_ids=("a",)),
        )
    )
    walk = related_walk(cat, start_id="a", max_depth=5)
    # No infinite loop; visited once.
    assert len(walk) == 2


def test_related_walk_negative_depth_rejected():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a"),))
    with pytest.raises(ValueError):
        related_walk(cat, start_id="a", max_depth=-1)


def test_related_walk_unknown_start_rejected():
    cat = PrimerCatalog(concepts=(_concept(concept_id="a"),))
    with pytest.raises(ValueError):
        related_walk(cat, start_id="missing")


# --- search ----------------------------------


def test_search_empty_query_returns_empty():
    cat = default_catalog()
    assert search(cat, "") == ()
    assert search(cat, " ") == ()


def test_search_exact_id_match():
    cat = default_catalog()
    out = search(cat, "riba")
    assert out and out[0].concept_id == "riba"


def test_search_alias_match():
    cat = default_catalog()
    out = search(cat, "interest")
    assert out and out[0].concept_id == "riba"


def test_search_title_substring():
    cat = default_catalog()
    out = search(cat, "profit")  # matches "Profit-Sharing" in Mudarabah title
    ids = {c.concept_id for c in out}
    assert "mudarabah" in ids


def test_search_case_insensitive():
    cat = default_catalog()
    out = search(cat, "RIBA")
    assert out and out[0].concept_id == "riba"


def test_search_invalid_max_results_rejected():
    cat = default_catalog()
    with pytest.raises(ValueError):
        search(cat, "riba", max_results=0)


def test_search_caps_results():
    cat = default_catalog()
    out = search(cat, "halal", max_results=1)
    assert len(out) <= 1


# --- default_catalog ------------------------


def test_default_catalog_loads():
    cat = default_catalog()
    assert cat.by_id("riba") is not None
    assert cat.by_id("mudarabah") is not None


def test_default_catalog_graph_integrity():
    """Default catalogue's related_ids all resolve."""
    cat = default_catalog()
    for c in cat.concepts:
        for r in c.related_ids:
            assert cat.by_id(r) is not None


# --- Render --------------------------------


def test_render_concept_includes_title():
    c = _concept(title="Mudarabah")
    out = render_concept(c)
    assert "Mudarabah" in out


def test_render_concept_aliases_visible():
    c = _concept(aliases=("alpha", "beta"))
    out = render_concept(c)
    assert "alpha" in out
    assert "beta" in out


def test_render_concept_no_aliases_section_when_empty():
    c = _concept(aliases=())
    out = render_concept(c)
    assert "Aliases" not in out
