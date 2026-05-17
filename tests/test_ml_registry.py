"""Tests for `ml/registry.py` (model registry + semver + lineage).

Pins the semver parse / compare / bump rules, the lineage-chain
walk including the corrupt-registry rejections, the
`should_promote` policy across every check (compatibility,
lineage continuity, fitness uplift), and the cold-start
bootstrap path.
"""

from __future__ import annotations

import pytest

from halal_trader.ml.registry import (
    ModelLineage,
    PromotionDecision,
    PromotionPolicy,
    RegistryRecord,
    Semver,
    lineage_chain,
    render_lineage,
    should_promote,
)

# ── Semver parse + compare ───────────────────────────────


def test_semver_parses_with_or_without_v_prefix():
    assert Semver.parse("v1.2.3") == Semver(1, 2, 3)
    assert Semver.parse("1.2.3") == Semver(1, 2, 3)


def test_semver_strips_whitespace():
    assert Semver.parse("  v1.2.3  ") == Semver(1, 2, 3)


def test_semver_rejects_malformed_strings():
    """Pin: malformed strings raise rather than rounding to (0,0,0).
    A corrupt registry row must surface immediately."""
    for bad in ("v1.2", "1.2", "v1.2.3.4", "abc", ""):
        with pytest.raises(ValueError, match="invalid semver"):
            Semver.parse(bad)


def test_semver_str_round_trip():
    sv = Semver(1, 2, 3)
    assert Semver.parse(str(sv)) == sv


def test_semver_orders_by_tuple_comparison():
    """Pin: (1, 2, 3) < (1, 3, 0) — minor wins; major wins over both."""
    assert Semver(1, 0, 0) < Semver(1, 0, 1)
    assert Semver(1, 0, 1) < Semver(1, 1, 0)
    assert Semver(1, 9, 9) < Semver(2, 0, 0)
    assert not Semver(2, 0, 0) < Semver(1, 9, 9)


def test_semver_equality():
    assert Semver(1, 2, 3) == Semver(1, 2, 3)
    assert Semver(1, 2, 3) != Semver(1, 2, 4)


# ── Semver bumps ─────────────────────────────────────────


def test_bump_major_resets_minor_and_patch():
    """Pin: a major bump signals a backwards-incompatible change;
    the minor + patch counters reset to 0."""
    assert Semver(1, 2, 3).bump_major() == Semver(2, 0, 0)


def test_bump_minor_resets_patch():
    assert Semver(1, 2, 3).bump_minor() == Semver(1, 3, 0)


def test_bump_patch_only_increments_patch():
    assert Semver(1, 2, 3).bump_patch() == Semver(1, 2, 4)


def test_bump_methods_return_new_instances():
    """Frozen dataclasses; the bump must not try to mutate."""
    sv = Semver(1, 2, 3)
    sv.bump_minor()
    assert sv == Semver(1, 2, 3)


# ── compatibility ────────────────────────────────────────


def test_is_compatible_with_same_major():
    assert Semver(1, 0, 0).is_compatible_with(Semver(1, 9, 9))
    assert Semver(1, 5, 3).is_compatible_with(Semver(1, 0, 0))


def test_is_compatible_with_different_major():
    """Pin: a major mismatch means feature vector reshape — never
    compatible."""
    assert not Semver(1, 9, 9).is_compatible_with(Semver(2, 0, 0))


# ── RegistryRecord validation ────────────────────────────


def test_registry_record_rejects_bad_payload_kind():
    with pytest.raises(ValueError, match="payload_kind"):
        RegistryRecord(name="x", version=Semver(1, 0, 0), payload_kind="weird")


def test_registry_record_rejects_negative_size():
    with pytest.raises(ValueError, match="size_bytes"):
        RegistryRecord(name="x", version=Semver(1, 0, 0), size_bytes=-1)


def test_registry_record_accepts_valid_inputs():
    r = RegistryRecord(
        name="anomaly_detector",
        version=Semver(1, 2, 3),
        lineage=ModelLineage(
            parent_version=Semver(1, 2, 2),
            training_run_id="run-abc123",
            fitness_score=0.85,
        ),
        payload_kind="bytes",
        size_bytes=1024,
    )
    assert r.name == "anomaly_detector"
    assert r.lineage.fitness_score == 0.85


# ── lineage_chain ────────────────────────────────────────


def _record(
    *,
    name: str = "model",
    version: Semver,
    parent: Semver | None = None,
    fitness: float | None = None,
    is_active: bool = False,
) -> RegistryRecord:
    return RegistryRecord(
        name=name,
        version=version,
        lineage=ModelLineage(parent_version=parent, fitness_score=fitness),
        is_active=is_active,
    )


def test_lineage_chain_walks_from_root_to_target():
    """Pin: the chain returns oldest → newest so the dashboard
    renders left-to-right chronologically."""
    v1 = _record(version=Semver(1, 0, 0))
    v2 = _record(version=Semver(1, 1, 0), parent=Semver(1, 0, 0))
    v3 = _record(version=Semver(1, 2, 0), parent=Semver(1, 1, 0))
    chain = lineage_chain([v1, v2, v3], target=v3)
    assert [r.version for r in chain] == [Semver(1, 0, 0), Semver(1, 1, 0), Semver(1, 2, 0)]


def test_lineage_chain_handles_root_record():
    """A v1.0.0 with no parent returns a chain of length 1."""
    v1 = _record(version=Semver(1, 0, 0), parent=None)
    chain = lineage_chain([v1], target=v1)
    assert chain == [v1]


def test_lineage_chain_filters_to_target_name_only():
    """Multiple model families coexist in the registry; the walk
    must ignore other names."""
    v1_a = _record(name="anomaly", version=Semver(1, 0, 0))
    v1_b = _record(name="anomaly", version=Semver(1, 1, 0), parent=Semver(1, 0, 0))
    other = _record(name="signal", version=Semver(1, 0, 0))
    chain = lineage_chain([v1_a, v1_b, other], target=v1_b)
    assert all(r.name == "anomaly" for r in chain)
    assert len(chain) == 2


def test_lineage_chain_rejects_missing_parent():
    """Corrupt registry: the candidate references a parent that
    isn't in the records list. Pin the rejection so it surfaces
    immediately rather than silently truncating."""
    orphan = _record(version=Semver(1, 1, 0), parent=Semver(1, 0, 0))
    with pytest.raises(ValueError, match="parent version"):
        lineage_chain([orphan], target=orphan)


def test_lineage_chain_rejects_cycle():
    """Pin: a corrupt registry where v1 → v2 → v1 must not loop
    forever; the walk surfaces the cycle and raises."""
    v1 = _record(version=Semver(1, 0, 0), parent=Semver(1, 1, 0))
    v2 = _record(version=Semver(1, 1, 0), parent=Semver(1, 0, 0))
    with pytest.raises(ValueError, match="cycle"):
        lineage_chain([v1, v2], target=v2)


def test_lineage_chain_rejects_duplicate_versions():
    """A model name with two records at the same version means the
    registry has been corrupted by a race; pin the rejection."""
    v1a = _record(version=Semver(1, 0, 0))
    v1b = _record(version=Semver(1, 0, 0))
    with pytest.raises(ValueError, match="duplicate version"):
        lineage_chain([v1a, v1b], target=v1a)


# ── should_promote: cold start ───────────────────────────


def test_promote_cold_start_with_fitness_passes():
    candidate = _record(version=Semver(1, 0, 0), fitness=0.5)
    decision = should_promote(candidate=candidate, incumbent=None)
    assert decision.passed
    assert "cold start" in decision.reasons[0].lower()


def test_promote_cold_start_without_fitness_fails():
    """Pin: the registry refuses to bootstrap on a model with no
    measured fitness — caller must record one before promotion."""
    candidate = _record(version=Semver(1, 0, 0), fitness=None)
    decision = should_promote(candidate=candidate, incumbent=None)
    assert not decision.passed
    assert any("fitness" in f.lower() for f in decision.failures)


# ── should_promote: compatibility check ──────────────────


def test_promote_blocks_major_version_change_by_default():
    """Pin: a v2.0.0 candidate against a v1.x incumbent is
    blocked — the operator must explicitly opt in."""
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.5)
    candidate = _record(version=Semver(2, 0, 0), parent=Semver(1, 5, 0), fitness=0.6)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert not decision.passed
    assert any("major-version" in f for f in decision.failures)


def test_promote_allows_major_change_when_policy_relaxed():
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.5)
    candidate = _record(version=Semver(2, 0, 0), parent=Semver(1, 5, 0), fitness=0.6)
    decision = should_promote(
        candidate=candidate,
        incumbent=incumbent,
        policy=PromotionPolicy(require_compatible_major=False),
    )
    assert decision.passed


# ── should_promote: lineage check ────────────────────────


def test_promote_rejects_candidate_without_lineage_to_incumbent():
    """A fresh-trained model that didn't descend from the current
    production model must not bypass the chain. Pin so a registry
    refactor can't silently let a sideloaded model through."""
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.5)
    candidate = _record(version=Semver(1, 6, 0), parent=None, fitness=0.6)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert not decision.passed
    assert any("parent" in f for f in decision.failures)


def test_promote_accepts_candidate_with_correct_lineage():
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.5)
    candidate = _record(version=Semver(1, 6, 0), parent=Semver(1, 5, 0), fitness=0.6)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert decision.passed


def test_promote_lineage_check_can_be_disabled():
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.5)
    candidate = _record(version=Semver(1, 6, 0), parent=None, fitness=0.6)
    decision = should_promote(
        candidate=candidate,
        incumbent=incumbent,
        policy=PromotionPolicy(require_lineage=False),
    )
    assert decision.passed


# ── should_promote: fitness uplift ───────────────────────


def test_promote_blocks_when_fitness_below_threshold():
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.50)
    candidate = _record(version=Semver(1, 5, 1), parent=Semver(1, 5, 0), fitness=0.51)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    # uplift 0.01 < default 0.05 threshold
    assert not decision.passed
    assert any("uplift" in f for f in decision.failures)


def test_promote_passes_when_fitness_above_threshold():
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.50)
    candidate = _record(version=Semver(1, 5, 1), parent=Semver(1, 5, 0), fitness=0.60)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert decision.passed


def test_promote_blocks_on_negative_uplift():
    """Symmetric: a worse candidate must never promote."""
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.50)
    candidate = _record(version=Semver(1, 5, 1), parent=Semver(1, 5, 0), fitness=0.45)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert not decision.passed


def test_promote_handles_missing_candidate_fitness():
    incumbent = _record(version=Semver(1, 5, 0), fitness=0.5)
    candidate = _record(version=Semver(1, 5, 1), parent=Semver(1, 5, 0), fitness=None)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert not decision.passed
    assert any("fitness_score" in f for f in decision.failures)


def test_promote_passes_on_missing_incumbent_fitness():
    """Partial bootstrap: incumbent never recorded fitness — the
    registry promotes on the candidate's score alone."""
    incumbent = _record(version=Semver(1, 5, 0), fitness=None)
    candidate = _record(version=Semver(1, 5, 1), parent=Semver(1, 5, 0), fitness=0.5)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert decision.passed


# ── render_lineage ───────────────────────────────────────


def test_render_lineage_handles_empty_chain():
    text = render_lineage([])
    assert "(empty)" in text


def test_render_lineage_includes_each_record():
    chain = [
        _record(version=Semver(1, 0, 0), fitness=0.40),
        _record(version=Semver(1, 1, 0), parent=Semver(1, 0, 0), fitness=0.50),
    ]
    text = render_lineage(chain)
    assert "v1.0.0" in text
    assert "v1.1.0" in text


def test_render_lineage_marks_active_record_with_star():
    """Pin: the operator-facing display must call out which model
    is currently serving traffic."""
    chain = [
        _record(version=Semver(1, 0, 0), fitness=0.40),
        _record(version=Semver(1, 1, 0), parent=Semver(1, 0, 0), fitness=0.50, is_active=True),
    ]
    text = render_lineage(chain)
    assert "★" in text


def test_render_lineage_handles_missing_fitness():
    chain = [_record(version=Semver(1, 0, 0), fitness=None)]
    text = render_lineage(chain)
    assert "n/a" in text


# ── decision structure ───────────────────────────────────


def test_promotion_decision_carries_both_sides():
    incumbent = _record(version=Semver(1, 0, 0), fitness=0.5)
    candidate = _record(version=Semver(1, 0, 1), parent=Semver(1, 0, 0), fitness=0.6)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    assert isinstance(decision, PromotionDecision)
    assert decision.candidate.version == Semver(1, 0, 1)
    assert decision.incumbent is not None
    assert decision.incumbent.version == Semver(1, 0, 0)


def test_decision_is_immutable():
    incumbent = _record(version=Semver(1, 0, 0), fitness=0.5)
    candidate = _record(version=Semver(1, 0, 1), parent=Semver(1, 0, 0), fitness=0.6)
    decision = should_promote(candidate=candidate, incumbent=incumbent)
    with pytest.raises(Exception):
        decision.passed = False  # type: ignore[misc]
