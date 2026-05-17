"""Model registry — semver + lineage on top of `db.ml_artefacts`.

Round-4 wave 6.A: the existing `db/ml_artefacts.py` keys models by
``(name, version: int)`` — useful for "load the latest", less
useful for "which retrain produced this model" or "is v1.2.3
compatible with v1.2.0?". This module adds semantic versioning,
lineage metadata, and a promotion-comparison helper without
touching the persisted table.

Three concerns:

* **Semantic versioning.** `Semver(major, minor, patch)` — parse
  from `"v1.2.3"` / `"1.2.3"`, compare with `<` / `>`, bump for
  the operator's chosen semantics. Major-bumps are
  *not-backwards-compatible* (e.g. feature vector reshape);
  minor-bumps add behaviour without breaking inference; patch-
  bumps are pure retrains on a fresh window.
* **Lineage.** Every record carries the parent version it was
  trained from, the training-run identifier (so the operator can
  trace back to the data window + hyperparams + commit hash),
  and the measured fitness score. The chain lets the dashboard
  render "v1.2.3 ← v1.2.2 ← v1.2.1" + per-step Sharpe deltas.
* **Promotion comparison.** The `should_promote(current, candidate)`
  helper composes lineage's fitness with the existing
  `PromotionThresholds` (Wave 4.F) — a candidate must
  out-perform the current model by at least
  `min_fitness_uplift` before the registry advances the active
  pointer.

Why a separate module rather than extending `ml_artefacts.py`:

* `ml_artefacts.py` is the persistence layer (SQL, pickle, JSONB).
  The registry is *policy* — semver / lineage / promotion rules.
  Keeping them apart means a future SQL refactor (e.g. switch to
  S3 + manifest) doesn't ripple into the policy contract.
* The registry is pure-Python so its tests run without Postgres,
  matching the rest of the Round-4 isolated-module pattern.

Halal alignment: the registry is metadata only. It never opens a
position or screens an asset. Every promotion is operator-
initiated and recorded in the lineage chain so a future audit can
reproduce who promoted what when.

Pure-Python; no DB / network / async. Frozen dataclasses safe to
cache for the dashboard's "model timeline" tile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

# ── Semver ────────────────────────────────────────────────


_SEMVER_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True, order=True)
class Semver:
    """Major / minor / patch triple.

    Field order matters — the dataclass `order=True` means
    Python's tuple comparison drives `<` / `>` correctly:
    `(1, 2, 3) < (1, 3, 0)` because minor differs first.

    Pre-release and build metadata (the `+sha`, `-rc1` suffixes
    of the full SemVer 2.0 spec) are deliberately omitted — the
    bot's models don't ship pre-release; if they ever do, this
    is a one-line addition with bumped tests.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> "Semver":
        """Parse `"v1.2.3"` or `"1.2.3"`. Raises `ValueError` on
        anything else — pin so a malformed registry row surfaces
        immediately rather than silently rounding to (0,0,0)."""
        match = _SEMVER_PATTERN.match(text.strip())
        if match is None:
            raise ValueError(
                f"invalid semver {text!r}; expected major.minor.patch (with optional 'v' prefix)"
            )
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    def bump_major(self) -> "Semver":
        """Major bump: feature vector reshape, schema-incompatible
        change. Resets minor + patch to 0."""
        return Semver(self.major + 1, 0, 0)

    def bump_minor(self) -> "Semver":
        """Minor bump: new behaviour, backwards-compatible. Resets
        patch to 0."""
        return Semver(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "Semver":
        """Patch bump: pure retrain on new window, no algorithmic
        change."""
        return Semver(self.major, self.minor, self.patch + 1)

    def is_compatible_with(self, other: "Semver") -> bool:
        """Same major version → compatible (i.e. same feature
        vector shape, same prediction interface). Pin: a major
        mismatch is the only thing that can't be hot-swapped."""
        return self.major == other.major


# ── Lineage ───────────────────────────────────────────────


@dataclass(frozen=True)
class ModelLineage:
    """Provenance chain for one model record.

    ``parent_version`` is `None` for v1.0.0 / first-of-its-kind
    records. ``training_run_id`` is the operator's unique ID for
    the run that produced this model (commit hash, retrainer
    timestamp, RL training run ID — caller's choice). Fitness is
    the headline score the operator used to decide whether to
    promote (Sharpe, OOS return, F1 — caller's choice; pinned via
    name on the lineage rather than the registry so different
    model families can use different metrics).
    """

    parent_version: Semver | None = None
    training_run_id: str = ""
    fitness_score: float | None = None
    fitness_metric_name: str = "sharpe"
    notes: str = ""


# ── Registry record ───────────────────────────────────────


@dataclass(frozen=True)
class RegistryRecord:
    """One (name, version) row with lineage + payload metadata.

    ``payload_kind`` distinguishes JSON-shaped artefacts (slippage
    model, calibration curve) from pickled sklearn / xgboost
    blobs — matches the existing `ml_artefacts` table's
    `payload_json` vs `payload_bytes` split.

    ``size_bytes`` is metadata only — operator's "is the new
    model 10× larger than the previous one" sanity check.
    """

    name: str
    version: Semver
    lineage: ModelLineage = field(default_factory=ModelLineage)
    created_at: datetime | None = None
    payload_kind: str = "json"  # "json" or "bytes"
    size_bytes: int | None = None
    is_active: bool = False  # serving traffic right now

    def __post_init__(self) -> None:
        if self.payload_kind not in {"json", "bytes"}:
            raise ValueError(f"payload_kind must be 'json' or 'bytes'; got {self.payload_kind!r}")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0; got {self.size_bytes}")


# ── Lineage chain ─────────────────────────────────────────


def lineage_chain(
    records: Iterable[RegistryRecord], *, target: RegistryRecord
) -> list[RegistryRecord]:
    """Walk back from ``target`` to the root using
    ``lineage.parent_version``.

    Returns the chain from oldest to newest (root first, target
    last). Pin so the dashboard's lineage panel renders left-to-
    right chronologically without re-sorting.

    Raises `ValueError` if a parent version is referenced but not
    present in ``records`` — pin so a corrupt registry surfaces
    rather than silently truncating the chain.
    """
    by_version: dict[Semver, RegistryRecord] = {}
    for r in records:
        if r.name != target.name:
            continue
        if r.version in by_version:
            raise ValueError(f"duplicate version {r.version} for {r.name!r}")
        by_version[r.version] = r

    chain_newest_first: list[RegistryRecord] = []
    seen: set[Semver] = set()
    current: RegistryRecord | None = target
    while current is not None:
        if current.version in seen:
            raise ValueError(f"lineage cycle detected at {current.version} for {current.name!r}")
        seen.add(current.version)
        chain_newest_first.append(current)
        parent_version = current.lineage.parent_version
        if parent_version is None:
            break
        parent = by_version.get(parent_version)
        if parent is None:
            raise ValueError(f"parent version {parent_version} of {current.version} not in records")
        current = parent

    return list(reversed(chain_newest_first))


# ── Promotion ─────────────────────────────────────────────


@dataclass(frozen=True)
class PromotionPolicy:
    """How much better a candidate must be before promoting.

    ``min_fitness_uplift`` is the absolute difference in the
    fitness metric — for Sharpe, 0.05 means "must be at least
    0.05 higher than the current production model". Set higher
    for noisy metrics, lower for stable ones.

    ``require_compatible_major`` blocks promote-to-live across a
    major-version change unless the operator explicitly opts in.
    Pin to True by default — major-bumps need a new dashboard tile
    + a re-test of every downstream consumer.

    ``require_lineage`` rejects candidates whose
    `parent_version` doesn't match the current model's version.
    Stops a fresh-trained model from short-circuiting the chain.
    """

    min_fitness_uplift: float = 0.05
    require_compatible_major: bool = True
    require_lineage: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of a promotion comparison.

    ``passed`` is True iff every check cleared. ``reasons`` lists
    every check + its status; the dashboard renders the failures
    with their `passed=False` markers."""

    passed: bool
    candidate: RegistryRecord
    incumbent: RegistryRecord | None
    reasons: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def should_promote(
    *,
    candidate: RegistryRecord,
    incumbent: RegistryRecord | None,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Decide whether ``candidate`` should replace ``incumbent``.

    ``incumbent`` is `None` when the registry has no active model
    yet (cold start). In that case the candidate is promoted as
    long as it carries a fitness score — pin so the registry can
    bootstrap from an empty state.

    Pin: returns the *same* `PromotionDecision` shape regardless
    of pass/fail — the dashboard renders both sides off `passed`.
    """
    policy = policy or PromotionPolicy()
    reasons: list[str] = []
    failures: list[str] = []

    # Cold-start: any candidate with a fitness score is fine.
    if incumbent is None:
        if candidate.lineage.fitness_score is None:
            failures.append("candidate has no fitness_score")
            return PromotionDecision(
                passed=False,
                candidate=candidate,
                incumbent=None,
                reasons=["cold start, but candidate fitness_score missing"],
                failures=failures,
            )
        reasons.append(
            f"cold start — promoting candidate "
            f"{candidate.version} (fitness {candidate.lineage.fitness_score})"
        )
        return PromotionDecision(
            passed=True,
            candidate=candidate,
            incumbent=None,
            reasons=reasons,
            failures=[],
        )

    # Compatibility check.
    if policy.require_compatible_major and not candidate.version.is_compatible_with(
        incumbent.version
    ):
        failures.append(
            f"major-version change ({incumbent.version} → {candidate.version}) blocked by policy"
        )
    else:
        reasons.append(f"version {candidate.version} compatible with incumbent {incumbent.version}")

    # Lineage check.
    if policy.require_lineage:
        parent = candidate.lineage.parent_version
        if parent is None or parent != incumbent.version:
            failures.append(
                f"candidate's parent {parent} does not match incumbent {incumbent.version}"
            )
        else:
            reasons.append(f"lineage continuous: parent {parent} matches incumbent")

    # Fitness uplift check.
    candidate_fitness = candidate.lineage.fitness_score
    incumbent_fitness = incumbent.lineage.fitness_score
    if candidate_fitness is None:
        failures.append("candidate has no fitness_score")
    elif incumbent_fitness is None:
        # Incumbent never recorded fitness — promote on the
        # candidate's score alone (the registry is partially
        # self-bootstrapping).
        reasons.append(
            f"incumbent has no fitness_score; promoting on candidate's "
            f"{candidate.lineage.fitness_metric_name}={candidate_fitness}"
        )
    else:
        uplift = candidate_fitness - incumbent_fitness
        if uplift >= policy.min_fitness_uplift:
            reasons.append(f"fitness uplift {uplift:+.4f} ≥ threshold {policy.min_fitness_uplift}")
        else:
            failures.append(
                f"fitness uplift {uplift:+.4f} below threshold {policy.min_fitness_uplift} "
                f"(candidate {candidate_fitness} vs incumbent {incumbent_fitness})"
            )

    passed = len(failures) == 0
    return PromotionDecision(
        passed=passed,
        candidate=candidate,
        incumbent=incumbent,
        reasons=reasons,
        failures=failures,
    )


# ── Render helper ─────────────────────────────────────────


def render_lineage(chain: list[RegistryRecord]) -> str:
    """Pretty multi-line summary of a lineage chain.

    Mirrors the visual shape of `crypto/stress.render_report` and
    `core/promotion_gate.render_verdict` so the operator sees a
    consistent layout across observability tools."""
    if not chain:
        return "=== Lineage chain ===\n(empty)"
    lines = [f"=== Lineage chain — {chain[0].name} ==="]
    for record in chain:
        active_marker = " ★" if record.is_active else ""
        fitness = record.lineage.fitness_score
        fitness_str = (
            f"{record.lineage.fitness_metric_name}={fitness:.4f}"
            if fitness is not None
            else f"{record.lineage.fitness_metric_name}=n/a"
        )
        lines.append(
            f"  {record.version}{active_marker}  "
            f"{fitness_str}  run={record.lineage.training_run_id or 'n/a'}"
        )
        if record.lineage.notes:
            lines.append(f"      notes: {record.lineage.notes}")
    return "\n".join(lines)


__all__ = [
    "ModelLineage",
    "PromotionDecision",
    "PromotionPolicy",
    "RegistryRecord",
    "Semver",
    "lineage_chain",
    "render_lineage",
    "should_promote",
]
