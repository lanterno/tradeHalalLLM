"""Feature schema + migration helpers for the ML stack.

Round-4 wave 6.B: today's bot computes RSI / MACD / vol-ratio / ATR
on the fly and feeds them straight into the model. When a feature
is renamed (`rsi_14` → `rsi_period_14`), removed (drop the broken
ATR computation), or its dtype changes (float → int8 quantile
bucket), every downstream consumer breaks unless the rename is
synchronised across training, inference, replay, and backtest at
the same commit.

This module is the migration layer that decouples those consumers.
A `FeatureSchema` declares the set of features a model expects;
`migrate(payload, from_schema, to_schema, rules)` walks a feature
dict from one schema to another using `MigrationRule`s
(`Rename` / `Drop` / `Cast`). Validation (`validate(payload,
schema)`) checks for required features + dtype compatibility
before inference.

Why declarative rules rather than a function-per-migration:

* Migrations stack — a model trained on schema v1.2.0 must be
  served from schema v1.4.0 when the live cycle has migrated
  twice. A rule list composes; a function-per-migration would
  force the operator to chain `migrate_v1_2_to_v1_3` →
  `migrate_v1_3_to_v1_4` manually every time.
* Rules are inspectable — the dashboard renders the schema diff
  by walking the rules, no need to read function bodies.
* Rules are testable — each rule is a small dataclass, trivially
  unit-tested in isolation. Pin so a refactor that "consolidates"
  two migrations into one breaks loud.

Halal alignment: schema migration is metadata only — never opens
a position, never bypasses the screener. Operators can spot a
schema regression (a feature renamed without a migration rule)
*before* it produces a silently-wrong model output.

Pure-Python; no NumPy / DB / network. Versioning uses the
`Semver` from `ml/registry.py` (Wave 6.A) so feature schema
versions and model versions follow the same parse / compare
rules — operator only learns one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from halal_trader.ml.registry import Semver

# ── Vocabulary ────────────────────────────────────────────


class FeatureDType(str, Enum):
    """Supported feature dtypes.

    Pinned to a small set the bot actually uses:

    * ``float`` — RSI, MACD histogram, ATR, vol ratio (the common case).
    * ``int`` — counts (e.g. number of catalysts in the next 24h).
    * ``bool`` — flags (regime gate active, halt engaged).
    * ``str`` — categorical labels (regime name, sector tag).

    Anything else raises so the schema can't accept a `np.float16`
    or a custom enum without an explicit addition here + a matching
    test case.
    """

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STR = "str"


_DTYPE_PYTHON_TYPES: dict[FeatureDType, tuple[type, ...]] = {
    FeatureDType.FLOAT: (float, int),  # int auto-promotes to float
    FeatureDType.INT: (int,),
    FeatureDType.BOOL: (bool,),
    FeatureDType.STR: (str,),
}


def _matches_dtype(value: Any, dtype: FeatureDType) -> bool:
    """Pin: bool is a subtype of int in Python, so a True passed
    where INT is expected technically matches. We intentionally
    accept that — a 1/0 flag computed elsewhere as bool is a
    common-enough case. The reverse (int passed for BOOL) is
    rejected because misclassifying 5 as True is a real bug."""
    types = _DTYPE_PYTHON_TYPES[dtype]
    if dtype == FeatureDType.BOOL:
        return isinstance(value, bool)
    if dtype == FeatureDType.INT:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, types)


# ── Spec / Schema ─────────────────────────────────────────


@dataclass(frozen=True)
class FeatureSpec:
    """One feature's contract.

    ``producer`` is the operator's free-form label for what
    computes the feature (`"crypto.indicators.rsi"`,
    `"sentiment.cryptopanic"`). Used by the dashboard's lineage
    tile and by `pyproject.toml`-style tooling that wants to
    surface "which module produced this column".

    ``required`` defaults to True — a feature listed in the
    schema must appear in the payload unless the spec opts out.
    """

    name: str
    dtype: FeatureDType
    description: str = ""
    producer: str = ""
    required: bool = True


@dataclass(frozen=True)
class FeatureSchema:
    """A versioned set of `FeatureSpec`.

    ``version`` is a `Semver` from `ml/registry.py` so feature
    schema bumps follow the same major/minor/patch semantics as
    model versioning — operator only learns one rule set.
    """

    name: str
    version: Semver
    features: tuple[FeatureSpec, ...] = ()

    def feature_by_name(self, name: str) -> FeatureSpec | None:
        for f in self.features:
            if f.name == name:
                return f
        return None

    def names(self) -> set[str]:
        return {f.name for f in self.features}

    def __post_init__(self) -> None:
        # Pin: feature names must be unique within a schema. A
        # corrupt config that lists `rsi_14` twice silently drops
        # one definition under the dict-of-features model.
        seen: set[str] = set()
        for f in self.features:
            if f.name in seen:
                raise ValueError(f"duplicate feature name in schema: {f.name!r}")
            seen.add(f.name)


# ── Migration rules ───────────────────────────────────────


class MigrationRule:
    """Marker base class — subclasses are dataclass leaves."""

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Rename(MigrationRule):
    """Rename a feature in flight without touching its value."""

    from_name: str
    to_name: str

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.from_name == self.to_name:
            raise ValueError(f"Rename from_name == to_name ({self.from_name!r}); refuse no-op")
        if self.from_name not in payload:
            return payload
        if self.to_name in payload:
            raise ValueError(f"Rename collision: {self.to_name!r} already present in payload")
        out = dict(payload)
        out[self.to_name] = out.pop(self.from_name)
        return out


@dataclass(frozen=True)
class Drop(MigrationRule):
    """Remove a feature. Safe no-op if the feature isn't in the
    payload (a previous migration may already have removed it)."""

    name: str

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.name not in payload:
            return payload
        out = dict(payload)
        out.pop(self.name)
        return out


@dataclass(frozen=True)
class AddDefault(MigrationRule):
    """Add a feature with a default value. Used when a new
    feature is added to the schema and the migration must
    backfill in-flight payloads."""

    name: str
    default_value: Any

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.name in payload:
            return payload
        out = dict(payload)
        out[self.name] = self.default_value
        return out


@dataclass(frozen=True)
class Cast(MigrationRule):
    """Coerce a feature's value to a new dtype.

    Pin: silent failures forbidden. If the cast can't succeed
    (e.g. `int("abc")`), the migration raises rather than dropping
    the value or substituting a sentinel — operator wants to know
    immediately."""

    name: str
    target_dtype: FeatureDType

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.name not in payload:
            return payload
        out = dict(payload)
        original = out[self.name]
        try:
            if self.target_dtype == FeatureDType.FLOAT:
                out[self.name] = float(original)
            elif self.target_dtype == FeatureDType.INT:
                # Refuse a lossy float→int cast — pin so a 1.7
                # silently rounding to 1 doesn't slip through.
                if isinstance(original, float) and not original.is_integer():
                    raise ValueError(f"refusing lossy float→int cast on {self.name!r}: {original}")
                out[self.name] = int(original)
            elif self.target_dtype == FeatureDType.BOOL:
                # Strict: only accept exact 0/1/True/False.
                if isinstance(original, bool):
                    out[self.name] = original
                elif original in (0, 1):
                    out[self.name] = bool(original)
                else:
                    raise ValueError(f"refusing ambiguous bool cast on {self.name!r}: {original!r}")
            elif self.target_dtype == FeatureDType.STR:
                out[self.name] = str(original)
            else:  # pragma: no cover — exhaustive by enum
                raise ValueError(f"unknown target_dtype {self.target_dtype}")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cast of {self.name!r} to {self.target_dtype.value} failed: {exc}"
            ) from exc
        return out


# ── Migration driver ──────────────────────────────────────


@dataclass(frozen=True)
class MigrationPlan:
    """An ordered list of rules to walk a payload from
    ``from_version`` to ``to_version``.

    Plans are built and registered up-front; `migrate` looks them
    up by (schema_name, from_version, to_version). Pin: the
    registry is keyed by both versions so adding a v1.4 → v1.5
    plan can't accidentally apply when migrating v1.2 → v1.5.
    """

    schema_name: str
    from_version: Semver
    to_version: Semver
    rules: tuple[MigrationRule, ...] = field(default_factory=tuple)

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        for rule in self.rules:
            out = rule.apply(out)
        return out


def migrate(
    payload: Mapping[str, Any],
    *,
    from_schema: FeatureSchema,
    to_schema: FeatureSchema,
    plans: Iterable[MigrationPlan] = (),
) -> dict[str, Any]:
    """Walk ``payload`` from ``from_schema`` to ``to_schema``.

    Pin: schema names must match. Migrating a `rsi_features`
    schema to a `sentiment_features` schema is nonsense and
    surfaces immediately rather than producing garbage output.

    When the versions are equal, returns the payload unchanged
    (still a copy — pin the no-mutation invariant). When versions
    differ, finds the matching plan; if none exists, raises so the
    operator knows a migration rule is missing rather than
    inferring incorrectly.
    """
    if from_schema.name != to_schema.name:
        raise ValueError(f"schema name mismatch: {from_schema.name!r} → {to_schema.name!r}")
    if from_schema.version == to_schema.version:
        return dict(payload)
    matching = [
        p
        for p in plans
        if p.schema_name == from_schema.name
        and p.from_version == from_schema.version
        and p.to_version == to_schema.version
    ]
    if not matching:
        raise ValueError(
            f"no migration plan for {from_schema.name!r} "
            f"{from_schema.version} → {to_schema.version}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"ambiguous migration: {len(matching)} plans for "
            f"{from_schema.name!r} {from_schema.version} → {to_schema.version}"
        )
    return matching[0].apply(dict(payload))


# ── Validation ────────────────────────────────────────────


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of validating a payload against a schema.

    ``passed`` is True iff every required feature is present with
    a compatible dtype. ``violations`` lists each failure in
    operator-readable form."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)


def validate(payload: Mapping[str, Any], schema: FeatureSchema) -> ValidationOutcome:
    """Check the payload against the schema's feature specs.

    Required features must be present *and* type-compatible.
    Optional features that *are* present must be type-compatible
    (no point checking dtype for a feature the producer skipped).
    Extras (features present in the payload but not declared)
    are reported separately — non-fatal but the dashboard surfaces
    them so a stale producer doesn't silently flow data into
    inference.
    """
    violations: list[str] = []
    for spec in schema.features:
        if spec.name not in payload:
            if spec.required:
                violations.append(f"missing required feature {spec.name!r}")
            continue
        value = payload[spec.name]
        if not _matches_dtype(value, spec.dtype):
            violations.append(
                f"feature {spec.name!r} has type {type(value).__name__}, "
                f"expected {spec.dtype.value}"
            )
    declared = schema.names()
    extras = sorted(set(payload.keys()) - declared)
    return ValidationOutcome(
        passed=len(violations) == 0,
        violations=violations,
        extras=extras,
    )


# ── Render ────────────────────────────────────────────────


def render_validation(outcome: ValidationOutcome) -> str:
    """CLI / Slack-ready text for a validation outcome.

    Visual layout matches `core/promotion_gate.render_verdict`
    (✔ / ✘ markers) so the operator's eye lands consistently
    across the regression suite."""
    lines = ["=== Feature validation ==="]
    if outcome.passed:
        lines.append("✔ PASS")
    else:
        lines.append(f"✘ FAIL ({len(outcome.violations)} violations)")
    for v in outcome.violations:
        lines.append(f"  · {v}")
    if outcome.extras:
        lines.append(f"Extras (not in schema): {', '.join(outcome.extras)}")
    return "\n".join(lines)


__all__ = [
    "AddDefault",
    "Cast",
    "Drop",
    "FeatureDType",
    "FeatureSchema",
    "FeatureSpec",
    "MigrationPlan",
    "MigrationRule",
    "Rename",
    "ValidationOutcome",
    "migrate",
    "render_validation",
    "validate",
]
