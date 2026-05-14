"""Feature flag rollout engine.

Auxiliary primitive complementing Wave 10.F edition gating.
Wave 10.F's `feature_gate.py` answers "is feature X available in
this build's edition + tier?"; this module answers a different
question: "is feature X enabled for THIS user RIGHT NOW during
the gradual rollout?". The two layers compose — edition gating
is build-time / commercial; feature flags are runtime / staged.

Picked a focused rollout engine over scattered `if random() < 0.1:`
checks because (a) gradual rollout needs deterministic per-user
evaluation — a user assigned to the 10% test bucket should stay in
that bucket across reloads, otherwise they'd see flicker (feature
on / feature off / on / off) which is the worst possible UX during
A/B testing; SHA-256 hashing of (flag_id, user_id) gives
deterministic bucket assignment, (b) the flag registry needs to be
inspectable at runtime so operators can answer "what's currently
rolling out?" without grepping logs — a frozen registry queryable
by `all_flags()` is the operator's at-a-glance view, (c) cohort
allowlists are explicit (operator-curated user IDs) rather than
hash-based — beta-tester cohorts shouldn't drift if a user_id
hashes into a different bucket after a salt rotation.

Pinned semantics:
- **Closed-set RolloutKind enum.** OFF (disabled for all), ON
  (enabled for all), PERCENTAGE (deterministic hash-based bucket
  rollout), COHORT_ALLOWLIST (explicit user_id list). Adding a
  kind is a code review change.
- **Per-user evaluation is deterministic.** Same (flag_id, user_id)
  always returns the same enabled/disabled answer for a given
  rollout config — pinned via test that flicker is impossible.
- **Percentage rollout uses SHA-256 of (flag_id, user_id).**
  Hashing into 0-99; user is enabled if `hash % 100 < percentage`.
  Salt is the flag_id so two flags rolling out at 50% don't
  enable the same user-half (correlation-free rollouts).
- **Cohort allowlist is explicit.** Operator-curated frozenset
  of user IDs; not hash-based. A user added then removed from
  the cohort flips off cleanly.
- **Render output never includes user_ids — only counts.**
  Mirrors no-secret patterns of upstream waves.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class RolloutKind(str, Enum):
    """Closed-set rollout strategies.

    Pinned string values for JSON / DB stability.
    """

    OFF = "off"
    ON = "on"
    PERCENTAGE = "percentage"
    COHORT_ALLOWLIST = "cohort_allowlist"


@dataclass(frozen=True)
class FeatureFlag:
    """One feature flag's rollout configuration.

    `kind` determines which fields are meaningful:
    - OFF / ON: ignore percentage + cohort_user_ids
    - PERCENTAGE: percentage in [0, 100]; cohort_user_ids ignored
    - COHORT_ALLOWLIST: cohort_user_ids non-empty; percentage ignored

    Validation enforces consistency at construction.
    """

    flag_id: str
    description: str
    kind: RolloutKind
    percentage: int = 0
    cohort_user_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.flag_id or not self.flag_id.strip():
            raise ValueError("flag_id must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        if not 0 <= self.percentage <= 100:
            raise ValueError(f"percentage {self.percentage} must be in [0, 100]")
        if self.kind is RolloutKind.PERCENTAGE:
            # Percentage rollout requires an actual percentage value
            # (0 = OFF; 100 = ON; meaningful range is 1-99)
            if self.percentage == 0:
                raise ValueError("PERCENTAGE rollout with 0% should use OFF instead")
            if self.percentage == 100:
                raise ValueError("PERCENTAGE rollout with 100% should use ON instead")
        if self.kind is RolloutKind.COHORT_ALLOWLIST:
            if not self.cohort_user_ids:
                raise ValueError("COHORT_ALLOWLIST requires non-empty cohort_user_ids")
        if self.kind is not RolloutKind.COHORT_ALLOWLIST and self.cohort_user_ids:
            raise ValueError(f"{self.kind.value} kind must not have cohort_user_ids")


def _bucket_for(flag_id: str, user_id: str) -> int:
    """Deterministic 0-99 bucket assignment for (flag_id, user_id).

    Salts the hash with the flag_id so different flags rolling out
    at the same percentage don't enable the same user-half (the
    correlation-free pin).
    """

    digest = hashlib.sha256(f"{flag_id}:{user_id}".encode("utf-8")).digest()
    # Take first 4 bytes as a 32-bit unsigned int, mod 100
    bucket_int = int.from_bytes(digest[:4], "big")
    return bucket_int % 100


def is_enabled(flag: FeatureFlag, *, user_id: str) -> bool:
    """Evaluate the flag for a user.

    Pure: same (flag, user_id) always returns the same answer.
    """

    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")

    if flag.kind is RolloutKind.OFF:
        return False
    if flag.kind is RolloutKind.ON:
        return True
    if flag.kind is RolloutKind.PERCENTAGE:
        return _bucket_for(flag.flag_id, user_id) < flag.percentage
    if flag.kind is RolloutKind.COHORT_ALLOWLIST:
        return user_id in flag.cohort_user_ids
    raise ValueError(f"unknown RolloutKind {flag.kind!r}")


@dataclass(frozen=True)
class FlagRegistry:
    """Frozen registry of all known feature flags.

    Operators build the registry at boot from config; the registry
    is queryable but not mutable (a runtime "let's flip flag X to
    100%" requires a config reload + new registry, which is
    intentional — flag changes are deploy-event-worthy).
    """

    flags: frozenset[FeatureFlag]

    def __post_init__(self) -> None:
        # No duplicate flag_ids
        flag_ids = [f.flag_id for f in self.flags]
        if len(set(flag_ids)) != len(flag_ids):
            raise ValueError("duplicate flag_id in registry")


def lookup(registry: FlagRegistry, flag_id: str) -> FeatureFlag:
    """Look up a flag by id; raises KeyError if not found."""

    for flag in registry.flags:
        if flag.flag_id == flag_id:
            return flag
    raise KeyError(f"flag {flag_id!r} not in registry")


def is_enabled_in(registry: FlagRegistry, flag_id: str, *, user_id: str) -> bool:
    """Convenience: lookup + evaluate."""

    flag = lookup(registry, flag_id)
    return is_enabled(flag, user_id=user_id)


def all_flags(
    registry: FlagRegistry,
) -> tuple[FeatureFlag, ...]:
    """Return all flags sorted by flag_id (deterministic display order)."""

    return tuple(sorted(registry.flags, key=lambda f: f.flag_id))


def enabled_count(flag: FeatureFlag, *, sample_user_ids: Iterable[str]) -> int:
    """Count how many users in the sample have the flag enabled.

    Operators use this to verify "did the 10% rollout actually
    enable ~10% of our active users?" without leaking individual
    user IDs to the dashboard.
    """

    return sum(1 for uid in sample_user_ids if is_enabled(flag, user_id=uid))


_KIND_EMOJI: dict[RolloutKind, str] = {
    RolloutKind.OFF: "⚫",
    RolloutKind.ON: "✅",
    RolloutKind.PERCENTAGE: "📊",
    RolloutKind.COHORT_ALLOWLIST: "👥",
}


def render_flag(flag: FeatureFlag) -> str:
    """Format a flag for ops display.

    No-secret-leak: never includes individual cohort user_ids — only
    the cohort SIZE. Operators can audit individual cohort membership
    via a separate ops-tool that reads the registry directly.
    """

    emoji = _KIND_EMOJI[flag.kind]
    detail = ""
    if flag.kind is RolloutKind.PERCENTAGE:
        detail = f" — {flag.percentage}%"
    elif flag.kind is RolloutKind.COHORT_ALLOWLIST:
        detail = f" — cohort of {len(flag.cohort_user_ids)} users"
    return f"{emoji} {flag.flag_id}{detail}\n  {flag.description}"


def render_registry(registry: FlagRegistry) -> str:
    """Format the full registry for ops display.

    Lists flags sorted by flag_id with a count summary.
    """

    flags = all_flags(registry)
    counts: dict[RolloutKind, int] = {kind: 0 for kind in RolloutKind}
    for f in flags:
        counts[f.kind] += 1
    summary = (
        f"📋 Feature flag registry — {len(flags)} flags total\n"
        f"  ⚫ off: {counts[RolloutKind.OFF]} | "
        f"✅ on: {counts[RolloutKind.ON]} | "
        f"📊 percentage: {counts[RolloutKind.PERCENTAGE]} | "
        f"👥 cohort: {counts[RolloutKind.COHORT_ALLOWLIST]}"
    )
    flag_lines = [render_flag(f) for f in flags]
    return "\n".join([summary, "", *flag_lines])


__all__ = [
    "FeatureFlag",
    "FlagRegistry",
    "RolloutKind",
    "all_flags",
    "enabled_count",
    "is_enabled",
    "is_enabled_in",
    "lookup",
    "render_flag",
    "render_registry",
]
