"""Operator-tunable committee configuration — Round-5 Wave 8.D.

The default committee aggregator (`core/llm_committee.py`) hard-wires
role weights, debate rounds, and unanimity thresholds inside
`CommitteePolicy`. In production an operator should be able to:

- Pin a specific model per role (Sonnet for Bull, Opus for Halal-judge).
- Set a debate round count (1 = single-pass, ≥ 2 = adversarial loop).
- Pick a unanimity threshold (e.g. 5/6 vs simple plurality).
- Save the config to JSON, reload on restart, validate before applying.

This module ships the **config dataclass + (de)serialiser + validator**.
The persistence backend (file / Postgres) lives elsewhere; this layer
is pure-Python so unit-tests can exercise full lifecycle.

Pinned semantics:

- **Closed-set ModelTier** — TIER_FAST / TIER_BALANCED / TIER_DEEP.
  Each role gets one tier. Operator can also pin an explicit
  `model_id` string for fine-grained control.
- **Closed-set DebateMode** — SINGLE_PASS / TWO_ROUND / THREE_ROUND.
  Beyond three rounds is rejected (latency creep + degenerate
  arguments).
- **Unanimity threshold is a fraction in [0.5, 1.0]** — below 0.5 isn't
  a meaningful threshold; the simple plurality already wins.
- **Halal-judge weight floor**: in any saved config, the halal-judge
  must have weight ≥ 1.5× the average of other roles. Pinned because
  Sharia compliance is non-negotiable.
- **Roundtrip JSON is canonical** — `to_dict` / `from_dict` are
  inverses; the JSON schema is stable across versions (a `version`
  field is preserved).
- **No-secret-leak pin** on render — `model_id` strings are masked
  if they look like API keys.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from halal_trader.core.llm_committee import AgentRole

_CONFIG_VERSION: int = 1


class ModelTier(str, Enum):
    """Closed-set model-tier ladder."""

    TIER_FAST = "tier_fast"
    TIER_BALANCED = "tier_balanced"
    TIER_DEEP = "tier_deep"


class DebateMode(str, Enum):
    """Closed-set debate mode ladder."""

    SINGLE_PASS = "single_pass"
    TWO_ROUND = "two_round"
    THREE_ROUND = "three_round"


_DEBATE_ROUND_COUNT: dict[DebateMode, int] = {
    DebateMode.SINGLE_PASS: 1,
    DebateMode.TWO_ROUND: 2,
    DebateMode.THREE_ROUND: 3,
}


@dataclass(frozen=True)
class RoleAssignment:
    """Assigns a model + tier to a single committee role."""

    role: AgentRole
    tier: ModelTier = ModelTier.TIER_BALANCED
    model_id: str = ""
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        if self.weight > 10:
            raise ValueError("weight > 10 is suspicious; reject")
        if self.model_id and len(self.model_id) > 200:
            raise ValueError("model_id must be ≤ 200 chars")


@dataclass(frozen=True)
class CommitteeConfig:
    """The full operator-tunable committee config."""

    role_assignments: tuple[RoleAssignment, ...]
    debate_mode: DebateMode = DebateMode.SINGLE_PASS
    unanimity_threshold: float = 0.6
    require_quorum: int = 3
    halal_judge_veto_on_skip: bool = True
    name: str = "default"
    version: int = _CONFIG_VERSION

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.version != _CONFIG_VERSION:
            raise ValueError(
                f"unsupported config version {self.version} (expected {_CONFIG_VERSION})"
            )
        if not 0.5 <= self.unanimity_threshold <= 1.0:
            raise ValueError("unanimity_threshold must be in [0.5, 1.0]")
        if self.require_quorum <= 0:
            raise ValueError("require_quorum must be positive")
        # Roles must be unique.
        seen: set[AgentRole] = set()
        for ra in self.role_assignments:
            if ra.role in seen:
                raise ValueError(f"duplicate role {ra.role.value}")
            seen.add(ra.role)
        # Halal-judge weight pin.
        halal = next(
            (ra for ra in self.role_assignments if ra.role is AgentRole.HALAL_JUDGE),
            None,
        )
        others_enabled = [
            ra
            for ra in self.role_assignments
            if ra.role is not AgentRole.HALAL_JUDGE and ra.enabled
        ]
        if halal is not None and halal.enabled and others_enabled:
            avg_other = sum(ra.weight for ra in others_enabled) / len(others_enabled)
            if halal.weight < avg_other * 1.5 - 1e-9:
                raise ValueError("halal-judge weight must be ≥ 1.5× the average of other roles")
        if self.require_quorum > len(self.role_assignments):
            raise ValueError("require_quorum exceeds number of roles")

    def debate_rounds(self) -> int:
        return _DEBATE_ROUND_COUNT[self.debate_mode]

    def assignment_for(self, role: AgentRole) -> RoleAssignment | None:
        for ra in self.role_assignments:
            if ra.role is role:
                return ra
        return None

    def enabled_assignments(self) -> tuple[RoleAssignment, ...]:
        return tuple(ra for ra in self.role_assignments if ra.enabled)

    def to_dict(self) -> dict[str, Any]:
        """Canonical dict for JSON persistence."""
        return {
            "version": self.version,
            "name": self.name,
            "debate_mode": self.debate_mode.value,
            "unanimity_threshold": self.unanimity_threshold,
            "require_quorum": self.require_quorum,
            "halal_judge_veto_on_skip": self.halal_judge_veto_on_skip,
            "role_assignments": [
                {
                    "role": ra.role.value,
                    "tier": ra.tier.value,
                    "model_id": ra.model_id,
                    "weight": ra.weight,
                    "enabled": ra.enabled,
                }
                for ra in self.role_assignments
            ],
        }


def from_dict(payload: Mapping[str, Any]) -> CommitteeConfig:
    """Round-trip inverse of `to_dict`."""
    if "version" not in payload:
        raise ValueError("payload missing 'version' field")
    version = int(payload["version"])
    if version != _CONFIG_VERSION:
        raise ValueError(f"unsupported config version {version}")
    if "role_assignments" not in payload or not payload["role_assignments"]:
        raise ValueError("payload missing role_assignments")
    role_assignments = tuple(
        RoleAssignment(
            role=AgentRole(r["role"]),
            tier=ModelTier(r.get("tier", "tier_balanced")),
            model_id=r.get("model_id", ""),
            weight=float(r.get("weight", 1.0)),
            enabled=bool(r.get("enabled", True)),
        )
        for r in payload["role_assignments"]
    )
    return CommitteeConfig(
        role_assignments=role_assignments,
        debate_mode=DebateMode(payload.get("debate_mode", "single_pass")),
        unanimity_threshold=float(payload.get("unanimity_threshold", 0.6)),
        require_quorum=int(payload.get("require_quorum", 3)),
        halal_judge_veto_on_skip=bool(payload.get("halal_judge_veto_on_skip", True)),
        name=payload.get("name", "default"),
        version=version,
    )


def default_config() -> CommitteeConfig:
    """A reasonable starting config: 4 roles, weight floor satisfied."""
    return CommitteeConfig(
        role_assignments=(
            RoleAssignment(
                role=AgentRole.BULL,
                tier=ModelTier.TIER_BALANCED,
                weight=1.0,
            ),
            RoleAssignment(
                role=AgentRole.BEAR,
                tier=ModelTier.TIER_BALANCED,
                weight=1.0,
            ),
            RoleAssignment(
                role=AgentRole.QUANT,
                tier=ModelTier.TIER_BALANCED,
                weight=1.5,
            ),
            RoleAssignment(
                role=AgentRole.HALAL_JUDGE,
                tier=ModelTier.TIER_DEEP,
                weight=2.0,
            ),
        ),
        debate_mode=DebateMode.SINGLE_PASS,
        unanimity_threshold=0.6,
        require_quorum=3,
        halal_judge_veto_on_skip=True,
        name="default",
    )


def with_role_override(
    config: CommitteeConfig, role: AgentRole, **overrides: Any
) -> CommitteeConfig:
    """Return a new config with the given role's assignment overridden.

    Convenience for operators tweaking one role at a time without
    rebuilding the full assignment list.
    """
    new_assignments = []
    found = False
    for ra in config.role_assignments:
        if ra.role is role:
            new_assignments.append(replace(ra, **overrides))
            found = True
        else:
            new_assignments.append(ra)
    if not found:
        raise ValueError(f"role {role.value} not in config")
    return replace(config, role_assignments=tuple(new_assignments))


_API_KEY_RE = re.compile(r"(?:sk-|claude-|api[-_])\S{6,}")


def _mask_model_id(model_id: str) -> str:
    if not model_id:
        return ""
    if _API_KEY_RE.search(model_id):
        return "[redacted]"
    return model_id


def render_config(config: CommitteeConfig) -> str:
    """Operator-readable summary of the config."""
    head = (
        f"⚙️ Committee[{config.name} v{config.version}] "
        f"debate={config.debate_mode.value} "
        f"unanimity={config.unanimity_threshold:.2f} "
        f"quorum={config.require_quorum} "
        f"halal_veto={config.halal_judge_veto_on_skip}"
    )
    lines = [head]
    for ra in config.role_assignments:
        marker = "✓" if ra.enabled else "✗"
        model_str = f" model={_mask_model_id(ra.model_id)}" if ra.model_id else ""
        lines.append(
            f"  {marker} {ra.role.value}: tier={ra.tier.value} weight={ra.weight:.2f}{model_str}"
        )
    return "\n".join(lines)
