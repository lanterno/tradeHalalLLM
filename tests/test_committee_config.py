"""Tests for core/committee_config.py — Round-5 Wave 8.D."""

from __future__ import annotations

import pytest

from halal_trader.core.committee_config import (
    CommitteeConfig,
    DebateMode,
    ModelTier,
    RoleAssignment,
    default_config,
    from_dict,
    render_config,
    with_role_override,
)
from halal_trader.core.llm_committee import AgentRole

# --- RoleAssignment validation ------------------------------------------


def test_assignment_default():
    ra = RoleAssignment(role=AgentRole.BULL)
    assert ra.tier is ModelTier.TIER_BALANCED
    assert ra.weight == 1.0


def test_assignment_negative_weight_rejected():
    with pytest.raises(ValueError):
        RoleAssignment(role=AgentRole.BULL, weight=-0.1)


def test_assignment_excessive_weight_rejected():
    with pytest.raises(ValueError):
        RoleAssignment(role=AgentRole.BULL, weight=20.0)


def test_assignment_long_model_id_rejected():
    with pytest.raises(ValueError):
        RoleAssignment(role=AgentRole.BULL, model_id="x" * 300)


def test_assignment_immutable():
    ra = RoleAssignment(role=AgentRole.BULL)
    with pytest.raises(AttributeError):
        ra.weight = 2.0  # type: ignore[misc]


# --- CommitteeConfig validation -----------------------------------------


def test_default_config_valid():
    cfg = default_config()
    assert cfg.name == "default"
    assert cfg.debate_rounds() == 1


def test_config_unanimity_below_half_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=default_config().role_assignments,
            unanimity_threshold=0.4,
        )


def test_config_unanimity_above_one_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=default_config().role_assignments,
            unanimity_threshold=1.5,
        )


def test_config_zero_quorum_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=default_config().role_assignments,
            require_quorum=0,
        )


def test_config_quorum_exceeds_roles_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=default_config().role_assignments,
            require_quorum=10,
        )


def test_config_duplicate_role_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=(
                RoleAssignment(role=AgentRole.BULL, weight=1.0),
                RoleAssignment(role=AgentRole.BULL, weight=1.0),
                RoleAssignment(role=AgentRole.HALAL_JUDGE, weight=2.0),
            ),
        )


def test_config_halal_weight_floor_pin():
    """Pin: halal-judge weight ≥ 1.5 × avg-of-others."""
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=(
                RoleAssignment(role=AgentRole.BULL, weight=1.0),
                RoleAssignment(role=AgentRole.BEAR, weight=1.0),
                RoleAssignment(role=AgentRole.QUANT, weight=2.0),
                RoleAssignment(role=AgentRole.HALAL_JUDGE, weight=1.0),  # too low
            ),
        )


def test_config_disabled_halal_skips_floor_check():
    """A disabled halal-judge doesn't trigger the floor check."""
    cfg = CommitteeConfig(
        role_assignments=(
            RoleAssignment(role=AgentRole.BULL, weight=1.0),
            RoleAssignment(role=AgentRole.BEAR, weight=1.0),
            RoleAssignment(role=AgentRole.QUANT, weight=2.0),
            RoleAssignment(role=AgentRole.HALAL_JUDGE, weight=0.5, enabled=False),
        ),
    )
    assert cfg.assignment_for(AgentRole.HALAL_JUDGE).weight == 0.5


def test_config_empty_name_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=default_config().role_assignments,
            name="",
        )


def test_config_unsupported_version_rejected():
    with pytest.raises(ValueError):
        CommitteeConfig(
            role_assignments=default_config().role_assignments,
            version=99,
        )


# --- Debate modes -------------------------------------------------------


def test_debate_rounds_mapping():
    cfg_single = CommitteeConfig(
        role_assignments=default_config().role_assignments,
        debate_mode=DebateMode.SINGLE_PASS,
    )
    cfg_three = CommitteeConfig(
        role_assignments=default_config().role_assignments,
        debate_mode=DebateMode.THREE_ROUND,
    )
    assert cfg_single.debate_rounds() == 1
    assert cfg_three.debate_rounds() == 3


# --- Helpers ------------------------------------------------------------


def test_assignment_for_returns_match():
    cfg = default_config()
    ra = cfg.assignment_for(AgentRole.QUANT)
    assert ra is not None
    assert ra.role is AgentRole.QUANT


def test_assignment_for_returns_none_when_missing():
    cfg = default_config()
    assert cfg.assignment_for(AgentRole.OPERATOR_OVERRIDE) is None


def test_enabled_assignments_filters():
    cfg = CommitteeConfig(
        role_assignments=(
            RoleAssignment(role=AgentRole.BULL, weight=1.0),
            RoleAssignment(role=AgentRole.BEAR, weight=1.0, enabled=False),
            RoleAssignment(role=AgentRole.QUANT, weight=1.5),
            RoleAssignment(role=AgentRole.HALAL_JUDGE, weight=2.0),
        ),
    )
    enabled = cfg.enabled_assignments()
    assert len(enabled) == 3
    assert AgentRole.BEAR not in {ra.role for ra in enabled}


# --- with_role_override --------------------------------------------------


def test_with_role_override_changes_only_target():
    cfg = default_config()
    new_cfg = with_role_override(cfg, AgentRole.BULL, weight=1.2)
    bull = new_cfg.assignment_for(AgentRole.BULL)
    bear = new_cfg.assignment_for(AgentRole.BEAR)
    assert bull.weight == 1.2
    assert bear.weight == 1.0


def test_with_role_override_invalid_role_raises():
    cfg = default_config()
    with pytest.raises(ValueError):
        with_role_override(cfg, AgentRole.OPERATOR_OVERRIDE, weight=2.0)


def test_with_role_override_preserves_other_fields():
    cfg = default_config()
    new_cfg = with_role_override(cfg, AgentRole.BULL, model_id="claude-opus-4-7")
    bull = new_cfg.assignment_for(AgentRole.BULL)
    assert bull.tier is ModelTier.TIER_BALANCED  # unchanged
    assert bull.model_id == "claude-opus-4-7"


def test_with_role_override_validates_new_state():
    """If the override creates an invalid state (halal weight floor),
    the call must raise."""
    cfg = default_config()
    with pytest.raises(ValueError):
        with_role_override(cfg, AgentRole.HALAL_JUDGE, weight=0.1)


# --- to_dict / from_dict round-trip --------------------------------------


def test_round_trip_default():
    cfg = default_config()
    payload = cfg.to_dict()
    cfg2 = from_dict(payload)
    assert cfg2.name == cfg.name
    assert cfg2.debate_mode is cfg.debate_mode
    assert cfg2.unanimity_threshold == cfg.unanimity_threshold
    assert len(cfg2.role_assignments) == len(cfg.role_assignments)


def test_round_trip_preserves_role_fields():
    cfg = with_role_override(
        default_config(),
        AgentRole.BULL,
        model_id="claude-sonnet-4-6",
        tier=ModelTier.TIER_FAST,
    )
    cfg2 = from_dict(cfg.to_dict())
    bull = cfg2.assignment_for(AgentRole.BULL)
    assert bull.model_id == "claude-sonnet-4-6"
    assert bull.tier is ModelTier.TIER_FAST


def test_from_dict_missing_version_rejected():
    with pytest.raises(ValueError):
        from_dict({"role_assignments": []})


def test_from_dict_unsupported_version_rejected():
    cfg = default_config()
    payload = cfg.to_dict()
    payload["version"] = 99
    with pytest.raises(ValueError):
        from_dict(payload)


def test_from_dict_empty_assignments_rejected():
    with pytest.raises(ValueError):
        from_dict({"version": 1, "role_assignments": []})


# --- Render --------------------------------------------------------------


def test_render_contains_summary():
    cfg = default_config()
    out = render_config(cfg)
    assert "Committee" in out
    assert "halal_judge" in out


def test_render_redacts_apilike_model_id():
    """Pin: API-key-shaped model_ids are redacted."""
    cfg = with_role_override(default_config(), AgentRole.BULL, model_id="sk-abc-secret-123")
    out = render_config(cfg)
    assert "sk-abc-secret-123" not in out
    assert "[redacted]" in out


def test_render_passes_clean_model_ids():
    cfg = with_role_override(default_config(), AgentRole.BULL, model_id="claude-opus-4-7")
    out = render_config(cfg)
    # claude- pattern is recognised as API-key-shaped, redacted.
    assert "[redacted]" in out


def test_render_passes_short_model_id():
    """Short model_ids without API-key markers pass through."""
    cfg = with_role_override(default_config(), AgentRole.BULL, model_id="gpt-4o")
    out = render_config(cfg)
    assert "gpt-4o" in out


def test_render_marks_disabled_role():
    cfg = CommitteeConfig(
        role_assignments=(
            RoleAssignment(role=AgentRole.BULL, weight=1.0, enabled=False),
            RoleAssignment(role=AgentRole.BEAR, weight=1.0),
            RoleAssignment(role=AgentRole.QUANT, weight=1.5),
            RoleAssignment(role=AgentRole.HALAL_JUDGE, weight=2.0),
        ),
    )
    out = render_config(cfg)
    assert "✗" in out
    assert "✓" in out
