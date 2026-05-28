"""target_weight — hysteresis, banding, config guard (REARCHITECTURE B.5)."""

from __future__ import annotations

import pytest

from halabot.belief.schema import BeliefState, ComplianceVerdict, Direction
from halabot.policy.sizing import PolicyConfig, target_weight
from halabot.risk.engine import RiskState

CFG = PolicyConfig(conviction_entry_band=0.60, conviction_exit_band=0.45, max_weight_per_asset=0.20)
RISK = RiskState()


def _belief(conviction: float, direction=Direction.LONG_BIAS) -> BeliefState:
    return BeliefState(
        asset="NVDA",
        direction=direction,
        conviction=conviction,
        halal=ComplianceVerdict("NVDA", "halal"),
    )


def test_not_long_is_zero():
    assert target_weight(_belief(0.9, Direction.NEUTRAL), RISK, held=False, cfg=CFG) == 0.0


def test_below_entry_band_not_held_is_zero():
    assert target_weight(_belief(0.50), RISK, held=False, cfg=CFG) == 0.0


def test_above_entry_band_sizes_up():
    w = target_weight(_belief(0.80), RISK, held=False, cfg=CFG)
    assert 0.0 < w <= CFG.max_weight_per_asset


def test_hysteresis_holds_between_bands_only_when_held():
    mid = _belief(0.50)  # between exit (0.45) and entry (0.60)
    assert target_weight(mid, RISK, held=True, cfg=CFG) > 0.0   # stays held
    assert target_weight(mid, RISK, held=False, cfg=CFG) == 0.0  # but won't open


def test_weight_capped_at_max():
    assert target_weight(_belief(1.0), RISK, held=False, cfg=CFG) == CFG.max_weight_per_asset


def test_multipliers_scale_down():
    risk = RiskState(_correlation_mult={"NVDA": 0.5})
    full = target_weight(_belief(0.80), RISK, held=False, cfg=CFG)
    halved = target_weight(_belief(0.80), risk, held=False, cfg=CFG)
    assert halved == pytest.approx(full * 0.5)


def test_invalid_band_config_raises():
    with pytest.raises(ValueError):
        PolicyConfig(conviction_entry_band=0.4, conviction_exit_band=0.5)  # exit >= entry
    with pytest.raises(ValueError):
        PolicyConfig(conviction_entry_band=1.0, conviction_exit_band=0.5)  # entry not < 1
