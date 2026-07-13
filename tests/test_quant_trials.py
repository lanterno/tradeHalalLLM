"""Tests for the quant trials ledger (db/repos/quant_trials.py)."""

from __future__ import annotations

import pytest

from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl, config_hash


def test_config_hash_deterministic_and_order_insensitive():
    a = config_hash({"horizon": 5, "days": 400})
    b = config_hash({"days": 400, "horizon": 5})
    assert a == b
    assert len(a) == 12
    assert config_hash({"days": 401, "horizon": 5}) != a
    assert config_hash(None) == config_hash({})


@pytest.mark.asyncio
async def test_record_list_and_count(engine):
    repo = QuantTrialRepoImpl(engine)
    tid = await repo.record_trial(
        name="levels.swing_zones.touch_hold",
        kind="level_family",
        config={"horizon": 5, "days": 400},
        window="400d x 20sym daily (single window)",
        metrics={"hold_rate": 0.39, "placebo_hold_rate": 0.41, "uplift": -0.018},
        criterion="beats placebo on disjoint OOS",
        verdict="fail",
    )
    assert tid > 0
    await repo.record_trial(
        name="bands.zcal.pooled_walkforward",
        kind="band_calibration",
        config={"days": 400, "coverage": 0.8},
        window="400d x 20sym daily",
        metrics={"z_1d": 1.648, "z_5d": 1.836},
        verdict="pass",
    )

    rows = await repo.get_trials()
    assert len(rows) == 2
    assert rows[0]["name"] == "bands.zcal.pooled_walkforward"  # newest first
    assert rows[1]["verdict"] == "fail"
    assert rows[1]["metrics"]["uplift"] == pytest.approx(-0.018)
    assert rows[1]["config_hash"] == config_hash({"horizon": 5, "days": 400})

    levels_only = await repo.get_trials(name_prefix="levels.")
    assert len(levels_only) == 1
    assert await repo.count_trials() == 2
    assert await repo.count_trials(name_prefix="levels.") == 1
    assert await repo.count_trials(name_prefix="nope.") == 0
