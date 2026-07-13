"""Tests for quant/conformal.py — ACI band maintenance + drift hook."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from halal_trader.quant.calibration import (
    AciState,
    CalibrationArtifact,
    HorizonCalibration,
    load_artifact,
    save_artifact,
)
from halal_trader.quant.conformal import (
    GRID_LEVELS,
    aci_step,
    update_band_conformal,
    z_from_grid,
)

GRID = tuple(1.0 + 0.02 * i for i in range(50))  # monotone 1.00 .. 1.98
NOW = datetime(2026, 7, 13, tzinfo=UTC)


def _artifact(alpha: float | None = None) -> CalibrationArtifact:
    aci = (
        {5: AciState(alpha=alpha, n_obs=0, last_rec_id=0, updated_at="")}
        if alpha is not None
        else None
    )
    return CalibrationArtifact(
        version="zcal-test",
        created_at="2026-07-13T00:00:00+00:00",
        target_coverage=0.8,
        horizons={5: HorizonCalibration(z=1.836, n=3180, target_coverage=0.8, z_grid=GRID)},
        symbols=("AAPL",),
        aci=aci,
    )


class TestPrimitives:
    def test_z_from_grid_interpolates_and_clamps(self):
        # alpha 0.2 → level 0.8 → grid index 30 → 1.60.
        assert z_from_grid(GRID, 0.2) == pytest.approx(1.60)
        # Tiny alpha clamps at the 0.99 level (never extrapolates).
        assert z_from_grid(GRID, 0.0001) == pytest.approx(GRID[-1])
        # Huge alpha clamps at the 0.50 level.
        assert z_from_grid(GRID, 0.9) == pytest.approx(GRID[0])
        assert len(GRID_LEVELS) == len(GRID)

    def test_aci_step_direction_and_equilibrium(self):
        # A breach widens (alpha down), a cover narrows slightly (alpha up).
        assert aci_step(0.2, 0.2, breached=True, gamma=0.01) < 0.2
        assert aci_step(0.2, 0.2, breached=False, gamma=0.01) > 0.2
        # At the target breach rate the expected step is zero.
        alpha = 0.2
        steps = [aci_step(alpha, 0.2, b) for b in (True, False, False, False, False)]
        drift = sum(s - alpha for s in steps)
        assert abs(drift) < 1e-9
        # Clamps hold.
        assert aci_step(0.011, 0.2, breached=True, gamma=1.0) == 0.01

    def test_effective_z_uses_aci_state(self):
        base = _artifact()
        assert base.effective_z(5) == pytest.approx(1.836)  # no state → base z
        adapted = _artifact(alpha=0.1)  # level 0.9 → grid idx 40 → 1.80
        assert adapted.effective_z(5) == pytest.approx(1.80)

    def test_artifact_round_trip_with_grid_and_aci(self, tmp_path):
        path = tmp_path / "band_calibration.json"
        art = _artifact(alpha=0.15)
        save_artifact(art, path)
        loaded = load_artifact(path)
        assert loaded.horizons[5].z_grid == GRID
        assert loaded.aci[5].alpha == pytest.approx(0.15)
        assert loaded.effective_z(5) == art.effective_z(5)


class _FakeRepo:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    async def get_recent_recommendations(self, limit: int = 200):
        return list(self._rows)


def _rec(rec_id: int, date: str, covered_flags: list[bool | None]) -> dict[str, Any]:
    return {
        "id": rec_id,
        "date": date,
        "candidates": {
            f"SYM{i}": {"outcome": {"fwd_return_5d": 1.0, "band_covered_5d": c}}
            for i, c in enumerate(covered_flags)
        },
    }


class TestUpdateBandConformal:
    @pytest.mark.asyncio
    async def test_breach_heavy_outcomes_widen_the_band(self, tmp_path):
        path = tmp_path / "cal.json"
        save_artifact(_artifact(), path)
        # 3 matured recs, every candidate breached → alpha must fall,
        # effective z must rise above the base grid level.
        repo = _FakeRepo([_rec(i, f"2026-07-0{i}", [False] * 10) for i in (1, 2, 3)])
        res = await update_band_conformal(repo, path=path, now=NOW)
        assert res["updated"] is True
        assert res["consumed"] == 30
        assert res["alpha"] < 0.2
        assert res["effective_z"] > z_from_grid(GRID, 0.2)

    @pytest.mark.asyncio
    async def test_idempotent_consumption(self, tmp_path):
        path = tmp_path / "cal.json"
        save_artifact(_artifact(), path)
        repo = _FakeRepo([_rec(1, "2026-07-01", [True] * 8)])
        first = await update_band_conformal(repo, path=path, now=NOW)
        second = await update_band_conformal(repo, path=path, now=NOW)
        assert first["consumed"] == 8
        assert second["consumed"] == 0  # last_rec_id fence
        assert second["alpha"] == first["alpha"]

    @pytest.mark.asyncio
    async def test_unmatured_recs_are_deferred_whole(self, tmp_path):
        path = tmp_path / "cal.json"
        save_artifact(_artifact(), path)
        repo = _FakeRepo([_rec(1, "2026-07-12", [True] * 8)])  # 1 day old
        res = await update_band_conformal(repo, path=path, now=NOW)
        assert res["consumed"] == 0

    @pytest.mark.asyncio
    async def test_drift_fires_on_excessive_breaches(self, tmp_path):
        path = tmp_path / "cal.json"
        save_artifact(_artifact(), path)
        # 60 trailing outcomes with a 50% breach rate vs the 20% target.
        rows = [
            _rec(i, f"2026-06-{(i % 28) + 1:02d}", [True] * 5 + [False] * 5) for i in range(1, 7)
        ]
        repo = _FakeRepo(rows)
        res = await update_band_conformal(repo, path=path, now=NOW)
        assert res["trailing_n"] >= 30
        assert res["drift"] is True
        assert res["drift_p"] < 0.05

    @pytest.mark.asyncio
    async def test_no_drift_at_target_rate(self, tmp_path):
        path = tmp_path / "cal.json"
        save_artifact(_artifact(), path)
        # 20% breach rate == target → calibrated, no alarm.
        rows = [
            _rec(i, f"2026-06-{(i % 28) + 1:02d}", [True] * 8 + [False] * 2) for i in range(1, 7)
        ]
        res = await update_band_conformal(_FakeRepo(rows), path=path, now=NOW)
        assert res["drift"] is False

    @pytest.mark.asyncio
    async def test_missing_artifact_is_a_noop(self, tmp_path):
        res = await update_band_conformal(_FakeRepo([]), path=tmp_path / "nope.json", now=NOW)
        assert res["updated"] is False

    @pytest.mark.asyncio
    async def test_pre_grid_artifact_asks_for_recalibration(self, tmp_path):
        path = tmp_path / "cal.json"
        old = CalibrationArtifact(
            version="zcal-old",
            created_at="",
            target_coverage=0.8,
            horizons={5: HorizonCalibration(z=1.8, n=100, target_coverage=0.8)},
            symbols=(),
        )
        save_artifact(old, path)
        res = await update_band_conformal(_FakeRepo([]), path=path, now=NOW)
        assert res["updated"] is False
        assert "z_grid" in res["reason"]
