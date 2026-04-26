"""Insights routes — read-only views over the new analytics modules.

Each endpoint computes its result fresh from recent closed trades plus
in-process state (drift monitor, shadow ledger, regime memory) the
cycle has been populating. None of these endpoints write — that's the
cycle's job.

Exposed routes:

* ``GET /api/insights/regret``       — hindsight regret summary
* ``GET /api/insights/thesis``       — thesis attribution table
* ``GET /api/insights/drift``        — concept-drift state
* ``GET /api/insights/stress``       — last stress harness verdicts (in-memory)
* ``GET /api/insights/shadow``       — shadow ledger + alert level
* ``GET /api/insights/calibration``  — current calibration curve

The cycle stashes any live state under ``app_state["insights"]`` (a
dict). This lets the routes avoid hard imports of cycle-side modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register(app: FastAPI, app_state: dict[str, Any]) -> None:
    @app.get("/api/insights/regret")
    async def api_regret(limit: int = 200) -> JSONResponse:
        from halal_trader.cli.insights import (
            _load_closed_crypto_trades,
            _trades_to_closed_views,
        )
        from halal_trader.core.regret import aggregate_regret, hindsight_regret

        try:
            trades = await _load_closed_crypto_trades(limit)
            views = _trades_to_closed_views(trades)
            records = [hindsight_regret(v) for v in views]
            summary = aggregate_regret(records)
            return JSONResponse(
                {
                    "n": summary.n,
                    "mean_regret": summary.mean_regret,
                    "median_regret": summary.median_regret,
                    "pct_high_regret": summary.pct_high_regret,
                    "missed_edge_count": summary.missed_edge_count,
                    "tail_loss_count": summary.tail_loss_count,
                    "by_symbol": summary.by_symbol,
                }
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/insights/thesis")
    async def api_thesis(limit: int = 200) -> JSONResponse:
        from halal_trader.cli.insights import (
            _load_closed_crypto_trades,
            _trades_to_tagged,
        )
        from halal_trader.core.thesis import (
            attribute_pnl_by_thesis,
            deprecated_thesis_kill_list,
        )

        try:
            trades = await _load_closed_crypto_trades(limit)
            views = _trades_to_tagged(trades)
            rows = attribute_pnl_by_thesis(views)
            return JSONResponse(
                {
                    "rows": [
                        {
                            "tag": r.tag,
                            "n_trades": r.n_trades,
                            "wins": r.wins,
                            "losses": r.losses,
                            "win_rate": r.win_rate,
                            "avg_pnl_pct": r.avg_pnl_pct,
                        }
                        for r in rows.values()
                    ],
                    "kill_candidates": deprecated_thesis_kill_list(rows),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/insights/drift")
    async def api_drift() -> JSONResponse:
        insights = app_state.get("insights") or {}
        mon = insights.get("drift_monitor")
        if mon is None:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "state": mon.state,
                "n": mon.n,
                "drift_count": mon.drift_count,
                "last_drift_at": mon.last_drift_at,
            }
        )

    @app.get("/api/insights/stress")
    async def api_stress() -> JSONResponse:
        insights = app_state.get("insights") or {}
        verdicts = insights.get("stress_verdicts")
        if not verdicts:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "ts": insights.get("stress_ts"),
                "verdicts": [
                    {
                        "scenario_name": v.scenario_name,
                        "severity": v.severity,
                        "passed": v.passed,
                        "buys": v.buys,
                        "sells": v.sells,
                        "holds": v.holds,
                        "notes": v.notes,
                    }
                    for v in verdicts
                ],
            }
        )

    @app.get("/api/insights/shadow")
    async def api_shadow() -> JSONResponse:
        insights = app_state.get("insights") or {}
        ledger = insights.get("shadow_ledger")
        if ledger is None or ledger.size == 0:
            return JSONResponse({"available": False})
        from halal_trader.core.shadow import (
            divergence_metrics,
            shadow_alert_state,
        )

        metrics = divergence_metrics(ledger.entries)
        level = shadow_alert_state(metrics)
        return JSONResponse(
            {
                "available": True,
                "n": ledger.size,
                "level": level,
                "metrics": (
                    {
                        "n": metrics.n,
                        "mean_diff_pct": metrics.mean_diff_pct,
                        "last_diff_pct": metrics.last_diff_pct,
                        "max_drawdown_diff": metrics.max_drawdown_diff,
                        "paired_t_score": metrics.paired_t_score,
                        "direction": metrics.direction,
                    }
                    if metrics is not None
                    else None
                ),
                "ts": datetime.now(UTC).isoformat(),
            }
        )

    @app.get("/api/insights/calibration")
    async def api_calibration() -> JSONResponse:
        insights = app_state.get("insights") or {}
        curve = insights.get("calibration_curve")
        if curve is None:
            return JSONResponse({"available": False})
        return JSONResponse(
            {
                "available": True,
                "method": curve.method,
                "n_samples": curve.n_samples,
                "anchors": curve.anchors,
            }
        )
