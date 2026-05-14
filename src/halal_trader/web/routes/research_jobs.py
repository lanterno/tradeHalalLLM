"""Backtest job queue + research run endpoints.

Operators kick off backtests, walk-forward runs, and Monte Carlo
re-samples from the dashboard. Each request becomes a row in
``research_jobs`` and a background asyncio task that updates the row
when it finishes. The UI polls ``GET /api/research/jobs/{id}`` until
the status moves out of ``queued``/``running``.

Heavy backtests typically run < 30s, so a simple in-process task is
fine — no external worker dependency. If we ever need cross-process
parallelism we can swap the runner for an arq / Celery worker without
changing the row schema or the API surface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from halal_trader.core.context import DashboardContext
from halal_trader.db.repos import ResearchJobRepo
from halal_trader.web._serializer import serialize
from halal_trader.web.dependencies import get_ctx
from halal_trader.web.middleware.confirm import require_confirmation

logger = logging.getLogger(__name__)


_VALID_KINDS = ("backtest", "walk_forward", "monte_carlo")


class JobRequest(BaseModel):
    kind: str = Field(pattern="^(backtest|walk_forward|monte_carlo)$")
    name: str | None = Field(default=None, max_length=120)
    params: dict[str, Any] = Field(default_factory=dict)


def register(app: FastAPI) -> None:
    @app.post(
        "/api/research/backtest/run",
        dependencies=[Depends(require_confirmation)],
    )
    async def enqueue_job(
        req: JobRequest, ctx: DashboardContext = Depends(get_ctx)
    ) -> JSONResponse:
        if req.kind not in _VALID_KINDS:
            raise HTTPException(422, f"kind must be one of {_VALID_KINDS}")
        job_id = await ctx.repo.enqueue_research_job(
            kind=req.kind, name=req.name, params=req.params
        )
        asyncio.create_task(
            _run_job(ctx.repo, job_id=job_id, kind=req.kind, params=req.params),
            name=f"research-job-{job_id}",
        )
        return JSONResponse({"job_id": job_id, "status": "queued"})

    @app.get("/api/research/jobs/{job_id}")
    async def get_job(job_id: int, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        data = await ctx.repo.get_research_job(job_id)
        if data is None:
            raise HTTPException(404, f"job {job_id} not found")
        return JSONResponse(serialize(data))

    @app.get("/api/research/jobs")
    async def list_jobs(limit: int = 50, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        rows = await ctx.repo.list_research_jobs(limit=limit)
        return JSONResponse(serialize(rows))

    @app.post(
        "/api/research/jobs/{job_id}/pin",
        dependencies=[Depends(require_confirmation)],
    )
    async def pin_job(job_id: int, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        ok = await ctx.repo.pin_research_job(job_id, True)
        if not ok:
            raise HTTPException(404, f"job {job_id} not found")
        return JSONResponse({"job_id": job_id, "pinned": True})

    @app.delete(
        "/api/research/jobs/{job_id}/pin",
        dependencies=[Depends(require_confirmation)],
    )
    async def unpin_job(job_id: int, ctx: DashboardContext = Depends(get_ctx)) -> JSONResponse:
        ok = await ctx.repo.pin_research_job(job_id, False)
        if not ok:
            raise HTTPException(404, f"job {job_id} not found")
        return JSONResponse({"job_id": job_id, "pinned": False})


# ── In-process worker ─────────────────────────────────────────


async def _run_job(
    repo: ResearchJobRepo, *, job_id: int, kind: str, params: dict[str, Any]
) -> None:
    """Execute one job, recording the outcome on the row.

    The worker only catches exceptions to record them — it never lets
    them propagate up to the asyncio runner where they would become
    "unhandled task exception" warnings on stderr.
    """
    await repo.update_research_job(job_id, status="running")
    try:
        result = await _dispatch(kind, params)
        await repo.update_research_job(job_id, status="ok", result=result)
    except Exception as e:  # noqa: BLE001 — recorded then swallowed
        logger.warning("research job %d failed: %s", job_id, e)
        await repo.update_research_job(job_id, status="error", error=repr(e))


async def _dispatch(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Route the job to the right backtest engine.

    Backtests need real klines; for the dashboard MVP we accept a
    serialised list-of-dicts in ``params['klines']`` (same shape the
    existing backtest engine consumes). Walk-forward + Monte Carlo
    operate on the serialised return list. This keeps the worker
    pure and testable — the caller decides where the data comes from
    (live exchange fetch, replay file, fixture).
    """
    if kind == "backtest":
        return await _run_backtest(params)
    if kind == "walk_forward":
        return await _run_walk_forward(params)
    if kind == "monte_carlo":
        return await _run_monte_carlo(params)
    raise ValueError(f"unknown job kind: {kind}")


async def _run_backtest(params: dict[str, Any]) -> dict[str, Any]:
    from halal_trader.crypto.backtest import BacktestEngine
    from halal_trader.domain.models import Kline

    pair = params.get("pair", "BTCUSDT")
    klines = [Kline(**k) for k in params.get("klines", [])]
    if not klines:
        raise ValueError("backtest requires non-empty 'klines' in params")

    engine = BacktestEngine(
        initial_balance=params.get("initial_balance", 10_000.0),
        slippage_pct=params.get("slippage_pct", 0.0005),
        max_position_pct=params.get("max_position_pct", 0.25),
    )
    result = await engine.run(pair, klines, window_size=params.get("window_size", 100))
    return _result_to_dict(result)


async def _run_walk_forward(params: dict[str, Any]) -> dict[str, Any]:
    from halal_trader.crypto.backtest import BacktestEngine
    from halal_trader.crypto.walkforward import run_walk_forward
    from halal_trader.domain.models import Kline

    pair = params.get("pair", "BTCUSDT")
    klines = [Kline(**k) for k in params.get("klines", [])]
    if not klines:
        raise ValueError("walk_forward requires non-empty 'klines' in params")

    engine = BacktestEngine()

    async def _bt(p, slice_):
        return await engine.run(p, slice_)

    report = await run_walk_forward(
        pair,
        klines,
        backtest_fn=_bt,
        train_size=params.get("train_size", 200),
        test_size=params.get("test_size", 50),
    )
    return {
        "fold_count": report.fold_count,
        "avg_return_pct": report.avg_return_pct,
        "avg_sharpe": report.avg_sharpe,
        "win_rate": report.win_rate,
        "folds": [_result_to_dict(f) for f in report.folds],
    }


async def _run_monte_carlo(params: dict[str, Any]) -> dict[str, Any]:
    from halal_trader.crypto.backtest import SimulatedTrade
    from halal_trader.crypto.walkforward import monte_carlo_resample

    raw_trades = params.get("trades", [])
    trades = [
        SimulatedTrade(
            pair=t.get("pair", "X"),
            side=t.get("side", "buy"),
            quantity=t.get("quantity", 1.0),
            price=t.get("price", 100.0),
            timestamp=t.get("timestamp", 0),
            pnl=t.get("pnl", 0.0),
        )
        for t in raw_trades
    ]
    report = monte_carlo_resample(
        trades,
        initial_balance=params.get("initial_balance", 10_000.0),
        runs=params.get("runs", 500),
        seed=params.get("seed"),
    )
    return {
        "runs": report.runs,
        "final_return_pct_mean": report.final_return_pct_mean,
        "final_return_pct_p5": report.final_return_pct_p5,
        "final_return_pct_p95": report.final_return_pct_p95,
        "max_drawdown_pct_mean": report.max_drawdown_pct_mean,
        "max_drawdown_pct_p95": report.max_drawdown_pct_p95,
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Reduce a BacktestResult dataclass to a JSON-friendly dict."""
    if hasattr(result, "model_dump"):
        return result.model_dump()  # pydantic
    if hasattr(result, "__dict__"):
        out: dict[str, Any] = {}
        for k, v in vars(result).items():
            if k == "trades":
                out[k] = [vars(t) for t in v]
            elif k == "equity_curve":
                out[k] = list(v)
            else:
                out[k] = v
        return out
    return {"value": str(result)}
