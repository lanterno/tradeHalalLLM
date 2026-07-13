"""Quant trials ledger repository (advisory bookkeeping).

Records every evaluated quant variant — including failures — so the
Deflated Sharpe Ratio gets an honest trial count and every verdict has a
durable home (docs/QUANT_PREDICTION_ROADMAP.md Phase 0). Deliberately NOT
part of the legacy ``Repository`` facade: new code depends on the
narrowest repo it needs (the module docstring rule in ``db/repository.py``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import QuantTrial


def config_hash(config: dict[str, Any] | None) -> str:
    """Short deterministic digest of a config dict (sorted-key JSON, sha256)."""
    canon = json.dumps(config or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


class QuantTrialRepoImpl:
    """Concrete quant-trials repository."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_trial(
        self,
        *,
        name: str,
        kind: str,
        config: dict[str, Any] | None = None,
        window: str = "",
        metrics: dict[str, Any] | None = None,
        criterion: str | None = None,
        verdict: str | None = None,
    ) -> int:
        row = QuantTrial(
            name=name,
            kind=kind,
            config_hash=config_hash(config),
            config=config,
            window=window,
            metrics=metrics,
            criterion=criterion,
            verdict=verdict,
        )
        async with AsyncSession(self._engine) as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            assert row.id is not None
            return row.id

    async def get_trials(
        self, *, name_prefix: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(QuantTrial).order_by(col(QuantTrial.id).desc()).limit(limit)
            if name_prefix:
                statement = statement.where(col(QuantTrial.name).startswith(name_prefix))
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]

    async def count_trials(self, *, name_prefix: str | None = None) -> int:
        """Honest trial count for DSR deflation (includes failures)."""
        async with AsyncSession(self._engine) as session:
            statement = select(func.count()).select_from(QuantTrial)
            if name_prefix:
                statement = statement.where(col(QuantTrial.name).startswith(name_prefix))
            result = await session.exec(statement)
            return int(result.one())
