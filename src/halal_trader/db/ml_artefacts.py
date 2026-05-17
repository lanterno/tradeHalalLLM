"""Helpers for the versioned ML model blob store.

Wave K — replaces ``models/*.pkl`` with the ``ml_artefacts`` table.
Each (name, version) pair is one row; the loader picks the highest
version for a given name; the retrainer inserts version+1.

JSON-shaped artefacts (slippage model, calibration curve) live in
the ``payload_json`` JSONB column; pickled sklearn / xgboost models
live in ``payload_bytes`` BYTEA. The HuggingFace cache (~GB Chronos
weights) intentionally stays on disk — the table is for state we
own, not third-party model weights.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import MlArtefact

logger = logging.getLogger(__name__)


async def _next_version(*, engine: AsyncEngine, name: str) -> int:
    """Return the next version number for ``name`` (1 for first row)."""
    async with AsyncSession(engine) as s:
        stmt = (
            select(MlArtefact.version)
            .where(MlArtefact.name == name)
            .order_by(col(MlArtefact.version).desc())
            .limit(1)
        )
        result = await s.exec(stmt)
        latest = result.first()
        return (latest or 0) + 1


async def save_artefact(
    *,
    engine: AsyncEngine,
    name: str,
    payload_json: dict[str, Any] | None = None,
    payload_bytes: bytes | None = None,
    version: int | None = None,
    sklearn_version: str = "",
    feature_hash: str = "",
) -> int:
    """Insert one new artefact version. Returns the new row id.

    Exactly one of ``payload_json`` / ``payload_bytes`` must be set.
    """
    if (payload_json is None) == (payload_bytes is None):
        raise ValueError("must pass exactly one of payload_json / payload_bytes")

    if version is None:
        version = await _next_version(engine=engine, name=name)

    fmt = "json" if payload_json is not None else "pickle"
    async with AsyncSession(engine, expire_on_commit=False) as s:
        row = MlArtefact(
            name=name,
            version=version,
            payload_format=fmt,
            payload_json=payload_json,
            payload_bytes=payload_bytes,
            sklearn_version=sklearn_version,
            feature_hash=feature_hash,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        assert row.id is not None
        return row.id


async def load_artefact(*, engine: AsyncEngine, name: str) -> dict[str, Any] | None:
    """Return the *latest* version of ``name`` as a JSON dict, or None."""
    async with AsyncSession(engine) as s:
        stmt = (
            select(MlArtefact)
            .where(MlArtefact.name == name)
            .order_by(col(MlArtefact.version).desc())
            .limit(1)
        )
        result = await s.exec(stmt)
        row = result.first()
        if row is None:
            return None
        if row.payload_format == "json":
            return dict(row.payload_json or {})
        if row.payload_format == "pickle" and row.payload_bytes is not None:
            try:
                obj = pickle.loads(row.payload_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ml_artefact %s pickle.loads failed: %s", name, exc)
                return None
            return {"_pickle": obj}
        return None


async def list_versions(
    *, engine: AsyncEngine, name: str | None = None
) -> list[dict[str, Any]]:
    """All versions of ``name`` (newest first), or every artefact when None."""
    async with AsyncSession(engine) as s:
        stmt = select(MlArtefact).order_by(col(MlArtefact.created_at).desc())
        if name is not None:
            stmt = stmt.where(MlArtefact.name == name)
        result = await s.exec(stmt)
        return [
            {
                "id": r.id,
                "name": r.name,
                "version": r.version,
                "payload_format": r.payload_format,
                "sklearn_version": r.sklearn_version,
                "feature_hash": r.feature_hash,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in result.all()
        ]


def pickle_dumps(obj: Any) -> bytes:
    """Wrap ``pickle.dumps`` with the protocol the bot uses (HIGHEST)."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
