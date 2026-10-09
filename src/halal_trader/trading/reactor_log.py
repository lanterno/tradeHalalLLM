"""The news reactor's decision record (the ``reactor_decisions`` table).

Every scored catalyst the reactor acts on leaves one row: the shadow entry it
would have made, the order it placed, or why it did neither. Writing it never
raises: a lost row must not cost the reactor its loop.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

PLACED = frozenset({"filled", "partially_filled", "submitted", "accepted", "pending"})


async def record(
    engine: AsyncEngine | None,
    event: Any,
    result: dict[str, Any] | None,
    note: str,
) -> None:
    """One row for ``event`` and what came of it (``result`` None: observed only)."""
    if engine is None:
        return
    cls = event.classification
    status = str((result or {}).get("status") or "observed")

    def num(key: str) -> float | None:
        value = (result or {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO reactor_decisions (decided_at, symbol, score, tag, headline, "
                    "status, reason, placed, price, intraday_change, quantity, stop_loss) "
                    "VALUES (:at, :s, :score, :tag, :h, :st, :r, :placed, :p, :ch, :q, :sl)"
                ),
                {
                    "at": datetime.now(UTC),
                    "s": event.symbol,
                    "score": float(cls.score),
                    "tag": str(cls.tag) if cls.tag is not None else None,
                    "h": str(event.title)[:500],
                    "st": status,
                    "r": (result or {}).get("reason") or (note if result is None else None),
                    "placed": status in PLACED,
                    "p": num("price"),
                    "ch": num("intraday_change"),
                    "q": num("quantity"),
                    "sl": num("stop_loss"),
                },
            )
    except Exception as exc:  # noqa: BLE001 -- the record must not break the reactor
        logger.warning("reactor decision for %s not recorded: %r", event.symbol, exc)
