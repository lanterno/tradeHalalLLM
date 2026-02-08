"""Data access layer for the SQLite database."""

import json
from datetime import date
from typing import Any

import aiosqlite


class Repository:
    """Thin wrapper around aiosqlite for trade/PnL/halal operations."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    # ── Trades ──────────────────────────────────────────────────

    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_id: str | None = None,
        status: str = "pending",
        llm_reasoning: str | None = None,
    ) -> int:
        cursor = await self._db.execute(
            """INSERT INTO trades (symbol, side, quantity, price, order_id, status, llm_reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, side, quantity, price, order_id, status, llm_reasoning),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def update_trade_status(
        self, trade_id: int, status: str, price: float | None = None
    ) -> None:
        if price is not None:
            await self._db.execute(
                "UPDATE trades SET status = ?, price = ? WHERE id = ?",
                (status, price, trade_id),
            )
        else:
            await self._db.execute(
                "UPDATE trades SET status = ? WHERE id = ?",
                (status, trade_id),
            )
        await self._db.commit()

    async def get_today_trades(self) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE date(timestamp) = ? ORDER BY timestamp DESC",
            (today,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Daily P&L ───────────────────────────────────────────────

    async def start_day(self, starting_equity: float) -> None:
        today = date.today().isoformat()
        await self._db.execute(
            """INSERT OR IGNORE INTO daily_pnl (date, starting_equity) VALUES (?, ?)""",
            (today, starting_equity),
        )
        await self._db.commit()

    async def end_day(self, ending_equity: float, realized_pnl: float, trades_count: int) -> None:
        today = date.today().isoformat()
        row = await (
            await self._db.execute("SELECT starting_equity FROM daily_pnl WHERE date = ?", (today,))
        ).fetchone()
        starting = row[0] if row else ending_equity
        return_pct = (ending_equity - starting) / starting if starting else 0
        await self._db.execute(
            """UPDATE daily_pnl
               SET ending_equity = ?, realized_pnl = ?, return_pct = ?, trades_count = ?
               WHERE date = ?""",
            (ending_equity, realized_pnl, return_pct, trades_count, today),
        )
        await self._db.commit()

    async def get_pnl_history(self, limit: int = 30) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Halal Cache ─────────────────────────────────────────────

    async def cache_halal_status(
        self, symbol: str, compliance: str, detail: str | None = None
    ) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO halal_cache (symbol, compliance, detail, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (symbol, compliance, detail),
        )
        await self._db.commit()

    async def get_halal_status(self, symbol: str) -> str | None:
        row = await (
            await self._db.execute("SELECT compliance FROM halal_cache WHERE symbol = ?", (symbol,))
        ).fetchone()
        return row[0] if row else None

    async def get_halal_symbols(self) -> list[str]:
        cursor = await self._db.execute("SELECT symbol FROM halal_cache WHERE compliance = 'halal'")
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        row = await (
            await self._db.execute(
                """SELECT COUNT(*) FROM halal_cache
                   WHERE updated_at > datetime('now', ? || ' hours')""",
                (f"-{max_age_hours}",),
            )
        ).fetchone()
        return (row[0] or 0) > 0

    # ── LLM Decisions ───────────────────────────────────────────

    async def record_decision(
        self,
        provider: str,
        model: str,
        prompt_summary: str | None = None,
        raw_response: str | None = None,
        parsed_action: dict | None = None,
        symbols: list[str] | None = None,
        execution_ms: int | None = None,
    ) -> int:
        cursor = await self._db.execute(
            """INSERT INTO llm_decisions
               (provider, model, prompt_summary, raw_response, parsed_action, symbols, execution_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                provider,
                model,
                prompt_summary,
                raw_response,
                json.dumps(parsed_action) if parsed_action else None,
                json.dumps(symbols) if symbols else None,
                execution_ms,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid  # type: ignore[return-value]
