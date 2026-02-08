"""SQLite table definitions and database initialization."""

from __future__ import annotations

import aiosqlite

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity        REAL    NOT NULL,
    price           REAL,
    order_id        TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',
    llm_reasoning   TEXT
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,
    starting_equity REAL    NOT NULL,
    ending_equity   REAL,
    realized_pnl    REAL    DEFAULT 0,
    return_pct      REAL,
    trades_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS halal_cache (
    symbol          TEXT    PRIMARY KEY,
    compliance      TEXT    NOT NULL CHECK (compliance IN ('halal', 'not_halal', 'doubtful')),
    detail          TEXT,
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    prompt_summary  TEXT,
    raw_response    TEXT,
    parsed_action   TEXT,
    symbols         TEXT,
    execution_ms    INTEGER
);
"""


async def init_db(db_path: str) -> aiosqlite.Connection:
    """Open (or create) the SQLite database and ensure schema exists."""
    db = await aiosqlite.connect(db_path)
    await db.executescript(SCHEMA_SQL)
    await db.commit()
    return db
