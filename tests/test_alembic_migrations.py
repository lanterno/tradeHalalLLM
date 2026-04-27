"""Sanity checks on the Postgres-baseline Alembic migration.

Pre-Postgres history (17 sqlite-shaped revisions) was archived into
``alembic/versions_legacy_sqlite/`` and squashed into a single fresh
initial revision. These tests verify that:

* The squashed revision applies cleanly against an empty SQLite (the
  test fixtures use SQLite — production uses Postgres).
* Every model declared in ``halal_trader.db.models`` is materialised.
* ``admin.head()`` resolves to a non-empty revision id.

For dialect-specific behaviour (pgvector, Postgres-only constraints),
add Postgres-only tests guarded by an ``@pytest.mark.requires_pg``
marker once the suite has a docker-compose-backed test fixture.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config

from alembic import command
from halal_trader.db import admin


def _alembic_cfg(db_path: Path) -> Config:
    cfg = admin._alembic_config()
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    import logging

    path = tmp_path / "alembic_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    import halal_trader.config as _config

    _config._settings = None

    saved_handlers = list(logging.getLogger().handlers)
    saved_level = logging.getLogger().level
    try:
        yield path
    finally:
        _config._settings = None
        root = logging.getLogger()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _tables_in(db_path: Path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def test_initial_revision_creates_expected_tables(db_path):
    """The squashed initial migration must materialise every core table."""
    cfg = _alembic_cfg(db_path)
    command.upgrade(cfg, "head")

    tables = _tables_in(db_path)
    expected = {
        "alembic_version",
        "trades",
        "daily_pnl",
        "halal_cache",
        "llm_decisions",
        "crypto_trades",
        "crypto_daily_pnl",
        "crypto_halal_cache",
        "indicator_snapshots",
        "strategy_adjustments",
        "kill_switch",
        "reconciliation_log",
    }
    assert expected.issubset(tables)


def test_head_resolves_to_a_revision_id():
    head = admin.head()
    assert isinstance(head, str)
    assert len(head) >= 12


def test_initial_revision_includes_post_phase0_columns(db_path):
    """Smoke-check that the squashed migration carries forward the
    Phase-0.x columns we used to test individually."""
    cfg = _alembic_cfg(db_path)
    command.upgrade(cfg, "head")

    with sqlite3.connect(str(db_path)) as conn:
        crypto_cols = {r[1] for r in conn.execute("PRAGMA table_info(crypto_trades)").fetchall()}
        llm_cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_decisions)").fetchall()}
    for col in ("submitted_at", "filled_at", "filled_price", "filled_quantity"):
        assert col in crypto_cols, f"crypto_trades missing {col}"
    for col in (
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
    ):
        assert col in llm_cols, f"llm_decisions missing {col}"
