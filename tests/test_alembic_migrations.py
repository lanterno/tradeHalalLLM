"""Tests that apply each Alembic revision forward against a fresh DB.

Catches: a revision that fails on a clean DB; a revision that fails when
applied on top of an adopted (create_all) DB; head-revision drift.
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
    """Per-test fresh DB; also overrides Settings so alembic/env.py uses it.

    `alembic/env.py` calls `logging.config.fileConfig` from `alembic.ini`
    which wipes ALL existing handlers (including pytest's caplog handler).
    We save+restore the root logger's handlers around the test so later
    suites that depend on log capture still work.
    """
    import logging

    path = tmp_path / "alembic_test.db"
    monkeypatch.setenv("DB_PATH", str(path))
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


def _columns_in(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def test_clean_db_migrates_to_head(db_path):
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


def test_each_revision_is_applied(db_path):
    cfg = _alembic_cfg(db_path)
    # Walk forward one revision at a time so any single broken upgrade fails loudly.
    revisions = [
        "b9da4efd8872",
        "c1a2b3d4e5f6",
        "d3e4f5a6b7c8",
        "e4f5a6b7c8d9",
        "f5a6b7c8d9e0",
        "a6b7c8d9e0f1",
        "b7c8d9e0f1a2",
    ]
    for rev in revisions:
        command.upgrade(cfg, rev)

    tables = _tables_in(db_path)
    assert "kill_switch" in tables
    assert "reconciliation_log" in tables


def test_p1_8_added_fill_columns(db_path):
    cfg = _alembic_cfg(db_path)
    command.upgrade(cfg, "head")

    crypto_cols = _columns_in(db_path, "crypto_trades")
    stock_cols = _columns_in(db_path, "trades")

    for col in ("submitted_at", "filled_at", "filled_price", "filled_quantity"):
        assert col in crypto_cols, f"crypto_trades missing {col}"
        assert col in stock_cols, f"trades missing {col}"


def test_kill_switch_seeded_with_singleton_row(db_path):
    cfg = _alembic_cfg(db_path)
    command.upgrade(cfg, "head")

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT id, enabled FROM kill_switch").fetchall()
    assert rows == [(1, 0)]


def test_admin_head_matches_known_revision():
    """Tripwire so adding a new revision without updating tests fails CI."""
    assert admin.head() == "b7c8d9e0f1a2"


def test_idempotent_revision_is_safe_to_replay(db_path):
    """Adopted-DB scenario: tables created by create_all already exist; the
    catch-up revision e4f5a6b7c8d9 must not error on re-application."""

    # Pre-create the same tables that the catch-up revision adds, then
    # stamp at the prior revision and re-run forward to head. The catch-up
    # revision uses CREATE TABLE IF NOT EXISTS / column guards so it must
    # not error.
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE indicator_snapshots ("
            "id INTEGER PRIMARY KEY, trade_id INTEGER NOT NULL,"
            "pair TEXT, timestamp DATETIME)"
        )
        conn.execute(
            "CREATE TABLE strategy_adjustments ("
            "id INTEGER PRIMARY KEY, parameter TEXT, new_value REAL)"
        )

    cfg = _alembic_cfg(db_path)
    command.upgrade(cfg, "d3e4f5a6b7c8")  # land just before catch-up
    command.stamp(cfg, "d3e4f5a6b7c8")
    # Catch-up + later revisions must not error on the already-present tables.
    command.upgrade(cfg, "head")
