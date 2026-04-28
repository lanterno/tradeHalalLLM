"""Sanity checks on the Postgres-baseline Alembic migration."""

from __future__ import annotations

import psycopg
import pytest
from alembic.config import Config

from alembic import command
from halal_trader.db import admin
from tests.conftest import (
    _ADMIN_DSN_SYNC,
    PG_HOST,
    PG_PASS,
    PG_PORT,
    PG_USER,
    _terminate_and_drop,
)


@pytest.fixture
def scratch_db(monkeypatch):
    import uuid

    name = f"halal_trader_alembic_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(_ADMIN_DSN_SYNC, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{name}"')
    sync_url = f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{name}"
    raw_url = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{name}"
    async_url = f"postgresql+asyncpg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{name}"
    monkeypatch.setenv("DATABASE_URL", async_url)
    import halal_trader.config as _config

    monkeypatch.setattr(_config, "_settings", None, raising=False)
    try:
        yield raw_url, sync_url
    finally:
        _terminate_and_drop(name)


def _alembic_cfg(sync_url: str) -> Config:
    cfg = admin._alembic_config()
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


def _tables_in(raw_url: str) -> set[str]:
    with psycopg.connect(raw_url) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


def _columns_in(raw_url: str, table: str) -> set[str]:
    with psycopg.connect(raw_url) as conn:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        ).fetchall()
    return {r[0] for r in rows}


def test_initial_revision_creates_expected_tables(scratch_db):
    raw_url, sync_url = scratch_db
    command.upgrade(_alembic_cfg(sync_url), "head")

    tables = _tables_in(raw_url)
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


def test_initial_revision_includes_post_phase0_columns(scratch_db):
    """Smoke-check that the squashed migration carries forward the
    Phase-0.x columns we used to test individually."""
    raw_url, sync_url = scratch_db
    command.upgrade(_alembic_cfg(sync_url), "head")

    crypto_cols = _columns_in(raw_url, "crypto_trades")
    for col in ("submitted_at", "filled_at", "filled_price", "filled_quantity"):
        assert col in crypto_cols, f"crypto_trades missing {col}"
    llm_cols = _columns_in(raw_url, "llm_decisions")
    for col in (
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
    ):
        assert col in llm_cols, f"llm_decisions missing {col}"
