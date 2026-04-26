"""CLI tests for ``halal-trader llm-decisions`` — list / show / cost-summary."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from click.testing import CliRunner
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from halal_trader.cli import cli
from halal_trader.db import admin


def _seed_db_sync(db_path: Path) -> None:
    """Build a fresh DB at ``db_path`` and insert a few decision rows."""
    import asyncio

    async def _build() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        head = admin.head()
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
            await conn.execute(
                sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
            )
            await conn.execute(
                sa.text(f"INSERT INTO alembic_version (version_num) VALUES ('{head}')")
            )
            # Two decisions: one with full cost data, one minimal. Use bound
            # params so JSON colons aren't mistaken for SQLAlchemy bind names.
            await conn.execute(
                sa.text(
                    "INSERT INTO llm_decisions "
                    "(timestamp, provider, model, prompt_summary, raw_response, "
                    " parsed_action, symbols, execution_ms, prompt_version, "
                    " input_tokens, output_tokens, cache_read_tokens, "
                    " cache_write_tokens, cost_usd) "
                    "VALUES (:ts, :prov, :model, :summ, :raw, :parsed, :syms, "
                    " :ms, :pv, :it, :ot, :crt, :cwt, :cost)"
                ),
                {
                    "ts": "2026-04-26T12:00:00",
                    "prov": "anthropic",
                    "model": "claude-opus-4-7",
                    "summ": "crypto cycle: analyzed 10 pairs",
                    "raw": '{"decisions":[]}',
                    "parsed": '{"buys":1,"sells":0,"holds":9}',
                    "syms": '["BTCUSDT","ETHUSDT"]',
                    "ms": 1234,
                    "pv": "crypto.strategy.system@abc123",
                    "it": 1500,
                    "ot": 200,
                    "crt": 500,
                    "cwt": 100,
                    "cost": 0.0234,
                },
            )
            await conn.execute(
                sa.text(
                    "INSERT INTO llm_decisions "
                    "(timestamp, provider, model, prompt_summary, raw_response, "
                    " execution_ms) "
                    "VALUES (:ts, :prov, :model, :summ, :raw, :ms)"
                ),
                {
                    "ts": "2026-04-26T13:00:00",
                    "prov": "openai",
                    "model": "gpt-4o",
                    "summ": "stock cycle: analyzed 5 symbols",
                    "raw": '{"decisions":[]}',
                    "ms": 800,
                },
            )
        await engine.dispose()

    asyncio.run(_build())


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a seeded DB and point Settings at it for the duration of one test."""
    db_path = tmp_path / "cli.db"
    _seed_db_sync(db_path)

    # The CLI calls get_settings() which returns a cached Settings; override
    # only the resolved DB path so other defaults (Ollama, etc.) are intact.
    monkeypatch.setenv("DB_PATH", str(db_path))
    # Reset the cached Settings singleton so DB_PATH takes effect.
    from halal_trader import config

    monkeypatch.setattr(config, "_settings", None)
    # Rich respects COLUMNS for the terminal width; set wide so tabular
    # output isn't truncated below the cells we're asserting on.
    monkeypatch.setenv("COLUMNS", "240")
    yield db_path


def test_list_default_renders_table(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "list"])
    assert result.exit_code == 0, result.output
    assert "claude-opus-4-7" in result.output
    assert "gpt-4o" in result.output
    # Cost should render as $0.0234 — we treat absent cost as "—".
    assert "$0.0234" in result.output
    assert "—" in result.output  # row 2 has no cost


def test_list_filter_by_provider(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "list", "--provider", "anthropic"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output
    assert "openai" not in result.output


def test_show_prints_full_record(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "show", "1"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output
    assert "crypto.strategy.system@abc123" in result.output
    assert "BTCUSDT" in result.output
    assert "$0.0234" in result.output


def test_show_unknown_id_prints_message(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "show", "9999"])
    assert result.exit_code == 0
    assert "No decision found" in result.output


def test_cost_summary_groups_by_day_and_model(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "cost-summary", "--days", "365"])
    assert result.exit_code == 0, result.output
    assert "claude-opus-4-7" in result.output
    assert "gpt-4o" in result.output
    assert "Daily totals:" in result.output
