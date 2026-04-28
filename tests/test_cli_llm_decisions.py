"""CLI tests for ``halal-trader llm-decisions`` — list / show / cost-summary."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.cli import cli


async def _seed(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
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
                    "ts": datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC),
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
                    "ts": datetime(2026, 4, 26, 13, 0, 0, tzinfo=UTC),
                    "prov": "openai",
                    "model": "gpt-4o",
                    "summ": "stock cycle: analyzed 5 symbols",
                    "raw": '{"decisions":[]}',
                    "ms": 800,
                },
            )
    finally:
        await engine.dispose()


@pytest.fixture
def cli_db(database_url: str, monkeypatch: pytest.MonkeyPatch):
    import asyncio

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        _seed(database_url)
    ) if False else asyncio.run(_seed(database_url))
    monkeypatch.setenv("COLUMNS", "240")
    return database_url


def test_list_default_renders_table(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "list"])
    assert result.exit_code == 0, result.output
    assert "claude-opus-4-7" in result.output
    assert "gpt-4o" in result.output
    assert "$0.0234" in result.output
    assert "—" in result.output


def test_list_filter_by_provider(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli, ["llm-decisions", "list", "--provider", "anthropic"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output
    assert "openai" not in result.output


def test_show_prints_full_record(cli_db):
    runner = CliRunner()
    # Look up the first row id (sequential per fresh DB but assert by content).
    sync_url = cli_db.replace("+asyncpg", "+psycopg")
    eng = create_engine(sync_url)
    with eng.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT id FROM llm_decisions WHERE provider='anthropic' LIMIT 1")
        ).fetchone()
    eng.dispose()
    rid = rows[0]
    result = runner.invoke(cli, ["llm-decisions", "show", str(rid)])
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
