"""The Operations page's payload (web/operations.build) and its helpers."""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.config import get_settings
from halal_trader.core import heartbeat as hb
from halal_trader.market_hours import MARKET_TZ
from halal_trader.web.operations import NOT_DUMPED, _summary, build

# Thursday 8 October 2026, 12:00 in New York: the session is open.
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=MARKET_TZ)


def test_next_run_is_the_first_scheduled_run_after_now() -> None:
    research = hb.DAILY_JOBS[hb.RESEARCH]
    assert hb.next_run(research, NOW) == datetime(2026, 10, 8, 20, 30, tzinfo=MARKET_TZ)
    late = datetime(2026, 10, 8, 21, 0, tzinfo=MARKET_TZ)
    # Friday is next; the weekend is skipped after it.
    assert hb.next_run(research, late) == datetime(2026, 10, 9, 20, 30, tzinfo=MARKET_TZ)
    friday_night = datetime(2026, 10, 9, 21, 0, tzinfo=MARKET_TZ)
    assert hb.next_run(research, friday_night) == datetime(2026, 10, 12, 20, 30, tzinfo=MARKET_TZ)
    digest = hb.DAILY_JOBS[hb.WEEKLY_DIGEST]
    assert hb.next_run(digest, NOW) == datetime(2026, 10, 9, 17, 15, tzinfo=MARKET_TZ)


def test_next_run_keeps_an_early_close_time() -> None:
    eve = datetime(2026, 11, 27, 9, 0, tzinfo=MARKET_TZ)  # the Friday after Thanksgiving
    core = hb.DAILY_JOBS[hb.CORE_TRADE]
    assert hb.next_run(core, eve) == datetime(2026, 11, 27, 12, 40, tzinfo=MARKET_TZ)


def test_each_job_says_what_it_did() -> None:
    assert _summary(hb.RECOMMENDATION, {"symbol": "MSFT"}) == "picked MSFT"
    assert _summary(hb.STOCK_LEDGER, {"broker_fills": 2}) == "2 broker fill(s) recorded"
    assert _summary(hb.WEEKLY_DIGEST, None) == "sent"
    research = {"books": {"core": 1}, "screened": 1499, "rescreen_for": ["NKE"], "errors": 0}
    assert _summary(hb.RESEARCH, research) == (
        "1,499 screened · re-screened NKE · books: core · no errors"
    )
    assert _summary(hb.CORE_TRADE, {"account": "core", "refused": True}) == (
        "refused: see the alert"
    )


_JUSTFILE = Path(__file__).parents[1] / "justfile"
_PLAYBOOK_TABLES = {"hb_playbook_run", "hb_playbook_story", "hb_playbook_trade"}


def test_the_tables_marked_not_dumped_are_the_backup_recipes() -> None:
    recipe = re.search(r"for t in ([a-z_ ]+); do\s+excluded=", _JUSTFILE.read_text())
    assert recipe is not None
    assert set(recipe.group(1).split()) == NOT_DUMPED


def test_playbook_rows_leave_the_dump_and_runs_but_simulations_are_exported() -> None:
    justfile = _JUSTFILE.read_text()
    assert _PLAYBOOK_TABLES <= NOT_DUMPED
    assert '| gzip > "{{dest}}/playbook_runs.jsonl.gz"' in justfile
    assert "playbook_runs_kb" in justfile  # the nightly beat says how big it was


def _playbook_export() -> str:
    """The backup recipe's query for the playbook runs, as it runs there."""
    found = re.search(r'\n\s+runs="([^"]+)"\n', _JUSTFILE.read_text())
    assert found is not None
    return found.group(1)


async def _playbook_run(engine: AsyncEngine, mode: str, at: datetime, stories: int) -> str:
    """A run of ``mode`` with ``stories`` stories, each one traded."""
    from halabot.platform.db import playbook_run, playbook_story, playbook_trade

    run_id = uuid.uuid4()
    day = at.date()
    async with engine.begin() as conn:
        await conn.execute(
            sa.insert(playbook_run).values(
                run_id=run_id,
                created_at=at,
                mode=mode,
                playbook="bounce",
                playbook_version="1",
                cell="c",
                feed="sip-rt",
                window="train",
                stop_at="end",
                config={},
                config_hash="h",
            )
        )
        for n in range(stories):
            story = f"AAPL:{day.isoformat()}:{n}"
            await conn.execute(
                sa.insert(playbook_story).values(
                    run_id=run_id,
                    story_id=story,
                    symbol="AAPL",
                    session=day,
                    terminal_state="EXITED",
                    reason="target",
                )
            )
            await conn.execute(
                sa.insert(playbook_trade).values(
                    run_id=run_id,
                    story_id=story,
                    symbol="AAPL",
                    family_type="NSN_CORE",
                    cell="c",
                    variant="v",
                    feed="sip-rt",
                    session=day,
                    exit_session=day,
                    sessions_held=1,
                    start_case="in",
                    entry_decided_at=at,
                    entry_active_at=at,
                    entry_bar_ts=at,
                    exit_decided_at=at,
                    exit_active_at=at,
                    rank=1,
                    tech=True,
                    exit_reason="target",
                    hold_minutes=30,
                    flags=[],
                    legs=[],
                )
            )
    return str(run_id)


async def test_the_playbook_export_holds_every_run_but_the_simulations(engine) -> None:
    from halabot.platform.db import bootstrap_schema

    await bootstrap_schema(engine)
    await _playbook_run(engine, "sim", NOW - timedelta(days=4), stories=3)
    shadow = await _playbook_run(engine, "shadow", NOW - timedelta(days=3), stories=2)
    paper = await _playbook_run(engine, "paper", NOW - timedelta(days=2), stories=1)
    live = await _playbook_run(engine, "live", NOW - timedelta(days=1), stories=0)
    async with engine.connect() as conn:
        raw = (
            await conn.execute(text(f"SELECT j::text FROM ({_playbook_export()}) AS e(j)"))
        ).scalars()
        exported = list(raw)
    assert all("\n" not in line for line in exported)  # one run per line of the .jsonl
    lines = [json.loads(line) for line in exported]
    assert [(r["run_id"], r["mode"]) for r in lines] == [
        (shadow, "shadow"),
        (paper, "paper"),
        (live, "live"),
    ]
    assert [len(r["stories"]) for r in lines] == [2, 1, 0]
    assert [len(r["trades"]) for r in lines] == [2, 1, 0]
    first = lines[0]
    assert first["stories"][0]["story_id"] == "AAPL:2026-10-05:0"
    assert first["trades"][1]["exit_reason"] == "target"
    assert first["config_hash"] == "h"  # every column of the run


async def _beat(engine: AsyncEngine, component: str, at: datetime, detail: object = None) -> None:
    await hb.beat(engine, component, detail, now=at)  # type: ignore[arg-type]


async def _seed(engine: AsyncEngine) -> None:
    fresh = NOW - timedelta(seconds=30)
    await _beat(engine, hb.STOCK_PROCESS, fresh, {"core": True})
    await _beat(engine, hb.STOCK_MONITOR, fresh)
    await _beat(engine, hb.STOCK_CYCLE, NOW - timedelta(minutes=10))
    await _beat(engine, hb.MARKET_SNAPSHOT, fresh)
    await _beat(engine, hb.WATCHDOG, fresh, {"suspect": [], "alerting": []})
    # No shadow beat at all; the evening research ran Tuesday, not Wednesday.
    await _beat(engine, hb.RESEARCH, datetime(2026, 10, 6, 20, 45, tzinfo=MARKET_TZ), {"errors": 0})
    await _beat(
        engine, hb.RECOMMENDATION, datetime(2026, 10, 8, 9, 5, tzinfo=MARKET_TZ), {"symbol": "MSFT"}
    )
    await _beat(engine, hb.BACKUP_NIGHTLY, NOW - timedelta(hours=5), {"dump_mb": 434})
    await _beat(engine, hb.BACKUP_OFFSITE, NOW - timedelta(hours=5), {"snapshot": "b7d3f1a9"})
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO llm_spend (day, consumer, calls, spent_usd) VALUES "
                "(:t, 'stock', 300, 0.14), (:t, 'shadow', 600, 0.15), "
                "(:y, 'stock', 310, 0.12), (:t, 'research', 11, 0.03)"
            ),
            {"t": date(2026, 10, 8), "y": date(2026, 10, 7)},
        )
        await conn.execute(
            text(
                "INSERT INTO core_runs (account, run_on, monthly, executed, equity, cash, "
                "orders, screen_as_of, recorded_at) VALUES "
                "('core', '2026-10-07', false, true, 1000, 10, 0, '2026-10-05', now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES "
                "('2026-10-05', 'MSFT', 1, '', 'halal', '[]', '{}', 'v12', now()), "
                "('2026-10-05', 'XOM', 2, '', 'not_halal', '[]', '{}', 'v12', now())"
            )
        )


async def test_the_page_names_every_problem_and_judges_jobs_by_the_calendar(engine) -> None:
    await _seed(engine)
    body = await build(engine, get_settings(), now=NOW)

    fleet = body["fleet"]
    assert fleet["alive"] and fleet["verdict"] == "degraded"
    assert any(p.startswith("Shadow engine") for p in fleet["problems"])
    assert any(p.startswith("Evening research") for p in fleet["problems"])
    assert body["halt"]["enabled"] is False and body["market"]["open"] is True
    assert body["paper"] is True

    jobs = {j["component"]: j for j in body["jobs"]}
    assert jobs[hb.RESEARCH]["status"] == "stale"
    assert jobs[hb.RECOMMENDATION]["status"] == "ok" and jobs[hb.RECOMMENDATION]["ran_today"]
    assert jobs[hb.RECOMMENDATION]["summary"] == "picked MSFT"
    assert jobs[hb.CORE_TRADE]["status"] == "unknown" and not jobs[hb.CORE_TRADE]["due_today"]
    assert jobs[hb.CORE_TRADE]["next"] == "2026-10-08T15:40:00-04:00"
    assert jobs[hb.WEEKLY_DIGEST]["today_at"] is None  # Fridays only
    # Tuesday's research beat is no run of today's, nor would a catch-up at dawn be.
    assert not jobs[hb.RESEARCH]["ran_today"]

    procs = {p["component"]: p for p in body["processes"]}
    assert procs[hb.SHADOW_PROCESS]["status"] == "missing"
    assert procs[hb.STOCK_CYCLE]["due"] and procs[hb.STOCK_CYCLE]["status"] == "ok"
    assert procs[hb.STOCK_PROCESS]["age_seconds"] == 30.0


async def test_llm_spend_is_read_from_the_meter_by_pool(engine) -> None:
    await _seed(engine)
    llm = (await build(engine, get_settings(), now=NOW))["llm"]

    assert llm["today"] == pytest.approx(0.32) and llm["calls_today"] == 911
    live = next(p for p in llm["pools"] if p["pool"] == "live")
    assert live["today"] == pytest.approx(0.29) and live["daily_cap"] == 3.0
    assert live["month"] == pytest.approx(0.41) and live["monthly_cap"] == 25.0
    assert live["pace"] == pytest.approx(0.41 / 8 * 31, abs=1e-3)
    research = next(p for p in llm["pools"] if p["pool"] == "research")
    assert research["daily_cap"] is None and research["monthly_cap"] == 15.0
    stock = next(c for c in llm["consumers"] if c["consumer"] == "stock")
    assert (stock["today"], stock["yesterday"], stock["month"]) == pytest.approx((0.14, 0.12, 0.26))
    assert len(llm["days"]) == 14 and llm["days"][-1]["shadow"] == pytest.approx(0.15)


async def test_backups_freshness_database_and_deploy(engine) -> None:
    await _seed(engine)
    body = await build(engine, get_settings(), now=NOW)

    backups = body["backups"]
    assert backups["nightly"]["status"] == "ok" and backups["nightly"]["detail"]["dump_mb"] == 434
    assert backups["drill"]["status"] == "missing"
    assert backups["next_nightly"] == "2026-10-09T07:00:00+00:00"
    assert backups["next_drill"] == "2026-11-01T07:00:00+00:00"

    fresh = {f["name"]: f for f in body["freshness"]}
    assert (
        fresh["Halal screen"]["as_of"] == "2026-10-05" and fresh["Halal screen"]["status"] == "ok"
    )
    assert "2 names, 1 halal" in fresh["Halal screen"]["detail"]
    assert fresh["Daily bars"]["status"] == "unknown"  # nothing stored

    db = body["database"]
    assert db["bytes"] > 0 and db["tables"] > 10 and db["connections"] >= 1
    assert all(t["dumped"] == (t["name"] not in NOT_DUMPED) for t in db["biggest"])
    assert body["deploy"]["schema_ok"] is True

    jobs = {j["component"]: j for j in body["jobs"]}
    assert jobs[hb.CORE_TRADE]["summary"] is None  # no beat yet, whatever core_runs holds
    labels = [row[0] for row in body["config"]["core"]]
    assert "Rebalance band" in labels and "Trades at" in labels
    assert "key" not in json.dumps(body["config"]).lower()


async def test_the_core_trade_says_what_its_run_did(engine) -> None:
    await _seed(engine)
    await _beat(
        engine,
        hb.CORE_TRADE,
        datetime(2026, 10, 7, 15, 41, tzinfo=MARKET_TZ),
        {
            "account": "core",
            "refused": False,
        },
    )
    jobs = {j["component"]: j for j in (await build(engine, get_settings(), now=NOW))["jobs"]}
    assert jobs[hb.CORE_TRADE]["summary"] == "sells-only day · 0 order(s) · screen of Mon 05 Oct"


def test_the_route_serves_the_page(database_url, tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from halal_trader import config
    from halal_trader.web import app as web_app

    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "_settings", None)
    with TestClient(web_app.create_app()) as client:
        body = client.get("/api/operations").json()
    config._settings = None
    assert body["deploy"]["web_started"] is not None
    assert {"fleet", "jobs", "processes", "llm", "backups", "database"} <= set(body)


async def test_the_broker_ledger_is_expected_a_session_behind(engine) -> None:
    """Alpaca publishes a session's close the next day: on Thursday the 16:30
    sync brings in Wednesday, so until then Tuesday is as current as it gets."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO broker_equity (account, day, equity, profit_loss, profit_loss_pct, "
                "synced_at) VALUES ('core', '2026-10-06', 100000, 0, 0, now())"
            )
        )

    async def ledger(at: datetime) -> dict:
        body = await build(engine, get_settings(), now=at)
        return next(f for f in body["freshness"] if f["name"] == "Broker equity · core")

    assert (await ledger(NOW))["status"] == "ok"  # Thursday noon
    evening = await ledger(datetime(2026, 10, 8, 18, 0, tzinfo=MARKET_TZ))
    assert evening["status"] == "stale" and "expected Wed 07 Oct" in evening["detail"]


async def test_a_low_llm_credit_is_a_problem_on_the_page(engine) -> None:
    from halal_trader.core.llm.credits import LLM_CREDITS

    await _seed(engine)
    await _beat(
        engine,
        LLM_CREDITS,
        NOW,
        {"balance_usd": 0.1, "key_remaining_usd": 39.8, "pace_usd_per_day": 0.5},
    )
    body = await build(engine, get_settings(), now=NOW)
    credits = body["llm"]["credits"]
    assert credits["available_usd"] == pytest.approx(0.1) and credits["low"] is True
    assert any(p.startswith("LLM credit low: $0.10") for p in body["fleet"]["problems"])
