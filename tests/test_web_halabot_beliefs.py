"""/api/halabot/* — belief-board bridge from the dashboard to the shadow engine."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from halal_trader.web import app as web_app


@pytest.fixture
def client(database_url, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WEB_API_TOKEN", "secret")
    monkeypatch.setenv("WEB_REQUIRE_CONFIRMATION", "false")
    app = web_app.create_app()
    with TestClient(app) as c:
        yield c


def _seed_belief(database_url: str, asset: str, conviction: float) -> None:
    """Bootstrap the hb_ schema and persist one belief via the real store."""

    async def _go() -> None:
        from halabot.belief.schema import BeliefState, Catalyst, Direction
        from halabot.belief.store import PgBeliefStore
        from halabot.platform.db import bootstrap_schema

        engine = create_async_engine(database_url)
        try:
            await bootstrap_schema(engine)
            b = BeliefState.neutral(asset)
            b.direction = Direction.LONG_BIAS
            b.conviction = conviction
            b.thesis = "uptrend holding above support"
            b.catalysts_pending = [
                Catalyst(
                    kind="CPI",
                    scheduled_for=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
                    expected_impact=0.9,
                    detail="CPI release",
                )
            ]
            b.last_updated = datetime.now(UTC)
            await PgBeliefStore(engine).put(b)
        finally:
            await engine.dispose()

    asyncio.run(_go())


def test_board_empty_is_available_false(client):
    r = client.get("/api/halabot/beliefs")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["beliefs"] == []


def test_board_lists_seeded_belief_with_catalysts(client, database_url):
    _seed_belief(database_url, "NVDA", 0.62)
    r = client.get("/api/halabot/beliefs")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    row = next(b for b in body["beliefs"] if b["asset"] == "NVDA")
    assert row["conviction"] == pytest.approx(0.62)
    assert row["thesis"] == "uptrend holding above support"
    assert row["catalysts_pending"] == [
        {
            "kind": "CPI",
            "scheduled_for": "2026-07-14T12:30:00+00:00",
            "expected_impact": 0.9,
            "detail": "CPI release",
        }
    ]
    assert "support" in row and "horizon" in row  # new payload keys


def test_single_asset_detail_and_404(client, database_url):
    _seed_belief(database_url, "AAPL", 0.4)
    assert client.get("/api/halabot/beliefs/aapl").json()["asset"] == "AAPL"
    assert client.get("/api/halabot/beliefs/ZZZZ").status_code == 404


def test_decisions_endpoint_empty_list(client, database_url):
    _seed_belief(database_url, "AAPL", 0.4)  # ensures hb_ tables exist
    r = client.get("/api/halabot/decisions?limit=5")
    assert r.status_code == 200
    assert r.json() == []


def test_bad_correlation_id_is_400(client):
    assert client.get("/api/halabot/decisions/not-a-uuid").status_code == 400


# ── Belief Board v2: the board's context, the overview, plain decisions ──

NOW = datetime.now(UTC)


def _run(database_url: str, fn) -> None:  # type: ignore[no-untyped-def]
    """Run ``fn(engine)`` against the test DB with the hb_ schema in place."""

    async def _go() -> None:
        from halabot.platform.db import bootstrap_schema

        engine = create_async_engine(database_url)
        try:
            await bootstrap_schema(engine)
            await fn(engine)
        finally:
            await engine.dispose()

    asyncio.run(_go())


def _belief(
    database_url: str,
    asset: str,
    conviction: float,
    *,
    direction: str = "long_bias",
    evidence: list[tuple[str, float, float, str]] | None = None,
    catalysts: list[tuple[str, datetime, float]] | None = None,
) -> None:
    async def fn(engine) -> None:  # type: ignore[no-untyped-def]
        from halabot.belief.schema import (
            BeliefState,
            Catalyst,
            Direction,
            EvidenceItem,
            Levels,
        )
        from halabot.belief.store import PgBeliefStore

        b = BeliefState.neutral(asset)
        b.direction = Direction(direction)
        b.conviction = b.conviction_raw = conviction
        b.levels = Levels(support=95.0, resistance=110.0, stop=95.0, invalidation=95.0)
        b.evidence = [
            EvidenceItem(source=s, direction=d, weight=w, detail=det)
            for s, d, w, det in (evidence or [])
        ]
        b.catalysts_pending = [
            Catalyst(kind=k, scheduled_for=t, expected_impact=i, detail=f"{k} release")
            for k, t, i in (catalysts or [])
        ]
        b.last_updated = NOW
        await PgBeliefStore(engine).put(b)

    _run(database_url, fn)


def _event(database_url: str, kind: str, asset: str, at: datetime, payload: dict) -> None:
    async def fn(engine) -> None:  # type: ignore[no-untyped-def]
        from halabot.platform.clock import FakeClock
        from halabot.platform.event_log import PgEventLog
        from halabot.platform.events import EventType, new_event

        await PgEventLog(engine).append(
            new_event(FakeClock(at), EventType(kind), source="test", asset=asset, payload=payload)
        )

    _run(database_url, fn)


def _bar(database_url: str, asset: str, close: float, at: datetime) -> None:
    bar_ts = at.replace(minute=0, second=0, microsecond=0).isoformat()
    payload = {"o": close, "h": close, "low": close, "c": close, "v": 1.0, "bar_ts": bar_ts}
    _event(database_url, "observation.bar", asset, at, payload)


def _proposal(database_url: str, asset: str, at: datetime, payload: dict) -> None:
    _event(database_url, "policy.trade_proposed", asset, at, {"shadow": True, **payload})


def _sql(database_url: str, statement: str, params: list[dict] | dict | None = None) -> None:
    from sqlalchemy import text

    async def fn(engine) -> None:  # type: ignore[no-untyped-def]
        async with engine.begin() as conn:
            await conn.execute(text(statement), params or {})

    _run(database_url, fn)


def _screen(database_url: str, verdicts: dict[str, str], as_of=None) -> None:  # type: ignore[no-untyped-def]
    from halal_trader.market_hours import today_eastern

    _sql(
        database_url,
        "INSERT INTO halal_screen_results (as_of, symbol, sic_description, verdict, reasons, "
        "metrics, method, screened_at) VALUES (:d, :s, '', :v, '[]', '{}', 'test', now())",
        [{"d": as_of or today_eastern(), "s": s, "v": v} for s, v in verdicts.items()],
    )


def _position(database_url: str, asset: str, weight: float, ret: float, cohort: int | None) -> None:
    _sql(
        database_url,
        "INSERT INTO hb_open_position (asset, entry_ts, entry_vwap, weight, last_price, "
        "unrealized_return_pct, belief_version, updated_at, cohort) "
        "VALUES (:a, now() - interval '2 hours', 100, :w, 100 * (1 + CAST(:r AS float)), "
        "CAST(:r AS float), 1, now(), :c)",
        {"a": asset, "w": weight, "r": ret, "c": cohort},
    )


def _outcomes(database_url: str, rows: list[dict]) -> None:
    _sql(
        database_url,
        "INSERT INTO hb_outcome (asset, entry_ts, exit_ts, entry_price, exit_price, "
        "closed_weight, return_pct, hold_seconds, belief_version, label, created_at, cohort) "
        "VALUES (:a, CAST(:e AS timestamptz), CAST(:e AS timestamptz) + interval '3 hours', "
        "100, 100 * (1 + CAST(:r AS float)), 0.05, CAST(:r AS float), 10800, 1, "
        "CAST(CAST(:r AS float) > 0.002 AS int), now(), :c)",
        [{**r, "e": datetime.fromisoformat(r["e"])} for r in rows],
    )


def _board(client) -> dict:  # type: ignore[no-untyped-def]
    body = client.get("/api/halabot/beliefs").json()
    return {"_": body, **{b["asset"]: b for b in body["beliefs"]}}


def test_board_stances_follow_the_bands_and_the_strict_screen(client, database_url):
    _belief(database_url, "AVGO", 0.62)  # long: above the 35% entry band
    _belief(database_url, "MSFT", 0.20)  # leaning: above the 15% exit band only
    _belief(database_url, "AAPL", 0.10)  # no view: below both
    _belief(database_url, "AMAT", 0.80, direction="neutral")  # no view: not long
    _belief(database_url, "NVDA", 0.70)  # excluded: fails the strict screen
    _belief(database_url, "TSM", 0.50)  # excluded: doubtful is not halal
    _belief(database_url, "CRM", 0.50)  # excluded: not in the screen at all
    _belief(database_url, "SPY", 0.48)  # benchmark, though no screen covers it
    _screen(
        database_url,
        {
            "AVGO": "halal",
            "MSFT": "halal",
            "AAPL": "halal",
            "AMAT": "halal",
            "NVDA": "not_halal",
            "TSM": "doubtful",
        },
    )
    board = _board(client)
    assert board["_"]["entry_band"] == pytest.approx(0.35)
    assert board["_"]["exit_band"] == pytest.approx(0.15)
    assert board["_"]["benchmark"] == "SPY"
    assert board["_"]["screen_stale"] is False
    names = ("AVGO", "MSFT", "AAPL", "AMAT", "NVDA", "TSM", "CRM", "SPY")
    assert {a: board[a]["stance"] for a in names} == {
        "AVGO": "long",
        "MSFT": "leaning",
        "AAPL": "none",
        "AMAT": "none",
        "NVDA": "excluded",
        "TSM": "excluded",
        "CRM": "excluded",
        "SPY": "benchmark",
    }
    assert board["NVDA"]["strict"] == "not_halal"
    assert board["TSM"]["strict"] == "doubtful"
    assert board["CRM"]["strict"] == "unscreened"
    # Ranked by conviction, highest first.
    assert [b["asset"] for b in board["_"]["beliefs"]][:2] == ["AMAT", "NVDA"]


def test_board_entry_band_is_inclusive_and_read_from_config(client, database_url, monkeypatch):
    from halabot.platform import config

    monkeypatch.setenv("HALABOT_POLICY__CONVICTION_ENTRY_BAND", "0.5")
    config.get_settings.cache_clear()
    try:
        _belief(database_url, "AVGO", 0.5)
        _belief(database_url, "AMD", 0.49)
        _screen(database_url, {"AVGO": "halal", "AMD": "halal"})
        board = _board(client)
        assert board["_"]["entry_band"] == pytest.approx(0.5)
        assert board["AVGO"]["stance"] == "long"
        assert board["AMD"]["stance"] == "leaning"
    finally:
        config.get_settings.cache_clear()


def test_a_stale_screen_excludes_every_name(client, database_url):
    from halal_trader.market_hours import today_eastern

    _belief(database_url, "AVGO", 0.62)
    _screen(database_url, {"AVGO": "halal"}, as_of=today_eastern() - timedelta(days=30))
    board = _board(client)
    assert board["_"]["screen_stale"] is True
    assert board["AVGO"]["strict"] == "halal"  # what the old screen said
    assert board["AVGO"]["stance"] == "excluded"  # but the engine cannot follow it


def test_board_price_core_and_shadow_weights(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    _belief(database_url, "AMD", 0.55)
    _screen(database_url, {"AVGO": "halal", "AMD": "halal"})
    _bar(database_url, "AVGO", 370.0, NOW - timedelta(hours=2))
    _bar(database_url, "AVGO", 378.46, NOW - timedelta(minutes=20))
    _bar(database_url, "AMD", 600.0, NOW - timedelta(days=10))  # too old to be a price
    _sql(
        database_url,
        "INSERT INTO account_snapshots (account, taken_at, equity, cash, positions) "
        "VALUES ('core', now(), 100000, 1000, CAST(:p AS JSONB))",
        {"p": '[{"symbol": "AVGO", "market_value": 5900.0}, {"symbol": "X", "market_value": 1}]'},
    )
    _position(database_url, "AVGO", 0.1072, 0.0224, cohort=2)
    _position(database_url, "AMD", 0.08, 0.01, cohort=None)  # an earlier cohort's leftover
    board = _board(client)
    avgo, amd = board["AVGO"], board["AMD"]
    assert avgo["price"] == pytest.approx(378.46)
    assert datetime.fromisoformat(avgo["price_at"]) > NOW - timedelta(minutes=21)
    assert amd["price"] is None and amd["price_at"] is None  # no price in the window
    assert avgo["core_weight"] == pytest.approx(0.059)
    assert amd["core_weight"] is None  # not held by the core
    assert avgo["shadow"]["weight"] == pytest.approx(0.1072)
    assert avgo["shadow"]["return_pct"] == pytest.approx(0.0224)
    assert avgo["shadow"]["entry_price"] == pytest.approx(100.0)
    assert amd["shadow"] is None  # only the current cohort is the shadow's book
    assert board["_"]["core_as_of"] is not None


def test_board_without_a_core_snapshot_or_prices(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    board = _board(client)
    assert board["AVGO"]["core_weight"] is None
    assert board["AVGO"]["price"] is None
    assert board["AVGO"]["shadow"] is None
    assert board["_"]["core_as_of"] is None
    assert board["_"]["screen_as_of"] is None  # no screen: nothing is halal
    assert board["AVGO"]["stance"] == "excluded"


def test_board_evidence_in_plain_words(client, database_url):
    _belief(
        database_url,
        "AVGO",
        0.62,
        evidence=[
            ("indicator.relstrength", 1.0, 0.53, "rel +4.26% vs SPY"),
            ("indicator.rsi", 1.0, 0.38, "RSI 95"),
            ("indicator.alignment", 0.8, 0.46, "short +4.27% / long +8.47%"),
            ("news", -0.6, 0.30, "news(llm) -0.60: Chipmaker Faces Probe"),
            ("anomaly", 0.0, 1.0, "vol spike 2.3x baseline"),
        ],
    )
    b = _board(client)["AVGO"]
    by_source = {e["source"]: e for e in b["top_evidence"]}
    assert by_source["indicator.relstrength"]["label"] == "Relative strength"
    assert by_source["indicator.relstrength"]["plain"] == "Outperforming SPY by 4.3%"
    assert by_source["indicator.rsi"]["plain"] == "RSI 95 · overbought"
    assert by_source["indicator.alignment"]["plain"] == "Uptrend: +4.3% short, +8.5% long"
    assert by_source["news"]["plain"] == "News: “Chipmaker Faces Probe”"
    assert b["main_reason"] == "Outperforming SPY by 4.3%"
    assert b["counter_reason"] == "News: “Chipmaker Faces Probe”"
    assert b["caution"] is not None and "RSI 95" in b["caution"]


def test_board_with_nothing_in_favour(client, database_url):
    _belief(
        database_url,
        "CRM",
        0.0,
        direction="neutral",
        evidence=[("indicator.relstrength", -1.0, 0.5, "rel -3.50% vs SPY")],
    )
    b = _board(client)["CRM"]
    assert b["main_reason"] is None
    assert b["counter_reason"] == "Lagging SPY by 3.5%"
    assert b["caution"] is None


def test_overview_with_an_empty_cohort_is_unproven_on_the_review_baseline(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    body = client.get("/api/halabot/overview").json()
    assert body["available"] is True
    assert body["cohort"]["current"] == {
        "closed": 0,
        "wins": 0,
        "win_rate": None,
        "mean_return_pct": None,
    }
    assert body["cohort"]["started"] is None
    assert body["verdict"]["status"] == "unproven"
    assert body["verdict"]["label"] == "Unproven"
    assert body["baseline_win_rate"] == pytest.approx(0.35)
    assert body["baseline_measured"] is False
    assert "2026-10-06 review" in body["baseline_note"]
    assert body["calibration"]["status"] == "unknown"  # nothing scored in 24 h
    assert body["book"] == {
        "positions": [],
        "invested": 0,
        "cash": 1.0,
        "return_pct": None,
        "contribution_pct": 0,
    }
    assert body["engine"]["status"] == "missing"
    assert body["engine"]["names"] == 1
    assert body["llm"]["per_day_usd"] is None
    assert body["calendar"] == []
    assert body["entry_band"] == pytest.approx(0.35)


def test_overview_on_an_empty_database_is_not_available(client):
    body = client.get("/api/halabot/overview").json()
    assert body["available"] is False
    assert body["verdict"]["status"] == "unproven"


def test_overview_splits_the_current_cohort_from_earlier_outcomes(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    started = (NOW - timedelta(days=40)).isoformat()
    # 30 closed trades over 40 days in cohort 2: 12 wins.
    _outcomes(
        database_url,
        [{"a": "AVGO", "e": started, "r": 0.01 if i < 12 else -0.004, "c": 2} for i in range(30)],
    )
    _outcomes(database_url, [{"a": "AVGO", "e": started, "r": 0.01, "c": None} for _ in range(4)])
    body = client.get("/api/halabot/overview").json()
    cur, old = body["cohort"]["current"], body["cohort"]["earlier"]
    assert (cur["closed"], cur["wins"]) == (30, 12)
    assert cur["win_rate"] == pytest.approx(0.4)
    assert cur["mean_return_pct"] == pytest.approx((12 * 0.01 - 18 * 0.004) / 30, abs=1e-5)
    assert (old["closed"], old["wins"], old["win_rate"]) == (4, 4, 1.0)
    assert body["cohort"]["median_hold_s"] == 10800
    assert body["cohort"]["started"][:10] == started[:10]
    # No bars to measure the baseline on: the review's 35%, which 40% beats.
    assert body["baseline_measured"] is False
    assert body["verdict"]["status"] == "beating"
    assert body["verdict"]["label"] == "Beating random entry"
    assert body["calibration"]["samples"] == 30


def test_overview_measures_the_baseline_from_stored_bars(client, database_url):
    from halal_trader.web import belief_board

    belief_board._baseline_cache.clear()
    _belief(database_url, "AVGO", 0.62)
    _belief(database_url, "SPY", 0.5)  # the benchmark is not part of the baseline
    monday = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)  # the 09:00 ET bar
    _outcomes(database_url, [{"a": "AVGO", "e": monday.isoformat(), "r": 0.01, "c": 2}])
    price = 100.0
    for day in range(5):  # Mon-Fri, seven hourly bars a session, each 1% up
        for hour in range(7):
            _bar(database_url, "AVGO", price, monday + timedelta(days=day, hours=hour))
            _bar(database_url, "SPY", 500.0, monday + timedelta(days=day, hours=hour))
            price *= 1.01
    body = client.get("/api/halabot/overview").json()
    assert body["baseline_measured"] is True
    assert body["baseline_trades"] >= 30
    assert body["baseline_win_rate"] == pytest.approx(1.0)  # every random entry rose
    assert "held 3.0 h" in body["baseline_note"]
    assert body["verdict"]["status"] == "unproven"  # one closed trade proves nothing


def test_overview_young_cohort_stays_unproven_however_many_trades(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    started = (NOW - timedelta(days=5)).isoformat()
    _outcomes(database_url, [{"a": "AVGO", "e": started, "r": 0.01, "c": 2} for _ in range(40)])
    body = client.get("/api/halabot/overview").json()
    assert body["verdict"]["status"] == "unproven"
    assert body["verdict"]["age_days"] == 5


def test_overview_book_health_spend_and_calendar(client, database_url):
    cpi = NOW + timedelta(days=8)
    _belief(
        database_url,
        "AVGO",
        0.62,
        catalysts=[("CPI", cpi, 0.9), ("GDP", NOW + timedelta(days=20), 0.6)],
    )
    _belief(
        database_url,
        "AMD",
        0.55,
        catalysts=[("CPI", cpi, 0.9), ("FOMC", NOW - timedelta(days=2), 0.9)],
    )
    _screen(database_url, {"AVGO": "halal", "AMD": "not_halal"})
    _position(database_url, "AVGO", 0.10, 0.02, cohort=2)
    _position(database_url, "AMD", 0.05, -0.01, cohort=2)
    _sql(
        database_url,
        "INSERT INTO heartbeats (component, beat_at) "
        "VALUES ('shadow.process', now() - interval '30 seconds')",
    )
    _sql(
        database_url,
        "INSERT INTO llm_spend (day, consumer, calls, spent_usd) VALUES "
        "(CAST(:d AS date) - 1, 'shadow', 10, 0.20), (CAST(:d AS date) - 2, 'shadow', 10, 0.30), "
        "(CAST(:d AS date) - 1, 'stock', 10, 5.00), (CAST(:d AS date), 'shadow', 3, 0.05)",
        {"d": datetime.now(UTC).date()},
    )
    _sql(
        database_url,
        "INSERT INTO hb_conviction_score "
        "(asset, ts, raw_score, calibrated, features, belief_version) "
        "VALUES ('AVGO', now(), 0.6, 0.6, '{}', 1), ('AVGO', now(), 1.2, 1.0, '{}', 1)",
    )
    body = client.get("/api/halabot/overview").json()
    book = body["book"]
    assert [p["asset"] for p in book["positions"]] == ["AVGO", "AMD"]
    assert book["positions"][1]["strict"] == "not_halal"
    assert book["invested"] == pytest.approx(0.15)
    assert book["cash"] == pytest.approx(0.85)
    assert book["return_pct"] == pytest.approx((0.10 * 0.02 - 0.05 * 0.01) / 0.15, abs=1e-5)
    assert body["engine"]["status"] == "live"
    assert 0 <= body["engine"]["heartbeat_age_s"] < 120
    assert body["llm"]["per_day_usd"] == pytest.approx(0.25)  # the shadow's, complete days
    assert body["llm"]["today_usd"] == pytest.approx(0.05)
    # Identity output (the raw score clamped to [0, 1]) is the uncalibrated engine.
    assert body["calibration"]["status"] == "identity"
    assert body["calibration"]["scored_24h"] == 2
    cal = body["calendar"]
    assert [c["kind"] for c in cal] == ["CPI", "GDP"]  # once each; the past FOMC dropped
    assert cal[0]["names"] == 2 and cal[0]["plain"] == "inflation print"


def test_overview_reports_a_fitted_calibrator(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    _sql(
        database_url,
        "INSERT INTO hb_conviction_score "
        "(asset, ts, raw_score, calibrated, features, belief_version) "
        "VALUES ('AVGO', now(), 0.6, 0.41, '{}', 1)",
    )
    assert client.get("/api/halabot/overview").json()["calibration"]["status"] == "fitted"


def test_overview_stale_heartbeat(client, database_url):
    _belief(database_url, "AVGO", 0.62)
    _sql(
        database_url,
        "INSERT INTO heartbeats (component, beat_at) "
        "VALUES ('shadow.process', now() - interval '1 hour')",
    )
    assert client.get("/api/halabot/overview").json()["engine"]["status"] == "stale"


def test_decisions_carry_a_plain_reason_and_the_session_flag(client, database_url):
    # Tue 6 Oct 2026: 14:17 UTC is 10:17 ET (open), 11:48 UTC is 07:48 ET (pre-market).
    _proposal(
        database_url,
        "AVGO",
        datetime(2026, 10, 6, 14, 17, tzinfo=UTC),
        {
            "side": "buy",
            "current_weight": 0.0,
            "target_weight": 0.076,
            "weight_delta": 0.076,
            "conviction_raw": 0.4,
            "reason": "conviction",
        },
    )
    _proposal(
        database_url,
        "AVGO",
        datetime(2026, 10, 6, 11, 48, tzinfo=UTC),
        {
            "side": "sell",
            "current_weight": 0.078,
            "target_weight": 0.0,
            "weight_delta": -0.078,
            "conviction_raw": 0.08,
            "reason": "conviction",
        },
    )
    _proposal(
        database_url,
        "NVDA",
        datetime(2026, 10, 6, 15, 0, tzinfo=UTC),
        {
            "side": "sell",
            "current_weight": 0.05,
            "target_weight": 0.0,
            "weight_delta": -0.05,
            "reason": "price_break",
            "forced_exit": True,
        },
    )
    rows = client.get("/api/halabot/decisions?limit=10").json()
    by_time = {r["ts"][11:16]: r for r in rows}
    assert by_time["14:17"]["plain"] == "conviction crossed 35%"
    assert by_time["14:17"]["outside_session"] is False
    assert by_time["11:48"]["plain"] == "conviction fell below 15%"
    assert by_time["11:48"]["outside_session"] is True
    assert by_time["15:00"]["plain"] == "forced exit: price broke its invalidation level"
    only = client.get("/api/halabot/decisions?asset=avgo").json()
    assert {r["asset"] for r in only} == {"AVGO"} and len(only) == 2
