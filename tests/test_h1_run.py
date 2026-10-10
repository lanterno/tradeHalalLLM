"""H1 end to end on a synthetic world: register, Stage A, train, validation, verdict,
the implementability trial and the sensitivities (events/h1.py).

One halal name (AAA, "Acme") gets an analyst downgrade before the open on three
train and three validation sessions; its minute bars fall 5% from the previous
close, go quiet above the low, reclaim the anchored VWAP and reach the 50%
target, so every story enters at the 10:23 bar (96.00) and exits at the 11:02
bar. BBB ("Bolt") is downgraded too but no screen holds it: it never starts.
SPY is flat. Every number is synthetic; the count rule is relaxed so a handful
of entries can be eligible.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.clock import SIP_DELAYED
from halabot.playbooks.loader import Window, WindowGuard, WindowLocked, WindowUnlock
from halal_trader.db.repos.quant_trials import config_hash
from halal_trader.events import h1
from halal_trader.events.h1 import (
    CELLS,
    Carrier,
    Check,
    CodeState,
    H1Locked,
    Preconditions,
    StoriesStale,
)
from halal_trader.events.stories import build_range
from halal_trader.market_hours import (
    MARKET_TZ,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)
from tests._stories import add_aliases, news_row, store
from tests.halabot.playbooks._seed import mark_done, seed_bars
from tests.halabot.playbooks._support import session_bars

ID, MD3 = CELLS
TRAIN = (date(2017, 3, 7), date(2018, 6, 5), date(2019, 9, 10))
VALIDATION = (date(2022, 3, 8), date(2023, 6, 6), date(2024, 9, 10))
EXIT = dict(zip(TRAIN + VALIDATION, (97.6, 97.9, 97.7, 97.6, 97.9, 97.7)))
ENTRY = 96.0
COST = 7.0  # study.cost_bps of rank 0, one way
HISTORY = (date(2015, 6, 1), date(2025, 1, 10))
LENIENT = h1.CountRule(validation_per_year=1.0, validation_dates=3, train_n=3, train_dates=3)
EPS = 1e-12


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), MARKET_TZ)


def _path(day: date, n: int) -> list[date]:
    out = [day]
    while len(out) < n:
        out.append(next_trading_day(out[-1]))
    return out


def _bounce_rows(exit_px: float) -> dict[tuple[int, int], tuple[float, ...]]:
    """S's bars from the open to 10:59: the fall, the quiet, the reclaim; ``exit_px`` after."""
    rows: dict[tuple[int, int], tuple[float, ...]] = {}
    for k in range(30):  # 09:30-09:59: 100.00 down to 95.65
        p = 100.0 - 0.15 * k
        rows[(9, 30 + k)] = (p, p, p, p, 100.0, p)
    rows[(10, 0)] = (95.65, 95.65, 95.0, 95.2, 100.0, 95.2)  # the low, L* = 95
    for m in range(1, 21):  # quiet above the low: armed at the 10:20 bar
        rows[(10, m)] = (95.3, 95.3, 95.3, 95.3, 10_000.0, 95.3)
    rows[(10, 21)] = (95.3, 95.8, 95.3, 95.8, 10_000.0, 95.6)  # reclaims the AVWAP: the entry
    for m in range(22, 60):
        rows[(10, m)] = (ENTRY, ENTRY, ENTRY, ENTRY, 10_000.0, ENTRY)
    return rows  # from 11:00 the session's flat price, exit_px >= TGT = 97.5


async def _daily(engine: AsyncEngine) -> None:
    eves = {previous_trading_day(d) for d in EXIT}
    rows: list[dict[str, Any]] = []
    d, i = HISTORY[0], 0
    while d <= HISTORY[1]:
        if is_trading_day(d):
            aaa = 100.0 if d in eves else 100.0 + (i % 2)
            spy = 200.0 if d in eves else 200.0 + 2.0 * ((i // 2) % 2)  # beta needs SPY to move
            for adjustment in ("raw", "all"):
                rows.append({"s": "SPY", "d": d, "a": adjustment, "p": spy})
                rows.append({"s": "AAA", "d": d, "a": adjustment, "p": aaa})
            i += 1
        d += timedelta(days=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, volume, "
                "fetched_at) VALUES (:s, :d, :a, :p, :p, :p, :p, 1e6, now())"
            ),
            rows,
        )


async def _screens_and_liquidity(engine: AsyncEngine) -> None:
    quarter_ends = []
    for year in range(2016, 2025):
        quarter_ends += [date(year, m, 30 if m in (6, 9) else 31) for m in (3, 6, 9, 12)]
    months = []
    d = date(2015, 1, 1)
    while d <= date(2024, 12, 1):
        months.append(d)
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO halal_screen_results (as_of, symbol, cik, sic_description, verdict, "
                "reasons, metrics, method, screened_at) VALUES (:a, 'AAA', 1, "
                "'SERVICES-PREPACKAGED SOFTWARE', 'halal', '[]', '{}', 'v12', now())"
            ),
            [{"a": q} for q in quarter_ends],
        )
        await conn.execute(
            text(
                "INSERT INTO monthly_bars (symbol, month, close, volume) "
                "VALUES ('AAA', :m, 100, 1e6)"
            ),
            [{"m": m} for m in months],
        )


async def _news(engine: AsyncEngine) -> None:
    rows = [
        news_row(n, "AAA", ny(day, 8), "Morgan Stanley Downgrades Acme to Equal-Weight")
        for n, day in enumerate(sorted(EXIT), 1)
    ]
    rows.append(news_row(99, "BBB", ny(TRAIN[0], 8), "Barclays Downgrades Bolt to Underweight"))
    await store(engine, rows)
    await add_aliases(
        engine,
        [
            ("AAA", "Acme", "name"),
            ("AAA", "AAA", "ticker"),
            ("BBB", "Bolt", "name"),
            ("BBB", "BBB", "ticker"),
        ],
    )
    await build_range(engine, start=date(2016, 10, 3), end=date(2024, 12, 31), force=True)


async def _minutes(engine: AsyncEngine) -> None:
    units: set[tuple[str, date]] = set()
    for day, exit_px in EXIT.items():
        for k, d in enumerate(_path(day, 3)):
            rows = _bounce_rows(exit_px) if k == 0 else None
            await seed_bars(
                engine, "AAA", session_bars(d, price=exit_px, rows=rows, volume=10_000.0)
            )
            await seed_bars(engine, "SPY", session_bars(d, price=200.0, volume=1e5))
            units |= {("AAA", d), ("SPY", d)}
    await mark_done(engine, sorted(units))


FROZEN = CodeState("c" * 40, (h1.TAG,), False)  # the tagged, clean checkout registered


def _report() -> Preconditions:
    data = {"commit": FROZEN.commit, "tags": list(FROZEN.tags), "dirty": FROZEN.dirty}
    return Preconditions(
        tuple(Check(i, True, "ok", data if i == "C0" else {}) for i in h1.REQUIRED_CHECKS)
    )


@pytest.fixture(autouse=True)
def frozen(monkeypatch: pytest.MonkeyPatch) -> list[CodeState]:
    """Every step sees the registered checkout unless a test changes ``frozen[0]``."""
    state = [FROZEN]
    monkeypatch.setattr(h1, "code_state", lambda root=None: state[0])
    return state


@pytest.fixture
async def world(engine: AsyncEngine) -> AsyncEngine:
    await _daily(engine)
    await _screens_and_liquidity(engine)
    await _news(engine)
    await _minutes(engine)
    return engine


async def _rows(engine: AsyncEngine, name: str, kind: str | None = None) -> list[Any]:
    sql = 'SELECT id, kind, config, config_hash, "window", metrics, verdict FROM quant_trials '
    sql += "WHERE name = :n" + (" AND kind = :k" if kind else "") + " ORDER BY id"
    async with engine.connect() as conn:
        return list((await conn.execute(text(sql), {"n": name, "k": kind})).all())


def _r_net(exit_px: float) -> float:
    return exit_px / ENTRY - 1.0 - 2 * COST / 1e4


async def test_h1_runs_in_order_and_records_every_row(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = world
    monkeypatch.setattr(h1, "COUNT_RULE", LENIENT)
    with pytest.raises(H1Locked, match="not registered"):
        await h1.stage_a(engine, workers=1)
    reg_id = await h1.register(engine, report=_report())
    with pytest.raises(H1Locked, match="Stage A has not run"):
        await h1.run_window(engine, "train", workers=1)

    stage = await h1.stage_a(engine, workers=1)
    assert stage.budget_ok and stage.eligible == CELLS and stage.verdict == "pass"
    for window, days in (("train", TRAIN), ("validation", VALIDATION)):
        for cell in CELLS:
            c = stage.counts[(window, cell.key)]
            assert (c.nsn, c.eligible, c.entries, c.dates) == (
                len(days) + (window == "train"),
                3,
                3,
                3,
            )
            assert c.reasons == ({"ok": 3, "not_halal": 1} if window == "train" else {"ok": 3})
            assert c.triggered == c.armed == 3 and c.data_skips == 0 and c.skips == {}
    with pytest.raises(H1Locked, match="data is frozen"):
        await h1.stage_a(engine, workers=1)
    with pytest.raises(H1Locked, match="train has not run"):
        await h1.run_window(engine, "validation", workers=1)
    with pytest.raises(ValueError, match="runs"):
        await h1.run_window(engine, "train", [ID], workers=1)

    train = await h1.run_window(engine, "train", workers=1)
    assert set(train) == set(CELLS)
    for cell, ws in train.items():
        expected = [_r_net(EXIT[d]) for d in TRAIN]
        assert ws.n == 3 and ws.dates == 3 and ws.status == "pass", ws.as_dict()["tests"]
        assert abs(ws.mean - sum(expected) / 3) < EPS
        assert ws.t1 and ws.t2 and ws.t3 and ws.t4 and ws.t5 and ws.t6
        assert (ws.nw is not None) == (cell is MD3)
        assert ws.trial_id is not None and ws.dsr is not None and ws.dsr_plus6 is not None
        assert ws.dsr_plus6 <= ws.dsr
    with pytest.raises(H1Locked, match="amendment"):
        await h1.run_window(engine, "train", workers=1)

    validation = await h1.run_window(engine, "validation", workers=1)
    assert all(ws.status == "pass" for ws in validation.values())
    assert abs(validation[ID].mean - sum(_r_net(EXIT[d]) for d in VALIDATION) / 3) < EPS

    assert await h1.verdict(engine) == "pass"
    with pytest.raises(H1Locked, match="recorded already"):
        await h1.verdict(engine)

    reg = await h1.registration(engine)
    assert reg.id == reg_id
    (verdict_row,) = await _rows(engine, h1.NAME, "verdict")
    assert (
        verdict_row.verdict == "pass" and verdict_row.config_hash == reg.hash
    )  # opens the holdout
    (stage_row,) = await _rows(engine, h1.NAME, "stage-a")
    assert stage_row.verdict == "pass" and stage_row.metrics["eligible"] == [ID.key, MD3.key]
    for cell in CELLS:
        trials = await _rows(engine, "research." + cell.strategy, "backtest")
        assert [t.metrics["window_role"] for t in trials] == ["train", "validation"]
        assert {t.config_hash for t in trials} == {config_hash(h1.trial_config(cell, reg.hash))}
        assert all(t.window.endswith("vs SPY (exposure-matched)") for t in trials)
        assert all(t.metrics["pass"] and t.metrics["trades"] == 3 for t in trials)
        assert trials[1].metrics["n_trials"] == 2  # one trial per cell, both windows in it
    async with engine.connect() as conn:
        runs = (
            await conn.execute(
                text("SELECT stop_at, count(*) AS n FROM hb_playbook_run GROUP BY stop_at")
            )
        ).all()
        trades = await conn.scalar(text("SELECT count(*) FROM hb_playbook_trade"))
    assert {r.stop_at: r.n for r in runs} == {"entry": 4, "end": 4}
    assert trades == 12

    # The implementability trial: the cells that passed, on the delayed feed.
    impl = await h1.implementability(engine, workers=1)
    assert set(impl) == {"train", "validation"}
    for cell in CELLS:
        t = impl["train"][cell]
        assert t.status == "pass" and t.n == 3
        assert abs(t.mean - train[cell].mean) < EPS  # the same fills, 17 minutes later
    (impl_row,) = await _rows(engine, h1.NAME, "implementability")
    assert impl_row.verdict == "pass"
    delayed = await _rows(engine, "research." + ID.strategy, "backtest")
    assert [r.config["feed"] for r in delayed] == [
        "sip-rt",
        "sip-rt",
        SIP_DELAYED.name,
        SIP_DELAYED.name,
    ]
    with pytest.raises(H1Locked, match="run already"):
        await h1.implementability(engine, workers=1)

    # The sensitivities: every one, each cell, each window; never a trial.
    sens = await h1.sensitivities(engine, workers=1)
    assert set(sens) == set(h1.SENSITIVITIES)
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT name, config, metrics FROM quant_trials WHERE kind = 'sensitivity' "
                    "ORDER BY id"
                )
            )
        ).all()
    assert len(rows) == len(h1.SENSITIVITIES) * 2 * 2
    assert {r.name.removeprefix(h1.SENS_PREFIX) for r in rows} == set(h1.SENSITIVITIES)
    assert all(r.config["prereg"] == reg.hash and "active_sr_period" not in r.metrics for r in rows)
    lag = sens["news_lag_1200s"][f"train:{ID.key}"]
    assert lag["trades"] == 3 and abs(lag["mean_trade"] - train[ID].mean) < EPS
    assert sens["cost_x2"][f"train:{ID.key}"]["mean_trade"] == pytest.approx(
        train[ID].mean - 2 * COST / 1e4
    )
    assert sens["tech"][f"validation:{MD3.key}"]["tech"]["trades"] == 3

    # A window run again is an amendment; the verdict then waits for validation.
    again = await h1.run_window(engine, "train", workers=1, amend="loader fix (test)")
    assert all(ws.status == "pass" for ws in again.values())
    (amendment,) = await _rows(engine, h1.NAME, "amendment")
    assert amendment.metrics["reason"] == "loader fix (test)"
    assert await _holdout_opens(engine, reg) is False  # the pass no longer stands
    with pytest.raises(H1Locked, match="rerun validation"):
        await h1.verdict(engine)
    with pytest.raises(H1Locked, match="ran after the verdict"):
        await h1.sensitivities(engine, workers=1)  # they read the rows the verdict cites
    await h1.run_window(engine, "validation", workers=1, amend="loader fix (test)")
    assert await h1.verdict(engine) == "pass"
    assert await _holdout_opens(engine, reg) is True
    second = (await _rows(engine, h1.NAME, "verdict"))[-1]
    windows = await _rows(engine, h1.NAME, "window")
    assert [r["row"] for r in second.metrics["replaced"]] == [w.id for w in windows[:2]]
    assert second.metrics["replaced"][0]["cells"][ID.key]["status"] == "pass"  # the old results
    assert [a["reason"] for a in second.metrics["amendments"]] == ["loader fix (test)"] * 2
    assert [v["id"] for v in second.metrics["previous_verdicts"]] == [verdict_row.id]


async def _holdout_opens(engine: AsyncEngine, reg: h1.Registration) -> bool:
    unlock = WindowUnlock(prereg_id=reg.id, config_hash=reg.hash, holdout=True)
    guard = WindowGuard(window=Window.HOLDOUT, window_end=date(2025, 11, 28), unlock=unlock)
    try:
        await guard.verify(engine)
    except WindowLocked:
        return False
    return guard.allows("AAA", date(2025, 2, 3))


async def _through_validation(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(h1, "COUNT_RULE", LENIENT)
    reg_id = await h1.register(engine, report=_report())
    await h1.stage_a(engine, workers=1)
    await h1.run_window(engine, "train", workers=1)
    await h1.run_window(engine, "validation", workers=1)
    return reg_id


async def test_an_amended_train_without_passers_is_decided_and_closes_the_holdout(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = world
    await _through_validation(engine, monkeypatch)
    assert await h1.verdict(engine) == "pass"
    reg = await h1.registration(engine)
    assert await _holdout_opens(engine, reg)
    old_train, old_validation = await _rows(engine, h1.NAME, "window")

    real_judge = h1.judge_window

    def fixed(results: Any) -> dict[Any, Any]:  # after a simulator fix no cell passes train
        return {c: replace(ws, status="fail") for c, ws in real_judge(results).items()}

    monkeypatch.setattr(h1, "judge_window", fixed)
    amended = await h1.run_window(engine, "train", workers=1, amend="a simulator fix (test)")
    assert all(ws.status == "fail" for ws in amended.values())
    assert not await _holdout_opens(engine, reg)  # the old pass stands no more
    with pytest.raises(H1Locked, match="ran after the verdict"):
        await h1.implementability(engine, workers=1)
    with pytest.raises(H1Locked, match="no cell runs on validation"):
        await h1.run_window(engine, "validation", workers=1, amend="nothing to run")

    assert await h1.verdict(engine) == "fail"  # decided on train alone: no deadlock
    final = (await _rows(engine, h1.NAME, "verdict"))[-1]
    assert final.verdict == "fail" and final.metrics["validation_row"] is None
    assert final.metrics["validation_ignored"] == old_validation.id
    replaced = {r["row"]: r for r in final.metrics["replaced"]}
    assert set(replaced) == {old_train.id, old_validation.id}
    assert replaced[old_train.id]["cells"][ID.key]["status"] == "pass"  # old and new reported
    assert final.metrics["train"][ID.key]["status"] == "fail"
    (amendment,) = final.metrics["amendments"]
    assert amendment["replaces"] == old_train.id and amendment["window_role"] == "train"
    assert not await _holdout_opens(engine, reg)  # the latest verdict fails
    assert await h1.implementability(engine, workers=1) == {}  # no cell passed H1
    with pytest.raises(H1Locked, match="recorded already"):
        await h1.verdict(engine)


async def test_changed_code_runs_only_as_an_amendment_and_runs_record_their_head(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch, frozen: list[CodeState]
) -> None:
    engine = world
    monkeypatch.setattr(h1, "COUNT_RULE", LENIENT)
    await h1.register(engine, report=_report())
    real = h1.file_shas()
    fixed = {**real, "halabot.playbooks.sim": "0123456789ab"}
    monkeypatch.setattr(h1, "file_shas", lambda: fixed)  # a simulator fix after registering
    with pytest.raises(H1Locked, match=r"code differs.*--amend"):
        await h1.stage_a(engine, workers=1)
    assert await _rows(engine, h1.NAME, "stage-a") == []  # refused before anything ran
    await h1.stage_a(engine, workers=1, amend="sim fix (test)")
    (amendment,) = await _rows(engine, h1.NAME, "amendment")
    assert amendment.metrics["step"] == "stage-a"
    assert amendment.metrics["code_diff"] == {
        "files": {"halabot.playbooks.sim": [real["halabot.playbooks.sim"], "0123456789ab"]}
    }
    assert amendment.metrics["files"] == fixed  # the code later steps compare with
    await h1.run_window(engine, "train", workers=1)  # the same code: no amendment

    frozen[0] = CodeState("d" * 40, (), True)  # another, modified checkout
    with pytest.raises(H1Locked, match="code differs"):
        await h1.run_window(engine, "validation", workers=1)
    await h1.run_window(engine, "validation", workers=1, amend="unrelated commit (test)")
    last = (await _rows(engine, h1.NAME, "amendment"))[-1]
    assert last.metrics["code_diff"] == {"commit": ["c" * 40, "d" * 40], "dirty": [False, True]}
    assert last.metrics["window_role"] == "validation"
    assert await h1.verdict(engine) == "pass"  # the amended code is the code in force
    async with engine.connect() as conn:
        shas = (
            await conn.execute(
                text(
                    "SELECT config->>'role' AS role, code_sha FROM hb_playbook_run "
                    "ORDER BY created_at"
                )
            )
        ).all()
    assert {(r.role, r.code_sha) for r in shas} == {
        ("stage-a", "c" * 40),
        ("train", "c" * 40),
        ("validation", "d" * 40 + "-dirty"),
    }


async def test_data_changed_since_stage_a_runs_only_as_an_amendment(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = world
    monkeypatch.setattr(h1, "COUNT_RULE", LENIENT)
    await h1.register(engine, report=_report())
    await h1.stage_a(engine, workers=1)
    (stage,) = await _rows(engine, h1.NAME, "stage-a")
    frozen_train = stage.metrics["data"]["train"]
    assert frozen_train["candidates"] == 4  # three AAA stories and BBB's
    # Units: AAA and SPY on S and S+1 (ID's spare), S..S+3 (MD3 and its spare).
    assert frozen_train["units"] == 3 * 2 * 4 and frozen_train["done"] == 3 * 2 * 3

    spare = _path(TRAIN[0], 4)[3]
    await mark_done(engine, [("AAA", spare)])  # a unit fetched after Stage A
    with pytest.raises(H1Locked, match="data differs"):
        await h1.run_window(engine, "train", workers=1)
    assert await _rows(engine, h1.NAME, "window") == []
    await h1.run_window(engine, "train", workers=1, amend="spare session fetched (test)")
    (amendment,) = await _rows(engine, h1.NAME, "amendment")
    diff = amendment.metrics["data_diff"]["train"]
    assert set(diff) == {"done", "done_sha"} and diff["done"] == [18, 19]
    assert amendment.metrics["data"]["train"]["done"] == 19  # what later steps compare with
    await h1.run_window(engine, "validation", workers=1)  # its own data did not change

    async with engine.begin() as conn:  # a screen re-run makes AAA not halal from 2018
        await conn.execute(
            text(
                "UPDATE halal_screen_results SET verdict = 'not_halal' "
                "WHERE symbol = 'AAA' AND as_of >= '2018-01-01' AND as_of < '2022-01-01'"
            )
        )
    await h1.verdict(engine)
    with pytest.raises(H1Locked, match="data differs"):
        await h1.sensitivities(engine, workers=1)


async def test_an_unfinished_window_run_reruns_only_as_an_amendment(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = world
    monkeypatch.setattr(h1, "COUNT_RULE", LENIENT)
    await h1.register(engine, report=_report())
    await h1.stage_a(engine, workers=1)
    real = h1._record

    async def crash(engine_: AsyncEngine, **kw: Any) -> int:
        if kw["kind"] == "window":
            raise RuntimeError("the machine died (test)")
        return await real(engine_, **kw)

    monkeypatch.setattr(h1, "_record", crash)
    with pytest.raises(RuntimeError, match="died"):
        await h1.run_window(engine, "train", workers=1)
    monkeypatch.setattr(h1, "_record", real)
    async with engine.connect() as conn:
        partial = [
            int(r.id)
            for r in await conn.execute(
                text("SELECT id FROM quant_trials WHERE kind = 'backtest' ORDER BY id")
            )
        ]
    assert len(partial) == 2  # both cells' trials, but no window row
    with pytest.raises(H1Locked, match="unfinished run"):
        await h1.run_window(engine, "train", workers=1)
    await h1.run_window(engine, "train", workers=1, amend="rerun after a crash (test)")
    (amendment,) = await _rows(engine, h1.NAME, "amendment")
    assert amendment.metrics["partial"] == partial and "replaces" not in amendment.metrics
    await h1.run_window(engine, "validation", workers=1)  # validation is untouched


async def test_too_few_entries_fail_h1_at_stage_a(world: AsyncEngine) -> None:
    engine = world
    await h1.register(engine, report=_report())
    stage = await h1.stage_a(engine, workers=1)
    assert stage.budget_ok and stage.eligible == () and stage.verdict == "fail: insufficient events"
    with pytest.raises(H1Locked, match="no cell runs on train"):
        await h1.run_window(engine, "train", workers=1)
    assert await h1.verdict(engine) == "fail"
    (row,) = await _rows(engine, h1.NAME, "verdict")
    assert row.metrics["reason"] == "insufficient events"
    async with engine.connect() as conn:
        backtests = await conn.scalar(
            text("SELECT count(*) FROM quant_trials WHERE kind = 'backtest'")
        )
    assert backtests == 0  # no return was computed
    assert await h1.implementability(engine, workers=1) == {}
    assert await h1.sensitivities(engine, workers=1) == {}  # nothing ran, nothing to vary


async def test_the_window_starts_eligible_nsn_stories_only(world: AsyncEngine) -> None:
    engine = world
    data = await h1.load_window(engine, "train")
    assert data.candidates == {f"AAA:{d}" for d in TRAIN} | {f"BBB:{TRAIN[0]}"}
    inputs = h1.run_inputs(data, ID)
    started = [s for s in inputs.stories if not isinstance(s, Carrier)]
    assert sorted(s.story_id for s in started) == sorted(f"AAA:{d}" for d in TRAIN)
    assert sorted(inputs.context) == sorted(s.story_id for s in started)
    (bolt,) = [s for s in inputs.stories if s.symbol == "BBB"]
    assert isinstance(bolt, Carrier) and inputs.reasons == {"ok": 3, "not_halal": 1}
    pre, elig = inputs.context[f"AAA:{TRAIN[0]}"]
    assert pre is not None and pre.prev_close_s == 100.0 and elig.liquidity_rank == 0
    lagged = await h1.load_window(engine, "train", lag=timedelta(seconds=60))
    first = next(s for s in lagged.stories if s.story_id == f"AAA:{TRAIN[0]}")
    assert first.nsn_at(ny(TRAIN[0], 15)) == ny(TRAIN[0], 8, 1)

    async with engine.begin() as conn:  # the table no longer says what the code builds
        await conn.execute(
            text(
                "UPDATE news_stories SET nsn_at = nsn_at + interval '1 minute' WHERE story_id = :i"
            ),
            {"i": f"AAA:{TRAIN[1]}"},
        )
    with pytest.raises(StoriesStale, match=f"AAA:{TRAIN[1]}"):
        await h1.load_window(engine, "train")


async def test_a_live_eligible_story_blocks_the_next_while_carriers_never_do(
    world: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AAA: S's story enters; S+1 brings a product item (a carrier); S+2 another downgrade.

    Stage A assumes the full hold, so in MD3 S's playbook is live through S+2:
    S+2's story is ``blocked_open``. The carrier in between is passed to the
    simulator (its item reaches the live playbook) but starts nothing, blocks
    nothing and has no outcome. In ID, S's playbook is done by S's close.
    """
    engine = world
    s0, s1, s2 = _path(TRAIN[0], 3)
    await store(
        engine,
        [
            news_row(201, "AAA", ny(s1, 8), "Acme Announces New Product Line"),
            news_row(202, "AAA", ny(s2, 8), "Barclays Downgrades Acme to Underweight"),
        ],
    )
    await build_range(engine, start=date(2016, 10, 3), end=date(2024, 12, 31), force=True)
    data = await h1.load_window(engine, "train")
    assert f"AAA:{s2}" in data.candidates and f"AAA:{s1}" not in data.candidates
    inputs = h1.run_inputs(data, MD3)
    carrier = next(s for s in inputs.stories if s.story_id == f"AAA:{s1}")
    assert isinstance(carrier, Carrier)
    assert f"AAA:{s2}" in inputs.context  # eligible: it starts, and is blocked by S's

    monkeypatch.setattr(h1, "COUNT_RULE", LENIENT)
    await h1.register(engine, report=_report())
    stage = await h1.stage_a(engine, workers=1)
    md3, id_ = stage.counts[("train", MD3.key)], stage.counts[("train", ID.key)]
    assert (md3.eligible, md3.blocked_open, md3.entries) == (4, 1, 3)
    assert (id_.eligible, id_.blocked_open) == (4, 0)
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT r.cell, s.story_id, s.terminal_state, s.reason "
                    "FROM hb_playbook_story s JOIN hb_playbook_run r ON r.run_id = s.run_id "
                    "WHERE r.window = 'train' AND s.symbol = 'AAA' AND s.session BETWEEN :a AND :b"
                ),
                {"a": s0, "b": s2},
            )
        ).all()
    by_cell = {(r.cell, r.story_id): (r.terminal_state, r.reason) for r in rows}
    assert by_cell[(MD3.key, f"AAA:{s2}")] == ("DISMISSED", "blocked_open")
    assert by_cell[(ID.key, f"AAA:{s2}")] != ("DISMISSED", "blocked_open")
    assert not any(sid == f"AAA:{s1}" for _, sid in by_cell)  # the carrier has no outcome
