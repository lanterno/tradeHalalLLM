"""The replay of the day-trader's scored trips under three rules (research/trip_replay.py)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.research import trip_replay as tr
from halal_trader.trading import decision_quality as dq

DAY = date(2026, 10, 14)
OPEN = datetime(2026, 10, 14, 13, 30, tzinfo=UTC)  # 09:30 New York (EDT)


def _bar(at: datetime, close: float) -> dict[str, object]:
    return {"t": at.isoformat().replace("+00:00", "Z"), "c": close}


def _hours(*closes: float, end: datetime = OPEN) -> list[tuple[datetime, float]]:
    """Hourly bars ending an hour apart, the last at ``end``."""
    n = len(closes)
    return [(end - timedelta(hours=n - 1 - i), c) for i, c in enumerate(closes)]


def _score(day: int, edge: float, symbol: str = "AAA") -> dq.Score:
    at = OPEN + timedelta(days=day, minutes=30)
    trip = dq.Trip(symbol, at, at + timedelta(hours=1), 100.0, 101.0, 10, "llm_sell", "cycle")
    return dq.Score(trip, edge, 0.0, 0.0)


def test_hourly_bars_are_regular_session_clock_hours() -> None:
    raw = [
        _bar(OPEN - timedelta(minutes=15), 9.0),  # 09:15: before the open
        _bar(OPEN + timedelta(minutes=15), 2.0),  # out of order: sorted by the bar's start
        _bar(OPEN, 1.0),
        _bar(OPEN + timedelta(minutes=30), 3.0),
        _bar(OPEN + timedelta(minutes=45), 4.0),
        _bar(OPEN + timedelta(hours=6, minutes=30), 9.0),  # 16:00: after the close
    ]

    assert tr.hourly_closes(raw) == [
        (OPEN + timedelta(minutes=30), 2.0),  # 09:30-10:00
        (OPEN + timedelta(minutes=90), 4.0),  # 10:00-11:00
    ]


def test_a_bar_counts_once_it_has_ended() -> None:
    bars = _hours(1.0, 2.0)

    assert tr.completed(bars, OPEN) == [1.0, 2.0]
    assert tr.completed(bars, OPEN - timedelta(seconds=1)) == [1.0]


def test_the_trend_gate_reads_spy_against_its_50_hour_mean() -> None:
    assert tr.below_trend(_hours(*[100.0] * 49, 99.0), OPEN) is True
    assert tr.below_trend(_hours(*[100.0] * 49, 101.0), OPEN) is False
    assert tr.below_trend(_hours(*[100.0] * 49), OPEN) is None  # too few bars


def test_strength_is_the_names_return_minus_spys_over_20_hours() -> None:
    name = _hours(100.0, *[105.0] * 18, 110.0)
    spy = _hours(100.0, *[100.0] * 18, 105.0)

    assert tr.strength(name, spy, OPEN) == pytest.approx(0.05)
    assert tr.strength(name[1:], spy, OPEN) is None


def test_a_plans_buys_are_read_with_their_confidence() -> None:
    plan = {
        "decisions": [
            {"action": "buy", "symbol": "aaa", "confidence": 0.8, "reasoning": "breakout"},
            {"action": "sell", "symbol": "BBB", "confidence": 0.9, "reasoning": "x"},
            {"action": "buy", "symbol": "CCC", "reasoning": "no confidence"},
        ]
    }

    (d,) = tr.buys_in(OPEN, json.dumps(plan))
    assert (d.symbol, d.reasoning, d.confidence) == ("AAA", "breakout", 0.8)
    assert tr.buys_in(OPEN, "timeout talking to the model") == []
    assert tr.buys_in(OPEN, None) == []


def test_a_buy_takes_the_confidence_of_the_latest_matching_decision() -> None:
    def at(minutes: int, reasoning: str, confidence: float) -> tr.Decision:
        return tr.Decision(OPEN + timedelta(minutes=minutes), "AAA", reasoning, confidence)

    decisions = [
        at(-40, "breakout", 0.3),  # too early
        at(-20, "breakout", 0.6),
        at(-5, "breakout", 0.9),
        at(-1, "another reason", 0.1),
        at(1, "breakout", 0.2),  # after the trade
    ]

    assert tr.confidence_of("AAA", "breakout", OPEN, decisions) == 0.9
    assert tr.confidence_of("BBB", "breakout", OPEN, decisions) is None


def test_welch_compares_the_two_groups_session_means() -> None:
    diff, t = tr.welch([0.0, 0.01, 0.02], [-0.03, -0.02, -0.01])

    assert diff == pytest.approx(-0.03)
    assert t == pytest.approx(-0.03 / (2 * 0.0001 / 3) ** 0.5)
    assert tr.welch([0.0], [-0.01]) == (pytest.approx(-0.01), None)
    assert tr.welch([], [0.01]) == (None, None)


def test_a_rule_passes_when_the_buys_it_drops_were_worse() -> None:
    judged = [
        (_score(0, 0.01), False),
        (_score(0, -0.02), True),
        (_score(1, 0.02), False),
        (_score(1, -0.01), True),
        (_score(2, 0.0), False),
        (_score(2, -0.03), True),
        (_score(2, 0.05), None),
    ]

    sp = tr.split("market_trend", judged)

    assert (sp.kept.trips, sp.dropped.trips, sp.unjudged) == (3, 3, 1)
    assert sp.difference == pytest.approx(-0.03)
    assert sp.verdict == "pass"
    flipped = tr.split("market_trend", [(s, None if d is None else not d) for s, d in judged])
    assert flipped.verdict == "no pass"


def test_the_confidence_rule_splits_at_the_median_and_the_others_need_bars() -> None:
    scores = [_score(0, 0.0, "A"), _score(0, 0.0, "B"), _score(1, 0.0, "C"), _score(1, 0.0, "D")]
    confidences = {
        (s.trip.symbol, s.trip.filled_at): c
        for s, c in zip(scores, (0.5, 0.6, 0.7, 0.8), strict=True)
    }

    trend, strength, confidence = tr.judge(scores, {}, confidences)

    assert (trend.unjudged, strength.unjudged, confidence.unjudged) == (4, 4, 0)
    assert (confidence.kept.trips, confidence.dropped.trips) == (2, 2)


async def test_each_buy_is_matched_to_the_decision_that_placed_it(engine: AsyncEngine) -> None:
    filled = OPEN + timedelta(minutes=31)
    plan = {"decisions": [{"action": "buy", "symbol": "AAA", "confidence": 0.8, "reasoning": "r"}]}
    async with engine.begin() as conn:
        for symbol in ("AAA", "BBB"):
            await conn.execute(
                text(
                    "INSERT INTO trades (symbol, side, quantity, status, timestamp, filled_at, "
                    "llm_reasoning) VALUES (:s, 'buy', 1, 'filled', :t, :t, 'r')"
                ),
                {"s": symbol, "t": filled},
            )
        await conn.execute(
            text(
                "INSERT INTO llm_decisions (timestamp, provider, model, raw_response) "
                "VALUES (:t, 'glm', 'glm-5.2', :raw)"
            ),
            {"t": OPEN + timedelta(minutes=30), "raw": json.dumps(plan)},
        )

    assert await tr._confidences(engine) == {("AAA", filled): 0.8, ("BBB", filled): None}


async def test_the_rules_are_registered_first_and_judged_once(engine: AsyncEngine) -> None:
    ids = await tr.register(engine)

    assert sorted(ids) == sorted(tr.RULES)
    assert await tr.register(engine) == ids  # registered once
    report = dq.Report(tr.START, tr.END)
    await tr.record(engine, report, [tr.split(rule, []) for rule in tr.RULES], ids)
    with pytest.raises(tr.AlreadyJudged, match="judged already"):
        await tr.register(engine)


class _Market:
    async def aclose(self) -> None:
        pass


@pytest.mark.usefixtures("database_url")
def test_the_command_registers_before_it_reads_and_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from halal_trader.cli import cli
    from halal_trader.data.alpaca_market import AlpacaMarketData
    from halal_trader.db.repos.quant_trials import QuantTrialRepoImpl

    registered: list[int] = []

    async def run(engine: AsyncEngine, market: Any) -> tuple[dq.Report, list[tr.Split]]:
        registered.append(await QuantTrialRepoImpl(engine).count_trials(name_prefix=tr.TRIAL))
        return dq.Report(tr.START, tr.END), [tr.split(rule, []) for rule in tr.RULES]

    monkeypatch.setattr(AlpacaMarketData, "from_settings", lambda settings: _Market())
    monkeypatch.setattr(tr, "run", run)

    first = CliRunner().invoke(cli, ["research", "replay-trips"])
    again = CliRunner().invoke(cli, ["research", "replay-trips"])

    assert first.exit_code == 0, first.output
    assert registered == [3]
    assert "market_trend: no pass" in first.output
    assert again.exit_code == 1
    assert "judged already" in again.output
