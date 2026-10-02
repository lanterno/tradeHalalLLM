"""Phase 0 of the event roadmap: the reactor records, bounds age, observes without trading."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.events.store import EventRecorder
from halal_trader.sentiment.stocks_events import (
    AlpacaNewsSource,
    FallbackNewsSource,
    GPTHeadlineClassifier,
    HeadlineClassification,
    StockNewsEventReactor,
)


class _Classifier:
    scorer_id = "fake:v1"

    def __init__(self, score: float = 0.9, scored: bool = True) -> None:
        self.score, self.scored = score, scored
        self.calls: list[tuple[str, datetime | None]] = []

    async def classify(self, *, symbol, headline, summary="", published_at=None):
        self.calls.append((symbol, published_at))
        return HeadlineClassification(self.score, "earnings", "why", scored=self.scored)


class _Source:
    name = "alpaca"

    def __init__(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        self.items = items
        self.asked: list[list[str]] = []

    async def fetch(self, symbols):
        self.asked.append(symbols)
        return self.items

    async def aclose(self):
        pass


def _item(n: int, *, age_min: float = 1.0) -> dict[str, Any]:
    return {
        "id": f"alpaca:{n}",
        "headline": f"headline {n}",
        "url": f"https://x/{n}",
        "source": "benzinga",
        "feed": "alpaca",
        "published_at": datetime.now(UTC) - timedelta(minutes=age_min),
    }


def _reactor(source, classifier, engine=None, observe=()):
    async def _observe():
        return list(observe)

    return StockNewsEventReactor(
        api_key="",
        symbols=["AAPL"],
        classifier=classifier,
        source=source,
        recorder=EventRecorder(engine) if engine is not None else None,
        observe_provider=_observe if observe else None,
        score_threshold=0.85,
    )


async def test_observe_only_symbols_are_scored_and_recorded_but_never_dispatched(
    engine: AsyncEngine,
) -> None:
    source = _Source([("AAPL", _item(1)), ("NFLX", _item(2))])
    classifier = _Classifier(0.95)
    r = _reactor(source, classifier, engine, observe=("NFLX", "AAPL"))

    events = await r._scan_all_symbols()

    assert [e.symbol for e in events] == ["AAPL"]
    assert sorted(source.asked[0]) == ["AAPL", "NFLX"]  # the trading list is not observed twice
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT e.symbol, e.source, e.source_id, e.payload->>'observe_only' AS obs, "
                    "s.scorer, s.score FROM events e JOIN event_scores s ON s.event_id = e.id "
                    "ORDER BY e.symbol"
                )
            )
        ).all()
    assert [(r.symbol, r.source, r.source_id, r.obs, r.scorer, r.score) for r in rows] == [
        ("AAPL", "alpaca", "alpaca:1", "false", "fake:v1", 0.95),
        ("NFLX", "alpaca", "alpaca:2", "true", "fake:v1", 0.95),
    ]


async def test_a_stale_headline_is_recorded_unscored_and_never_classified(
    engine: AsyncEngine,
) -> None:
    classifier = _Classifier(0.99)
    r = _reactor(_Source([("AAPL", _item(1, age_min=45))]), classifier, engine)

    assert await r._scan_all_symbols() == []
    assert classifier.calls == []
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM events"))).scalar() == 1
        assert (await conn.execute(text("SELECT count(*) FROM event_scores"))).scalar() == 0


async def test_an_unscored_result_is_not_recorded_as_a_score(engine: AsyncEngine) -> None:
    r = _reactor(_Source([("AAPL", _item(1))]), _Classifier(0.0, scored=False), engine)
    await r._scan_all_symbols()
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT count(*) FROM events"))).scalar() == 1
        assert (await conn.execute(text("SELECT count(*) FROM event_scores"))).scalar() == 0


async def test_the_classifier_is_told_when_the_headline_was_published() -> None:
    classifier = _Classifier(0.5)
    item = _item(1, age_min=3)
    await _reactor(_Source([("AAPL", item)]), classifier)._scan_all_symbols()
    assert classifier.calls == [("AAPL", item["published_at"])]


async def test_the_gpt_prompt_carries_the_headlines_age_and_the_scorer_names_its_prompt() -> None:
    llm = AsyncMock()
    llm.generate_json = AsyncMock(return_value={"score": 0.5})
    llm.model = "z-ai/glm-5.2"
    c = GPTHeadlineClassifier(llm)
    await c.classify(
        symbol="AAPL", headline="h", published_at=datetime.now(UTC) - timedelta(minutes=12)
    )
    prompt = llm.generate_json.await_args.args[0]
    assert "(12 minutes ago)" in prompt
    assert c.scorer_id.startswith("llm:z-ai/glm-5.2:")


async def test_a_cap_short_circuit_is_not_a_score() -> None:
    c = GPTHeadlineClassifier(AsyncMock(), daily_classify_cap=1)
    c._daily_count, c._daily_reset_date = 1, datetime.now(UTC).strftime("%Y-%m-%d")
    assert (await c.classify(symbol="A", headline="h")).scored is False


async def test_alpaca_articles_map_to_each_watched_symbol_they_name() -> None:
    from halal_trader.data.alpaca_market import NewsArticle

    market = AsyncMock()
    when = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)
    market.news = AsyncMock(
        return_value=[NewsArticle(7, "Beat", "", "", "benzinga", ("AAPL", "MSFT", "XYZ"), when)]
    )
    items = await AlpacaNewsSource(market).fetch(["AAPL", "MSFT"])
    assert [(s, i["id"], i["published_at"], i["url"]) for s, i in items] == [
        ("AAPL", "alpaca:7", when, "alpaca:7"),
        ("MSFT", "alpaca:7", when, "alpaca:7"),
    ]


async def test_fallback_serves_only_the_trading_symbols_when_the_primary_fails() -> None:
    primary = AsyncMock()
    primary.name = "alpaca"
    primary.fetch = AsyncMock(side_effect=RuntimeError("down"))
    fallback = AsyncMock()
    fallback.fetch = AsyncMock(return_value=[])
    await FallbackNewsSource(primary, fallback, ["aapl"]).fetch(["AAPL", "NFLX", "META"])
    fallback.fetch.assert_awaited_once_with(["AAPL"])


@pytest.mark.parametrize("first_wins", [True])
async def test_a_second_sighting_keeps_the_first_seen_at(engine: AsyncEngine, first_wins) -> None:
    from halal_trader.events.store import EventRecord

    rec = EventRecorder(engine)
    t0 = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)
    base = dict(source="alpaca", source_id="1", kind="news", symbol="AAPL", published_at=t0)
    assert await rec.record([EventRecord(**base, seen_at=t0)]) == 1
    assert await rec.record([EventRecord(**base, seen_at=t0 + timedelta(hours=1))]) == 0
    async with engine.connect() as conn:
        seen = (await conn.execute(text("SELECT seen_at FROM events"))).scalar()
    assert seen == t0


async def test_observe_only_scoring_stops_at_its_own_daily_allowance() -> None:
    items = [("NFLX", _item(n)) for n in range(3)] + [("AAPL", _item(9))]
    classifier = _Classifier(0.5)
    r = _reactor(_Source(items), classifier, observe=("NFLX",))
    r._observe_cap = 2
    await r._scan_all_symbols()
    assert [s for s, _ in classifier.calls] == ["NFLX", "NFLX", "AAPL"]
