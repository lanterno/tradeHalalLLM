"""One parser for Alpaca's snapshot, whatever shape the server sends."""

from __future__ import annotations

from halal_trader.core.num import to_float
from halal_trader.trading.bars import Snapshot, parse_snapshot

CAMEL = {
    "latestTrade": {"p": 101.0},
    "latestQuote": {"bp": 100.9, "ap": 101.1},
    "minuteBar": {"c": 100.8},
    "dailyBar": {"o": 99.0, "c": 100.5, "v": 1_000_000},
    "prevDailyBar": {"c": 98.0},
}
SNAKE = {
    "latest_trade": {"price": 101.0},
    "latest_quote": {"bid_price": 100.9, "ask_price": 101.1},
    "minute_bar": {"close": 100.8},
    "daily_bar": {"open": 99.0, "close": 100.5, "volume": 1_000_000},
    "prev_daily_bar": {"close": 98.0},
}
EXPECTED = Snapshot(101.0, 100.9, 101.1, 100.8, 99.0, 100.5, 1_000_000.0, 98.0)


def test_every_envelope_and_key_style_reads_the_same() -> None:
    for payload in (
        {"AAPL": CAMEL},
        {"AAPL": SNAKE},
        {"data": {"AAPL": CAMEL}},
        CAMEL,  # a single-symbol response, unkeyed
        {"aapl": CAMEL} | {"AAPL": CAMEL},
    ):
        assert parse_snapshot(payload, "AAPL") == EXPECTED


def test_missing_and_unparseable_fields_are_none() -> None:
    s = parse_snapshot({"AAPL": {"latestTrade": {"p": "abc"}, "dailyBar": {"c": 50}}}, "AAPL")
    assert s is not None and s.last_trade is None and s.daily_close == 50.0
    assert parse_snapshot("garbage", "AAPL") is None
    assert parse_snapshot({"AAPL": "garbage"}, "AAPL") is None


def test_the_executor_prices_by_last_trade_then_minute_then_daily_close() -> None:
    from halal_trader.trading.executor import TradeExecutor

    price = TradeExecutor._extract_price
    assert price(None, {"AAPL": CAMEL}, "AAPL") == 101.0  # type: ignore[arg-type]
    no_trade = {k: v for k, v in CAMEL.items() if k != "latestTrade"}
    assert price(None, {"AAPL": no_trade}, "AAPL") == 100.8  # type: ignore[arg-type]
    only_daily = {"dailyBar": {"c": 100.5}}
    assert price(None, {"AAPL": only_daily}, "AAPL") == 100.5  # type: ignore[arg-type]
    assert price(None, {}, "AAPL") == 0.0  # type: ignore[arg-type]


def test_to_float() -> None:
    assert to_float("1.5") == 1.5 and to_float(0) == 0.0 and to_float(2) == 2.0
    assert to_float(None) is None and to_float("") is None
    assert to_float("n/a") is None and to_float({}) is None
