"""broker_compare: tolerant on moving values, exact on positions and the clock."""

from __future__ import annotations

from typing import Any

from halal_trader.domain.models import Account, MarketClock, Position
from halal_trader.execution.broker_compare import compare


class Fake:
    def __init__(self, *, equity: float, qty: float, next_open: str, price: float) -> None:
        self.equity, self.qty, self.next_open, self.price = equity, qty, next_open, price

    async def get_account_info(self) -> Account:
        return Account(equity=self.equity, cash=1.0, buying_power=2.0, status="ACTIVE")

    async def get_clock(self) -> MarketClock:
        return MarketClock(
            is_open=False, next_open=self.next_open, next_close="2026-10-02T16:00:00-04:00"
        )

    async def get_all_positions(self) -> list[Position]:
        return [Position(symbol="MSFT", qty=self.qty)]

    async def get_stock_snapshot(self, symbols: str) -> Any:
        return {"AAPL": {"latestTrade": {"p": self.price}}}

    async def get_stock_bars(self, symbol: str, days: int = 5, timeframe: str = "1Day") -> Any:
        return {"bars": {"AAPL": [{"t": "2026-10-01T04:00:00Z", "c": self.price}]}}


def _fake(**overrides: Any) -> Any:
    base: dict[str, Any] = dict(
        equity=105_000.0, qty=24.0, next_open="2026-10-02T09:30:00-04:00", price=250.0
    )
    return Fake(**(base | overrides))


async def test_agreeing_adapters_pass_despite_format_and_small_moves() -> None:
    rest = _fake(equity=105_050.0, next_open="2026-10-02T13:30:00Z", price=250.5)
    checks = await compare(_fake(), rest, ["AAPL"])
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


async def test_a_position_difference_fails() -> None:
    checks = await compare(_fake(), _fake(qty=23.0), ["AAPL"])
    assert [c.name for c in checks if not c.ok] == ["positions"]


async def test_a_missing_price_fails() -> None:
    checks = await compare(_fake(), _fake(price=None), ["AAPL"])
    assert [c.name for c in checks if not c.ok] == ["snapshot.AAPL", "bars.AAPL"]
