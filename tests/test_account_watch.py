"""The account watch: a refused key alerts, a working one stays quiet."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from halal_trader.domain.models import Account
from halal_trader.trading.scheduler import TradingBot


def _bot(core_keys: bool) -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot.settings = SimpleNamespace(
        alpaca=SimpleNamespace(api_key="dk", secret_key="ds", paper_trade=True),
        core=SimpleNamespace(
            alpaca_api_key="ck" if core_keys else "",
            alpaca_secret_key="cs" if core_keys else "",
            paper=True,
        ),
    )
    bot._alerts = SimpleNamespace(notify=AsyncMock())
    return bot


class _Broker:
    refused: set[str] = set()

    def __init__(self, key: str, secret: str, *, paper: bool) -> None:
        self.key = key

    async def get_account_info(self) -> Account:
        if self.key in self.refused:
            raise RuntimeError("401 unauthorized")
        return Account(equity=1.0, status="ACTIVE")

    async def disconnect(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _fake_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("halal_trader.execution.alpaca_broker.AlpacaRestBroker", _Broker)


async def test_a_refused_key_alerts_and_names_the_account() -> None:
    _Broker.refused = {"dk"}
    bot = _bot(core_keys=True)
    failures = await bot.account_watch()
    assert len(failures) == 1 and failures[0].startswith("day-trader")
    kind, message = bot._alerts.notify.await_args.args
    assert kind == "broker.access_failed" and "day-trader" in message


async def test_working_keys_stay_quiet() -> None:
    _Broker.refused = set()
    bot = _bot(core_keys=True)
    assert await bot.account_watch() == []
    bot._alerts.notify.assert_not_awaited()
