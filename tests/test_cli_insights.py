"""`halal-trader insights catalysts`: what the next cycle's prompt would show, no database."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from click.testing import CliRunner

from halal_trader.cli import cli
from halal_trader.trading.catalysts import Catalyst, StockCatalystFeed


def test_catalysts_prints_what_the_sources_return(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[list[str]] = []

    async def fetch_all(self: StockCatalystFeed, symbols: Sequence[str]) -> list[Catalyst]:
        asked.append(list(symbols))
        return [
            Catalyst(
                symbol="AAPL",
                kind="news",
                title="Apple beats",
                timestamp=datetime.now(UTC),
                source="test",
            )
        ]

    monkeypatch.setattr(StockCatalystFeed, "fetch_all", fetch_all)

    result = CliRunner().invoke(cli, ["insights", "catalysts", "aapl"])
    assert result.exit_code == 0, result.output
    assert asked == [["AAPL"]]
    assert "Catalysts for AAPL" in result.output
    assert "Apple beats" in result.output


def test_catalysts_says_when_there_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fetch_all(self: StockCatalystFeed, symbols: Sequence[str]) -> list[Catalyst]:
        return []

    monkeypatch.setattr(StockCatalystFeed, "fetch_all", fetch_all)

    result = CliRunner().invoke(cli, ["insights", "catalysts", "MSFT"])
    assert result.exit_code == 0, result.output
    assert "no catalysts in window" in result.output
