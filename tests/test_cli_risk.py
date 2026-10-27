"""`halal-trader risk status` and `risk set`."""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from halal_trader.cli import cli
from halal_trader.core import risk_meter


@pytest.fixture
def announced(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    sent: list[Any] = []

    async def record(settings: Any, status: Any) -> bool:
        sent.append(status)
        return False

    monkeypatch.setattr(risk_meter, "announce", record)
    return sent


@pytest.mark.usefixtures("database_url")
def test_status_shows_the_level_and_every_levels_limits() -> None:
    result = CliRunner().invoke(cli, ["risk", "status"])

    assert result.exit_code == 0, result.output
    assert "Risk meter at level 3 (Standard)" in result.output
    for name in ("Cautious", "Careful", "Standard", "Bold", "Max"):
        assert name in result.output


@pytest.mark.usefixtures("database_url")
def test_set_stores_the_level_and_announces_it(announced) -> None:
    result = CliRunner().invoke(cli, ["risk", "set", "3", "--reason", "hold for the review"])

    assert result.exit_code == 0, result.output
    assert "hold for the review" in result.output
    assert "wasn't announced" in result.output
    assert [s.level for s in announced] == [3]


@pytest.mark.usefixtures("database_url")
def test_set_refuses_a_locked_level_with_the_reason(announced) -> None:
    result = CliRunner().invoke(cli, ["risk", "set", "5", "--reason", "go"])

    assert result.exit_code == 1
    assert "unlock test" in result.output
    assert announced == []


def test_set_needs_a_reason() -> None:
    result = CliRunner().invoke(cli, ["risk", "set", "3"])

    assert result.exit_code == 2
    assert "--reason" in result.output
