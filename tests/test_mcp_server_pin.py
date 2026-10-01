"""The broker subprocess is launched at an exact, configured version."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from halal_trader.config import AlpacaSettings, Settings
from halal_trader.mcp.client import server_parameters


def _settings(**alpaca: object) -> Settings:
    return Settings(alpaca=AlpacaSettings(api_key="k", secret_key="s", **alpaca))


def test_uvx_launches_the_pinned_release() -> None:
    params = server_parameters(_settings())

    assert params.command == "uvx"
    assert params.args == ["alpaca-mcp-server@2.3.2"]


def test_pin_follows_the_setting() -> None:
    params = server_parameters(_settings(mcp_server_version="9.9.9"))

    assert params.args == ["alpaca-mcp-server@9.9.9"]


def test_paper_flag_and_keys_reach_the_subprocess() -> None:
    env = server_parameters(_settings(paper_trade=True)).env or {}

    assert env["ALPACA_PAPER_TRADE"] == "True"
    assert env["ALPACA_API_KEY"] == "k"
    assert env["ALPACA_SECRET_KEY"] == "s"


@pytest.mark.parametrize("bad", ["", "latest", "2.1", ">=2.1.1", "2.1.1; rm -rf /"])
def test_only_an_exact_version_is_accepted(bad: str) -> None:
    with pytest.raises(ValidationError):
        AlpacaSettings(mcp_server_version=bad)
