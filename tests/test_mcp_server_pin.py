"""The broker subprocess is launched at an exact, known version -- never resolved fresh."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from halal_trader.config import AlpacaSettings, Settings
from halal_trader.mcp.client import server_parameters

NOT_BAKED = Path("/nonexistent/alpaca-mcp-server")


def _settings(**alpaca: object) -> Settings:
    return Settings(alpaca=AlpacaSettings(api_key="k", secret_key="s", **alpaca))


def test_uvx_launches_the_pinned_release() -> None:
    params = server_parameters(_settings(), baked_server=NOT_BAKED)

    assert params.command == "uvx"
    assert params.args == ["alpaca-mcp-server@2.3.2"]


def test_pin_follows_the_setting() -> None:
    params = server_parameters(_settings(mcp_server_version="9.9.9"), baked_server=NOT_BAKED)

    assert params.args == ["alpaca-mcp-server@9.9.9"]


def test_paper_flag_and_keys_reach_the_subprocess() -> None:
    env = server_parameters(_settings(paper_trade=True), baked_server=NOT_BAKED).env or {}

    assert env["ALPACA_PAPER_TRADE"] == "True"
    assert env["ALPACA_API_KEY"] == "k"
    assert env["ALPACA_SECRET_KEY"] == "s"


@pytest.mark.parametrize("bad", ["", "latest", "2.1", ">=2.1.1", "2.1.1; rm -rf /"])
def test_only_an_exact_version_is_accepted(bad: str) -> None:
    with pytest.raises(ValidationError):
        AlpacaSettings(mcp_server_version=bad)


def test_the_images_build_time_install_is_used_when_present(tmp_path: Path) -> None:
    baked = tmp_path / "alpaca-mcp-server"
    baked.touch()

    params = server_parameters(_settings(), baked_server=baked)

    assert (params.command, params.args) == (str(baked), [])


def test_an_empty_command_copied_from_env_example_does_not_bypass_the_install(
    tmp_path: Path,
) -> None:
    baked = tmp_path / "alpaca-mcp-server"
    baked.touch()

    params = server_parameters(_settings(mcp_server_command=""), baked_server=baked)

    assert params.command == str(baked)


def test_the_image_closure_and_the_host_pin_name_the_same_release() -> None:
    """infra/alpaca-mcp-server.txt (what the image runs) and the
    mcp_server_version default (what a host `uvx` runs) must not drift."""
    closure = Path(__file__).resolve().parents[1] / "infra" / "alpaca-mcp-server.txt"
    pins = dict(
        line.split("==", 1)
        for line in closure.read_text().splitlines()
        if line and not line.startswith("#") and "==" in line
    )
    assert pins["alpaca-mcp-server"] == AlpacaSettings.model_fields["mcp_server_version"].default
