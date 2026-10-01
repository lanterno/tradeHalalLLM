"""The test harness stays off live state: databases, the real .env, the network.

tests/conftest.py runs DROP DATABASE on the server TEST_PG_* points at, which
by default is the live bot's Postgres. These pin the allowlist that keeps it
off ``halal_trader`` itself.
"""

from __future__ import annotations

import pytest

from tests.conftest import _assert_disposable_db


@pytest.mark.parametrize(
    "name",
    [
        "halal_trader_test",
        "halal_trader_test_gw0",
        "halal_trader_test_gw12",
        "other_project_test",
        "halal_trader_alembic_0123456789",
        "halal_trader_init_abcdef0123",
    ],
)
def test_disposable_names_are_allowed(name: str) -> None:
    _assert_disposable_db(name)


@pytest.mark.parametrize(
    "name",
    [
        "halal_trader",  # the live database
        "postgres",
        "template1",
        "halal_trader_testing",
        "halal_trader_test; DROP DATABASE halal_trader",
        "halal_trader_alembic_live",
        "HALAL_TRADER_TEST",
    ],
)
def test_live_and_unknown_names_are_refused(name: str) -> None:
    with pytest.raises(RuntimeError, match="refusing"):
        _assert_disposable_db(name)


def test_outbound_network_is_blocked() -> None:
    """A test reaching a real API is either flaky or trading for real."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="network connection"):
            sock.connect(("203.0.113.1", 443))  # TEST-NET-3, never routable
    finally:
        sock.close()


def test_settings_do_not_read_the_operators_env_file() -> None:
    """The real .env holds live keys; the suite must run on code defaults."""
    from halal_trader import config

    assert config._BASE_CONFIG["env_file"] != ".env"
