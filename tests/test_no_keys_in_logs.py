"""Request URLs (which can carry API keys as query params) never reach the logs."""

from __future__ import annotations

import logging


def test_halabot_logging_keeps_httpx_request_lines_out() -> None:
    from halabot.platform.observability import setup_logging

    setup_logging(logging.INFO)

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING


def test_halal_trader_logging_keeps_httpx_request_lines_out(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import halal_trader.config as config
    from halal_trader.logging import setup_logging

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    config._settings = None
    setup_logging(config.get_settings())

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
