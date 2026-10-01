"""Guard: the live stock process never loads the Binance client.

Crypto trading was abandoned (2026-10-01) and its code deleted. These tests
pin the result transitively, not lexically: in a FRESH interpreter, importing
the stock bot's composition root, the dashboard app or the CLI must not pull
in any ``binance`` module (python-binance) or anything under the old
``halal_trader.crypto`` package. A shared helper that quietly re-imports the
exchange client fails here, before it reaches the running bot.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_FORBIDDEN_PREFIXES = ("binance", "halal_trader.crypto")


@pytest.mark.parametrize(
    "module",
    [
        "halal_trader.trading.scheduler",
        "halal_trader.web.app",
        "halal_trader.cli",
    ],
)
def test_importing_live_entrypoint_loads_no_crypto_module(module: str) -> None:
    code = (
        f"import {module}, sys; "
        f"prefixes = {_FORBIDDEN_PREFIXES!r}; "
        "leaked = sorted(m for m in sys.modules "
        "if any(m == p or m.startswith(p + '.') for p in prefixes)); "
        "assert not leaked, leaked; print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"{module} imported a crypto module: {result.stderr}"
    assert result.stdout.strip() == "clean"
