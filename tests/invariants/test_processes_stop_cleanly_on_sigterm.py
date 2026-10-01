"""INVARIANT: SIGTERM stops the stock bot and the shadow cleanly.

Both run as PID 1 in docker, and PID 1 ignores signals it has no handler
for: every `docker stop` / redeploy used to end in SIGKILL, so shutdown()
never ran (monitor/reactor not cancelled, broker subprocess not closed).
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from typing import Any

from halabot.cli import _stop_on_signals
from halal_trader.trading.scheduler import TradingBot


class FakeLoop:
    """Records add_signal_handler registrations so a test can 'deliver' one."""

    def __init__(self) -> None:
        self.handlers: dict[int, tuple[Callable[..., Any], tuple[Any, ...]]] = {}

    def add_signal_handler(self, sig: int, cb: Callable[..., Any], *args: Any) -> None:
        self.handlers[sig] = (cb, args)

    def deliver(self, sig: int) -> None:
        cb, args = self.handlers[sig]
        cb(*args)


def test_sigterm_stops_the_stock_bot_run_loop() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot._running = True
    loop = FakeLoop()

    bot._install_signal_handlers(loop)  # type: ignore[arg-type]
    loop.deliver(signal.SIGTERM)

    assert bot._running is False  # run()'s loop exits; its finally calls shutdown()


def test_sigint_is_handled_the_same_way() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot._running = True
    loop = FakeLoop()

    bot._install_signal_handlers(loop)  # type: ignore[arg-type]
    loop.deliver(signal.SIGINT)

    assert bot._running is False


def test_sigterm_stops_the_shadow() -> None:
    stop = asyncio.Event()
    loop = FakeLoop()

    _stop_on_signals(loop, stop)  # type: ignore[arg-type]
    loop.deliver(signal.SIGTERM)

    assert stop.is_set()


def test_the_fleet_stops_with_an_init_and_a_grace_period() -> None:
    """compose must run an init (signal forwarding, zombie reaping) and give
    shutdown() time to finish before docker escalates to SIGKILL."""
    from pathlib import Path

    import yaml

    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "infra" / "docker-compose.yml").read_text()
    )
    for name in ("trader-stocks", "trader-shadow"):
        svc = compose["services"][name]
        assert svc.get("init") is True, f"{name}: init: true missing"
        assert svc.get("stop_grace_period"), f"{name}: stop_grace_period missing"
