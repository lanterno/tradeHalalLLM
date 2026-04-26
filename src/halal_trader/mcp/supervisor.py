"""Alpaca MCP subprocess supervisor.

The Alpaca MCP client runs as a stdio child process; if it crashes
mid-cycle the bot loses broker access until the next restart. The
supervisor wraps any object exposing ``connect`` / ``disconnect`` and
re-runs the connect cycle on failure with exponential backoff. It also
provides a ``call_with_recovery`` helper that retries a single broker
call once after a reconnect — common case is "MCP died between cycles
and the first call surfaces the error."

Concrete MCP details (where the subprocess lives, how its stdio works)
live in ``mcp.client``. This module is generic so the same pattern
applies to any subprocess-backed client we add later.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T")


class SubprocessClient(Protocol):
    """Anything the supervisor can babysit."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...


@dataclass
class SupervisorPolicy:
    """Reconnect cadence + total budget."""

    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    factor: float = 2.0
    max_consecutive_failures: int = 10  # bail out & alert after N flaps


class MCPSupervisor:
    """Owns a single ``SubprocessClient`` and re-establishes it on failure.

    Usage::

        sup = MCPSupervisor(client, alerts=alerts)
        await sup.start()
        result = await sup.call_with_recovery(client.get_account_info)

    The supervisor is *not* a full process manager — it doesn't fork
    the binary itself. The client owns subprocess lifecycle; the
    supervisor coordinates the connect/disconnect/retry pattern around
    it.
    """

    def __init__(
        self,
        client: SubprocessClient,
        *,
        policy: SupervisorPolicy | None = None,
        alerts: Any = None,
    ) -> None:
        self._client = client
        self._policy = policy or SupervisorPolicy()
        self._alerts = alerts
        self._consecutive_failures = 0
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def start(self) -> None:
        """Connect once on startup; raise if first connect fails."""
        await self._client.connect()
        self._connected = True
        self._consecutive_failures = 0

    async def reconnect(self) -> bool:
        """Disconnect + reconnect with backoff. Returns True on success."""
        backoff = self._policy.initial_backoff_seconds
        attempt = 0
        try:
            await self._client.disconnect()
        except Exception as e:
            logger.debug("disconnect failed during reconnect: %s", e)
        self._connected = False

        while True:
            attempt += 1
            try:
                await self._client.connect()
                self._connected = True
                self._consecutive_failures = 0
                logger.info("MCP supervisor: reconnected after %d attempt(s)", attempt)
                return True
            except Exception as e:
                self._consecutive_failures += 1
                logger.warning(
                    "MCP supervisor reconnect attempt %d failed: %s — sleeping %.1fs",
                    attempt,
                    e,
                    backoff,
                )
                if self._consecutive_failures >= self._policy.max_consecutive_failures:
                    if self._alerts is not None:
                        try:
                            await self._alerts.notify(
                                "mcp.crash",
                                f"MCP supervisor: {self._consecutive_failures} consecutive "
                                f"reconnect failures — broker offline.",
                            )
                        except Exception as alert_err:
                            logger.debug("alert notify failed: %s", alert_err)
                    return False
                await asyncio.sleep(backoff)
                backoff = min(backoff * self._policy.factor, self._policy.max_backoff_seconds)

    async def call_with_recovery(
        self,
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        """Invoke ``fn``; on failure, reconnect once and retry once."""
        try:
            result = await fn()
            self._consecutive_failures = 0
            return result
        except Exception as e:
            logger.warning("MCP call failed (%s) — reconnecting and retrying once", e)
            ok = await self.reconnect()
            if not ok:
                raise
            return await fn()

    async def stop(self) -> None:
        try:
            await self._client.disconnect()
        finally:
            self._connected = False
