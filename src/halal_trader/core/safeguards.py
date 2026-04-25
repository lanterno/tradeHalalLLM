"""Live-mode safeguards — refuse to start without a daily confirmation token.

Today the system flips to real money with one env var. That asymmetry is
the highest-impact "real money" mistake to guard against. This module
adds a friction layer:

1. ``check_live_mode_token(settings)`` — called at scheduler startup. If
   the bot is configured for live trading (testnet/paper flag off) the
   ``LIVE_MODE_CONFIRMATION`` env var must match
   ``"I-UNDERSTAND-REAL-MONEY-<today>"`` for today's date in UTC,
   otherwise startup raises ``LiveModeError`` with the exact token to
   set.
2. ``LiveModeChecker.assert_safe(...)`` — called from the first cycle of
   each market in live mode. Checks (a) account balance ≤
   ``max_account_balance_usd``, (b) max single-order notional ≤
   ``max_single_order_usd``, (c) ``daily_loss_limit`` ≤
   ``live_mode_max_daily_loss_pct`` (a hard floor that cannot be
   loosened by config in live mode). Failure trips the kill-switch and
   sends a Telegram alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from halal_trader.config import Settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.notifications.telegram import AlertSink

logger = logging.getLogger(__name__)


class LiveModeError(RuntimeError):
    """Raised when live-mode safeguards refuse the configuration."""


def expected_token(today: datetime | None = None) -> str:
    """The exact LIVE_MODE_CONFIRMATION value the bot demands today (UTC)."""
    today = today or datetime.now(UTC)
    return f"I-UNDERSTAND-REAL-MONEY-{today.strftime('%Y-%m-%d')}"


def is_live_mode(settings: Settings, *, market: str) -> bool:
    """Return True if the configured market is operating against real money."""
    if market == "crypto":
        return not settings.binance_testnet
    if market == "stocks":
        return not settings.alpaca_paper_trade
    raise ValueError(f"unknown market: {market}")


def check_live_mode_token(settings: Settings, *, market: str, now: datetime | None = None) -> None:
    """Raise ``LiveModeError`` unless the operator confirmed live mode today."""
    if not is_live_mode(settings, market=market):
        return
    expected = expected_token(now)
    actual = settings.live_mode_confirmation.strip()
    if actual == expected:
        return

    raise LiveModeError(
        f"Refusing to start the {market} bot in LIVE mode without today's "
        f"confirmation token.\n\n"
        f"Set LIVE_MODE_CONFIRMATION={expected!s} (matches today's UTC date)\n"
        f"or set the testnet/paper flag back to true.\n\n"
        f"Got: {actual!r}"
    )


@dataclass
class LiveModeChecker:
    """Cycle-time assertions that hold for the lifetime of a live-mode bot."""

    settings: Settings
    market: str

    def __post_init__(self) -> None:
        self._tripped = False

    @property
    def active(self) -> bool:
        return is_live_mode(self.settings, market=self.market)

    @property
    def tripped(self) -> bool:
        return self._tripped

    def _effective_loss_limit(self) -> float:
        if self.market == "crypto":
            return self.settings.crypto_daily_loss_limit
        return self.settings.daily_loss_limit

    def _effective_max_position_notional(self, account_balance: float) -> float:
        if self.market == "crypto":
            pct = self.settings.crypto_max_position_pct
        else:
            pct = self.settings.max_position_pct
        return max(account_balance * pct, 0.0)

    async def assert_safe(
        self,
        *,
        account_balance: float,
        engine: "AsyncEngine | None" = None,
        alerts: "AlertSink | None" = None,
    ) -> bool:
        """Run live-mode invariants. Returns ``True`` when safe.

        On failure: trips the kill-switch (best-effort, requires
        ``engine``), fires a Telegram alert (best-effort), and returns
        ``False``. The caller should refuse to trade.
        """
        if not self.active or self._tripped:
            return not self._tripped

        violations: list[str] = []

        if account_balance > self.settings.max_account_balance_usd:
            violations.append(
                f"Account balance ${account_balance:,.2f} exceeds "
                f"max_account_balance_usd ${self.settings.max_account_balance_usd:,.2f}."
            )

        single_order_notional = self._effective_max_position_notional(account_balance)
        if single_order_notional > self.settings.max_single_order_usd:
            violations.append(
                f"Implied single-order notional ${single_order_notional:,.2f} "
                f"(={self._max_position_pct():.0%} of balance) exceeds "
                f"max_single_order_usd ${self.settings.max_single_order_usd:,.2f}."
            )

        loss_limit = self._effective_loss_limit()
        if loss_limit > self.settings.live_mode_max_daily_loss_pct:
            violations.append(
                f"daily_loss_limit {loss_limit:.2%} exceeds the live-mode floor "
                f"{self.settings.live_mode_max_daily_loss_pct:.2%}."
            )

        if not violations:
            return True

        self._tripped = True
        details = "\n".join(f"• {v}" for v in violations)
        logger.error(
            "Live-mode safeguard violations on %s — engaging kill-switch:\n%s",
            self.market,
            details,
            extra={"event": "safeguards.violation", "market": self.market},
        )

        if engine is not None:
            try:
                from halal_trader.core import halt as halt_module

                await halt_module.set_halt(
                    engine,
                    reason=f"live-mode safeguard ({self.market})",
                    set_by="LiveModeChecker",
                )
            except Exception as e:
                logger.error("Failed to engage kill-switch from safeguard: %s", e)

        if alerts is not None:
            await alerts.notify(
                "safeguards.violation",
                f"Live-mode safeguards on {self.market} tripped:\n{details}",
            )

        return False

    def _max_position_pct(self) -> float:
        if self.market == "crypto":
            return self.settings.crypto_max_position_pct
        return self.settings.max_position_pct
