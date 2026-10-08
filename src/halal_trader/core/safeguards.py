"""Live-mode safeguards — refuse to start without a daily confirmation token.

Today the system flips to real money with one env var. That asymmetry is
the highest-impact "real money" mistake to guard against. This module
adds a friction layer:

1. ``check_live_mode_token(settings)`` — called at scheduler startup. If
   the bot is configured for live trading (paper flag off) the
   ``LIVE_MODE_CONFIRMATION`` env var must match
   ``"I-UNDERSTAND-REAL-MONEY-<today>"`` for today's date in UTC,
   otherwise startup raises ``LiveModeError`` with the exact token to
   set.
2. ``LiveModeChecker.assert_safe(...)`` — called from every trading
   cycle in live mode. Checks (a) account balance ≤
   ``max_account_balance_usd``, (b) max single-order notional ≤
   ``max_single_order_usd``, (c) ``daily_loss_limit`` ≤
   ``live_mode_max_daily_loss_pct`` (a hard floor that cannot be
   loosened by config in live mode). Failure trips the kill-switch and
   sends a Telegram alert.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from halal_trader.config import Settings
from halal_trader.domain.models import Account

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
    if market == "stocks":
        return not settings.alpaca.paper_trade
    if market == "core":
        return not settings.core.paper
    raise ValueError(f"unknown market: {market}")


def check_live_mode_token(settings: Settings, *, market: str, now: datetime | None = None) -> None:
    """Raise ``LiveModeError`` unless the operator confirmed live mode today."""
    if not is_live_mode(settings, market=market):
        return
    expected = expected_token(now)
    actual = settings.live_mode.confirmation.strip()
    if actual == expected:
        return

    raise LiveModeError(
        f"Refusing to start the {market} bot in LIVE mode without today's "
        f"confirmation token.\n\n"
        f"Set LIVE_MODE_CONFIRMATION={expected!s} (matches today's UTC date)\n"
        f"or set the paper flag back to true.\n\n"
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
        return self.settings.stocks.daily_loss_limit

    def _effective_max_position_notional(self, account_balance: float) -> float:
        pct = self.settings.stocks.max_position_pct
        return max(account_balance * pct, 0.0)

    async def assert_safe(
        self,
        *,
        account_balance: float,
        engine: AsyncEngine | None = None,
        alerts: AlertSink | None = None,
    ) -> bool:
        """Run live-mode invariants. Returns ``True`` when safe.

        On failure: trips the kill-switch (best-effort, requires
        ``engine``), fires a Telegram alert (best-effort), and returns
        ``False``. The caller should refuse to trade.
        """
        if not self.active or self._tripped:
            return not self._tripped

        violations: list[str] = []

        if account_balance > self.settings.live_mode.max_account_balance_usd:
            violations.append(
                f"Account balance ${account_balance:,.2f} exceeds "
                f"max_account_balance_usd ${self.settings.live_mode.max_account_balance_usd:,.2f}."
            )

        single_order_notional = self._effective_max_position_notional(account_balance)
        if single_order_notional > self.settings.live_mode.max_single_order_usd:
            violations.append(
                f"Implied single-order notional ${single_order_notional:,.2f} "
                f"(={self._max_position_pct():.0%} of balance) exceeds "
                f"max_single_order_usd ${self.settings.live_mode.max_single_order_usd:,.2f}."
            )

        loss_limit = self._effective_loss_limit()
        if loss_limit > self.settings.live_mode.max_daily_loss_pct:
            violations.append(
                f"daily_loss_limit {loss_limit:.2%} exceeds the live-mode floor "
                f"{self.settings.live_mode.max_daily_loss_pct:.2%}."
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
                logger.error("Failed to engage kill-switch from safeguard: %r", e)

        if alerts is not None:
            await alerts.notify(
                "safeguards.violation",
                f"Live-mode safeguards on {self.market} tripped:\n{details}",
            )

        return False

    def _max_position_pct(self) -> float:
        return self.settings.stocks.max_position_pct


# ── The core portfolio (portfolio/core_executor.py) ──────────────────
#
# The core is a second order path on its own account. Going live is
# CORE_PAPER=false with the live account's keys, and every one of these must
# hold first, at bot start and on `halal-trader core run`:
#
# * a dated token, CORE_LIVE_CONFIRMATION = "I-UNDERSTAND-REAL-MONEY-CORE-<UTC
#   date>" (its own, so the day-trader's token never arms the core);
# * the paper rehearsal passed its gate (portfolio/readiness.py:live_gate);
# * a cash account: multiplier 1, shorting off, no debit. Any margin debit is
#   riba, so an account able to borrow is refused even if it never has;
# * keys that reach a different account from the day-trader's (checked on
#   paper as well: two strategies on one account corrupt each other's books).
#
# The ceiling on what the live core holds invested (CORE_LIVE_MAX_NOTIONAL)
# is enforced by the executor's budget.

CORE_TOKEN_PREFIX = "I-UNDERSTAND-REAL-MONEY-CORE"


def expected_core_token(today: datetime | None = None) -> str:
    """The exact CORE_LIVE_CONFIRMATION value live money needs today (UTC)."""
    today = today or datetime.now(UTC)
    return f"{CORE_TOKEN_PREFIX}-{today.strftime('%Y-%m-%d')}"


def core_token_problem(settings: Settings, *, now: datetime | None = None) -> str | None:
    """Why the live core is not confirmed today; None on paper or when it is."""
    if not is_live_mode(settings, market="core"):
        return None
    expected = expected_core_token(now)
    if settings.core.live_confirmation.strip() == expected:
        return None
    return (
        f"CORE_PAPER=false needs today's confirmation: CORE_LIVE_CONFIRMATION={expected} "
        "(or set CORE_PAPER=true)"
    )


def cash_account_problems(account: Account) -> list[str]:
    """Why ``account`` is not a cash account (empty when it is). Unknown is a no."""
    problems = []
    if account.multiplier is None:
        problems.append("the account's margin multiplier is unknown")
    elif account.multiplier > 1:
        problems.append(f"a margin account (multiplier {account.multiplier:g}): use a cash account")
    if account.shorting_enabled is not False:
        problems.append("shorting is enabled (or unknown) on the account")
    if account.cash < 0:
        problems.append(f"the account owes ${-account.cash:,.2f} (a margin debit)")
    return problems


def same_account_problem(core: Account, day_trader: Account | None) -> str | None:
    """Do the two sets of keys reach one account? Unknown ids are a refusal."""
    if day_trader is None:
        return None
    if not (core.account_id or core.account_number):
        return "the core's account id is unknown: cannot tell it from the day-trader's"
    for mine, theirs in (
        (core.account_id, day_trader.account_id),
        (core.account_number, day_trader.account_number),
    ):
        if mine and theirs and mine == theirs:
            return "the core's keys reach the day-trader's account: give the core its own"
    return None


async def day_trader_account(settings: Settings) -> Account | None:
    """The day-trader's account, read with its own keys (None when it has none)."""
    from halal_trader.execution.alpaca_broker import AlpacaRestBroker

    alpaca = settings.alpaca
    if not (alpaca.api_key and alpaca.secret_key):
        return None
    broker = AlpacaRestBroker(alpaca.api_key, alpaca.secret_key, paper=alpaca.paper_trade)
    try:
        return await broker.get_account_info()
    finally:
        await broker.disconnect()


async def core_preflight(
    settings: Settings,
    *,
    engine: AsyncEngine,
    core: Account,
    day_trader: Callable[[Settings], Awaitable[Account | None]],
    today: date,
    now: datetime | None = None,
    check_token: bool = True,
) -> list[str]:
    """Every reason the core must not trade now (empty: it may).

    ``check_token``: the bot checks the dated token once, at start (a token
    can only match the day it was written); `core run` checks it every time.
    """
    problems: list[str] = []
    if settings.core.alpaca_api_key and settings.core.alpaca_api_key == settings.alpaca.api_key:
        problems.append("the core and the day-trader share API keys: give the core its own")
    else:
        try:
            other = await day_trader(settings)
        except Exception as exc:  # noqa: BLE001 -- unverifiable must not crash the run
            # Live money: unverifiable is a refusal. Paper: the
            # day-trader's keys going stale (or one transient error) must not
            # stop the core's daily run, which the readiness gate counts; the
            # identical-keys check above still holds.
            msg = f"cannot read the day-trader's account to tell it from the core's ({exc!r})"
            if is_live_mode(settings, market="core"):
                problems.append(msg[:300])
            else:
                logger.warning("%s; paper core continues", msg)
        else:
            if (problem := same_account_problem(core, other)) is not None:
                problems.append(problem)
    if not is_live_mode(settings, market="core"):
        return problems

    if check_token and (token := core_token_problem(settings, now=now)) is not None:
        problems.append(token)
    problems.extend(cash_account_problems(core))
    from halal_trader.portfolio.readiness import live_gate

    problems.extend(await live_gate(engine, today=today))
    return problems
