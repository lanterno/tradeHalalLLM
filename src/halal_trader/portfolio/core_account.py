"""The broker accounts, and their names in the ledger.

The day-trader's account is ``paper`` (its original ledger name, kept so
its history stays where it is), whatever its environment.

The core portfolio's account name: one per Alpaca environment.

Every table that records the core (``broker_activities``, ``broker_equity``,
``account_snapshots``, ``purification_accruals``, ``zakat_assessments``,
``core_runs``, ``core_orders``) keys it by an account name. Paper and live
are different Alpaca accounts with different money, so they get different
names and their histories never mix: a month of paper rebalances must not
count as the live account's monthly run, and paper dividends are not money
to purify.

* ``core``: the paper account, as the core has always been recorded, so the
  existing paper history stays where it is;
* ``core-live``: the live account, once ``CORE_PAPER=false``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DAY_TRADER = "paper"
CORE_PAPER = "core"
CORE_LIVE = "core-live"
CORE_ACCOUNTS = (CORE_PAPER, CORE_LIVE)
BROKER_ACCOUNTS = (DAY_TRADER, *CORE_ACCOUNTS)


def core_account(paper: bool) -> str:
    """The ledger name of the core's account in the configured environment."""
    return CORE_PAPER if paper else CORE_LIVE


@dataclass(frozen=True, slots=True)
class BrokerAccount:
    name: str  # its ledger name
    label: str  # for people
    api_key: str
    secret_key: str
    paper: bool


def broker_accounts(settings: Any) -> list[BrokerAccount]:
    """The configured accounts: the day-trader's, then the core's once its keys are set."""
    alpaca, core = settings.alpaca, settings.core
    out = [
        BrokerAccount(
            DAY_TRADER, "day-trader", alpaca.api_key, alpaca.secret_key, alpaca.paper_trade
        )
    ]
    if core.enabled:
        out.append(
            BrokerAccount(
                core_account(core.paper),
                "core",
                core.alpaca_api_key,
                core.alpaca_secret_key,
                core.paper,
            )
        )
    return out
