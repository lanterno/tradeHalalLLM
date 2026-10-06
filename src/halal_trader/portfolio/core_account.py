"""The core portfolio's account name in the ledger: one per Alpaca environment.

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

CORE_PAPER = "core"
CORE_LIVE = "core-live"
CORE_ACCOUNTS = (CORE_PAPER, CORE_LIVE)


def core_account(paper: bool) -> str:
    """The ledger name of the core's account in the configured environment."""
    return CORE_PAPER if paper else CORE_LIVE
