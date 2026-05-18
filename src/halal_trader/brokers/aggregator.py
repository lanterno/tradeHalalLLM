"""Multi-broker portfolio aggregator.

Round-4 wave 1.F: pure-Python composition over the broker registry
that gives operators a unified view across N wired brokers (one
total equity, one positions list, one drawdown calc, one halal
universe view).

The aggregator is **read-only**: it doesn't place orders or modify
state. Order placement remains scoped to the per-asset broker the
strategy chose. The aggregator's job is to answer the dashboard's
"what's my total picture?" questions:

* Total equity across all wired brokers (USD-equivalent).
* Per-broker breakdown (so the operator sees attribution).
* Stocks-side positions (one list, regardless of which broker
  holds each name — useful when an operator wires both Alpaca
  paper + IBKR live for staged migration).
* Crypto-side balances aggregated by asset symbol.
* Health check: which brokers responded? which timed out?

When *any* broker fails (network blip, API down), the aggregator
returns the rest with the failing broker's contribution marked as
``available=False`` rather than hiding the failure or aborting.
The operator sees what they have AND what's broken.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from halal_trader.domain.models import Position
    from halal_trader.domain.ports import Broker, CryptoBroker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokerHealth:
    """Whether a single broker responded successfully on the last sweep."""

    name: str
    asset_class: str  # "stocks" | "crypto"
    available: bool
    error: str = ""


@dataclass(frozen=True)
class AggregatedPortfolio:
    """Unified-portfolio snapshot across N brokers."""

    total_equity_usd: float
    stocks_equity_usd: float
    crypto_equity_usd: float
    stock_positions: list["Position"]
    crypto_balances_by_asset: dict[str, float]
    per_broker_equity: dict[str, float]
    per_broker_health: list[BrokerHealth]

    @property
    def has_failures(self) -> bool:
        return any(not h.available for h in self.per_broker_health)

    @property
    def healthy_broker_names(self) -> list[str]:
        return [h.name for h in self.per_broker_health if h.available]


@dataclass
class PortfolioAggregator:
    """Read-only fan-out over a set of stock + crypto brokers.

    Operators wire multiple brokers via the registry then construct
    one of these (typically once at composition time) so the
    dashboard / risk engine / mobile summary all see the unified
    picture without each grew its own multi-broker plumbing.

    The aggregator caches nothing — every :meth:`snapshot` call
    fans out fresh queries. Callers that hit the dashboard hard
    should layer their own cache on top.
    """

    stock_brokers: dict[str, "Broker"] = field(default_factory=dict)
    crypto_brokers: dict[str, "CryptoBroker"] = field(default_factory=dict)
    timeout_seconds: float = 5.0
    """Per-broker call timeout. A failed broker counts as ``available=False``
    rather than blocking the aggregator's response. 5s is the same default
    the bot's reconcile loop uses; tune via the constructor for slower
    venues."""

    def add_stock_broker(self, name: str, broker: "Broker") -> None:
        """Wire a stock broker under ``name``. Names must be unique
        across the stocks set; re-adding the same name overrides the
        previous broker (useful for tests + restarts)."""
        self.stock_brokers[name] = broker

    def add_crypto_broker(self, name: str, broker: "CryptoBroker") -> None:
        self.crypto_brokers[name] = broker

    async def snapshot(self) -> AggregatedPortfolio:
        """Fan out to every wired broker and aggregate the result.

        Per-broker failures are isolated: each broker call runs under
        ``asyncio.wait_for(timeout_seconds)`` and exceptions are
        captured in the returned ``per_broker_health`` list. The other
        brokers' data is still returned.
        """
        stock_results = await asyncio.gather(
            *(self._fetch_stock(name, b) for name, b in self.stock_brokers.items()),
            return_exceptions=False,
        )
        crypto_results = await asyncio.gather(
            *(self._fetch_crypto(name, b) for name, b in self.crypto_brokers.items()),
            return_exceptions=False,
        )

        stocks_equity = 0.0
        stock_positions: list[Position] = []
        crypto_equity = 0.0
        crypto_balances: dict[str, float] = {}
        per_broker_equity: dict[str, float] = {}
        health: list[BrokerHealth] = []

        for name, equity, positions, err in stock_results:
            health.append(
                BrokerHealth(
                    name=name,
                    asset_class="stocks",
                    available=err is None,
                    error="" if err is None else err,
                )
            )
            if err is None:
                stocks_equity += equity
                per_broker_equity[name] = equity
                stock_positions.extend(positions)

        for name, equity, balances, err in crypto_results:
            health.append(
                BrokerHealth(
                    name=name,
                    asset_class="crypto",
                    available=err is None,
                    error="" if err is None else err,
                )
            )
            if err is None:
                crypto_equity += equity
                per_broker_equity[name] = equity
                # Sum balances across brokers — same asset on Binance
                # + Coinbase aggregates correctly.
                for asset, qty in balances.items():
                    crypto_balances[asset] = crypto_balances.get(asset, 0.0) + qty

        return AggregatedPortfolio(
            total_equity_usd=stocks_equity + crypto_equity,
            stocks_equity_usd=stocks_equity,
            crypto_equity_usd=crypto_equity,
            stock_positions=stock_positions,
            crypto_balances_by_asset=crypto_balances,
            per_broker_equity=per_broker_equity,
            per_broker_health=health,
        )

    async def _fetch_stock(
        self, name: str, broker: "Broker"
    ) -> tuple[str, float, list["Position"], str | None]:
        """Pull (equity, positions) from a single stock broker; isolate failures."""
        try:
            account_task = asyncio.wait_for(broker.get_account_info(), timeout=self.timeout_seconds)
            positions_task = asyncio.wait_for(
                broker.get_all_positions(), timeout=self.timeout_seconds
            )
            account, positions = await asyncio.gather(account_task, positions_task)
            equity = account.effective_equity or account.equity or account.portfolio_value or 0.0
            return name, float(equity), list(positions), None
        except Exception as exc:  # noqa: BLE001
            logger.debug("stock broker %r snapshot failed: %s", name, exc)
            return name, 0.0, [], repr(exc)

    async def _fetch_crypto(
        self, name: str, broker: "CryptoBroker"
    ) -> tuple[str, float, dict[str, float], str | None]:
        """Pull (equity, balances) from a single crypto broker; isolate failures."""
        try:
            account_task = asyncio.wait_for(broker.get_account(), timeout=self.timeout_seconds)
            balances_task = asyncio.wait_for(broker.get_balances(), timeout=self.timeout_seconds)
            account, balances = await asyncio.gather(account_task, balances_task)
            balances_by_asset: dict[str, float] = {}
            for b in balances:
                if (b.free + b.locked) > 0:
                    balances_by_asset[b.asset] = b.free + b.locked
            equity = float(account.total_balance_usdt or 0.0)
            return name, equity, balances_by_asset, None
        except Exception as exc:  # noqa: BLE001
            logger.debug("crypto broker %r snapshot failed: %s", name, exc)
            return name, 0.0, {}, repr(exc)
