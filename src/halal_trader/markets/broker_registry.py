"""Broker capability registry.

Auxiliary primitive for Wave 1.B (IBKR) / 1.C (Tradier) / 1.D
(Coinbase) / 1.E (Saxo) adapters. Each adapter wave is blocked
on its respective SDK integration; this registry is the
**pure-Python catalogue** that the broker-plugin layer consults
to know which brokers support which markets, asset classes,
order types, and the paper-vs-real boundary.

Picked a focused registry over per-adapter capability methods
because (a) operators choosing "which broker should I add next"
need a single matrix to compare, not 6 separate `BrokerAdapter.capabilities()`
methods that each return a different shape, (b) the cycle's
broker-routing layer asks "does this broker support market
orders on Saudi equities?" — a closed-set capability matrix
answers in O(1) without instantiating broker clients, (c) the
paper-vs-real boundary is the single most safety-critical attribute
(routing real money to a paper-only adapter is silent failure;
routing paper trades to a real account is a regulatory issue) —
pinning it as a registry attribute keeps the executor's gate
inspectable.

Pinned semantics:
- **Closed-set Broker enum.** Adding a broker is a code review
  change; the matrix can't drift.
- **Closed-set OrderType / AssetClass enums.** New order types
  (OCO, bracket, stop-limit, etc.) are code review changes.
- **Paper-vs-real is a per-broker attribute.** The Alpaca
  paper sandbox is distinct from the real Alpaca; Binance
  testnet from real Binance. The registry has separate entries.
- **Rate limit is requests-per-minute.** Operator-tunable per
  broker; defaults match each broker's documented limit.
- **Render output never includes API keys / tokens.** Pure
  catalogue data; structural no-secret pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from halal_trader.markets.international_registry import Exchange


class Broker(str, Enum):
    """Closed-set broker identifiers.

    Pinned string values for JSON / DB stability. Adding a broker
    is a code review change.
    """

    ALPACA_PAPER = "alpaca_paper"
    ALPACA_LIVE = "alpaca_live"
    BINANCE_TESTNET = "binance_testnet"
    BINANCE_LIVE = "binance_live"
    IBKR_PAPER = "ibkr_paper"
    IBKR_LIVE = "ibkr_live"
    TRADIER_SANDBOX = "tradier_sandbox"
    TRADIER_LIVE = "tradier_live"
    COINBASE_SANDBOX = "coinbase_sandbox"
    COINBASE_LIVE = "coinbase_live"
    SAXO_SIM = "saxo_sim"
    SAXO_LIVE = "saxo_live"


class AssetClass(str, Enum):
    """Closed-set asset classes."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTION = "option"
    ETF = "etf"
    FX = "fx"
    COMMODITY = "commodity"


class OrderType(str, Enum):
    """Closed-set order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"


@dataclass(frozen=True)
class BrokerProfile:
    """One broker adapter's capability profile.

    `is_paper` is the load-bearing safety attribute: routing real
    money to a paper-only broker is silent failure; routing paper
    trades to a real account is a regulatory issue. The executor's
    routing gate consults this attribute before any order
    submission.

    `rate_limit_per_min` is an integer best-effort cap; broker-side
    rate-limiting may be stricter (per-endpoint vs global). The
    cycle's pacing layer treats this as a soft cap.
    """

    broker: Broker
    display_name: str
    is_paper: bool
    asset_classes: frozenset[AssetClass]
    order_types: frozenset[OrderType]
    supported_exchanges: frozenset[Exchange]
    rate_limit_per_min: int

    def __post_init__(self) -> None:
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if not self.asset_classes:
            raise ValueError("asset_classes must be non-empty")
        if not self.order_types:
            raise ValueError("order_types must be non-empty")
        if OrderType.MARKET not in self.order_types:
            raise ValueError("every broker profile must support MARKET orders")
        if self.rate_limit_per_min <= 0:
            raise ValueError("rate_limit_per_min must be positive")


# Canonical broker capability registry. Module-level immutable.
_BROKER_REGISTRY: dict[Broker, BrokerProfile] = {
    Broker.ALPACA_PAPER: BrokerProfile(
        broker=Broker.ALPACA_PAPER,
        display_name="Alpaca Paper",
        is_paper=True,
        asset_classes=frozenset({AssetClass.EQUITY, AssetClass.ETF}),
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.TRAILING_STOP,
                OrderType.BRACKET,
            }
        ),
        supported_exchanges=frozenset({Exchange.NYSE, Exchange.NASDAQ}),
        rate_limit_per_min=200,
    ),
    Broker.ALPACA_LIVE: BrokerProfile(
        broker=Broker.ALPACA_LIVE,
        display_name="Alpaca Live",
        is_paper=False,
        asset_classes=frozenset({AssetClass.EQUITY, AssetClass.ETF}),
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.TRAILING_STOP,
                OrderType.BRACKET,
            }
        ),
        supported_exchanges=frozenset({Exchange.NYSE, Exchange.NASDAQ}),
        rate_limit_per_min=200,
    ),
    Broker.BINANCE_TESTNET: BrokerProfile(
        broker=Broker.BINANCE_TESTNET,
        display_name="Binance Testnet",
        is_paper=True,
        asset_classes=frozenset({AssetClass.CRYPTO}),
        order_types=frozenset(
            {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.OCO}
        ),
        supported_exchanges=frozenset(),  # crypto = no equity exchange
        rate_limit_per_min=1200,
    ),
    Broker.BINANCE_LIVE: BrokerProfile(
        broker=Broker.BINANCE_LIVE,
        display_name="Binance Live",
        is_paper=False,
        asset_classes=frozenset({AssetClass.CRYPTO}),
        order_types=frozenset(
            {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.OCO}
        ),
        supported_exchanges=frozenset(),
        rate_limit_per_min=1200,
    ),
    Broker.IBKR_PAPER: BrokerProfile(
        broker=Broker.IBKR_PAPER,
        display_name="Interactive Brokers Paper",
        is_paper=True,
        asset_classes=frozenset(
            {
                AssetClass.EQUITY,
                AssetClass.ETF,
                AssetClass.OPTION,
                AssetClass.FX,
                AssetClass.COMMODITY,
            }
        ),
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.TRAILING_STOP,
                OrderType.BRACKET,
                OrderType.OCO,
            }
        ),
        supported_exchanges=frozenset(
            {Exchange.NYSE, Exchange.NASDAQ, Exchange.LSE, Exchange.HKSE, Exchange.TSE}
        ),
        rate_limit_per_min=50,  # IBKR is conservative
    ),
    Broker.IBKR_LIVE: BrokerProfile(
        broker=Broker.IBKR_LIVE,
        display_name="Interactive Brokers Live",
        is_paper=False,
        asset_classes=frozenset(
            {
                AssetClass.EQUITY,
                AssetClass.ETF,
                AssetClass.OPTION,
                AssetClass.FX,
                AssetClass.COMMODITY,
            }
        ),
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.TRAILING_STOP,
                OrderType.BRACKET,
                OrderType.OCO,
            }
        ),
        supported_exchanges=frozenset(
            {Exchange.NYSE, Exchange.NASDAQ, Exchange.LSE, Exchange.HKSE, Exchange.TSE}
        ),
        rate_limit_per_min=50,
    ),
    Broker.TRADIER_SANDBOX: BrokerProfile(
        broker=Broker.TRADIER_SANDBOX,
        display_name="Tradier Sandbox",
        is_paper=True,
        asset_classes=frozenset({AssetClass.EQUITY, AssetClass.ETF, AssetClass.OPTION}),
        order_types=frozenset(
            {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
        ),
        supported_exchanges=frozenset({Exchange.NYSE, Exchange.NASDAQ}),
        rate_limit_per_min=120,
    ),
    Broker.TRADIER_LIVE: BrokerProfile(
        broker=Broker.TRADIER_LIVE,
        display_name="Tradier Live",
        is_paper=False,
        asset_classes=frozenset({AssetClass.EQUITY, AssetClass.ETF, AssetClass.OPTION}),
        order_types=frozenset(
            {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
        ),
        supported_exchanges=frozenset({Exchange.NYSE, Exchange.NASDAQ}),
        rate_limit_per_min=120,
    ),
    Broker.COINBASE_SANDBOX: BrokerProfile(
        broker=Broker.COINBASE_SANDBOX,
        display_name="Coinbase Advanced Trade Sandbox",
        is_paper=True,
        asset_classes=frozenset({AssetClass.CRYPTO}),
        order_types=frozenset({OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT}),
        supported_exchanges=frozenset(),
        rate_limit_per_min=600,
    ),
    Broker.COINBASE_LIVE: BrokerProfile(
        broker=Broker.COINBASE_LIVE,
        display_name="Coinbase Advanced Trade Live",
        is_paper=False,
        asset_classes=frozenset({AssetClass.CRYPTO}),
        order_types=frozenset({OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT}),
        supported_exchanges=frozenset(),
        rate_limit_per_min=600,
    ),
    Broker.SAXO_SIM: BrokerProfile(
        broker=Broker.SAXO_SIM,
        display_name="Saxo Bank Simulation",
        is_paper=True,
        asset_classes=frozenset(
            {
                AssetClass.EQUITY,
                AssetClass.ETF,
                AssetClass.OPTION,
                AssetClass.FX,
                AssetClass.COMMODITY,
            }
        ),
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.TRAILING_STOP,
            }
        ),
        supported_exchanges=frozenset(
            {
                Exchange.LSE,
                Exchange.HKSE,
                Exchange.TSE,
                Exchange.NSE,
                Exchange.TADAWUL,
                Exchange.DIFC,
                Exchange.KLSE,
            }
        ),
        rate_limit_per_min=120,
    ),
    Broker.SAXO_LIVE: BrokerProfile(
        broker=Broker.SAXO_LIVE,
        display_name="Saxo Bank Live",
        is_paper=False,
        asset_classes=frozenset(
            {
                AssetClass.EQUITY,
                AssetClass.ETF,
                AssetClass.OPTION,
                AssetClass.FX,
                AssetClass.COMMODITY,
            }
        ),
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP,
                OrderType.STOP_LIMIT,
                OrderType.TRAILING_STOP,
            }
        ),
        supported_exchanges=frozenset(
            {
                Exchange.LSE,
                Exchange.HKSE,
                Exchange.TSE,
                Exchange.NSE,
                Exchange.TADAWUL,
                Exchange.DIFC,
                Exchange.KLSE,
            }
        ),
        rate_limit_per_min=120,
    ),
}


def broker_profile(broker: Broker) -> BrokerProfile:
    return _BROKER_REGISTRY[broker]


def all_brokers() -> tuple[BrokerProfile, ...]:
    """Return all profiles in canonical (enum-defined) order."""

    return tuple(_BROKER_REGISTRY[b] for b in Broker)


def brokers_supporting_asset_class(
    asset_class: AssetClass,
) -> tuple[BrokerProfile, ...]:
    return tuple(p for p in all_brokers() if asset_class in p.asset_classes)


def brokers_supporting_exchange(
    exchange: Exchange,
) -> tuple[BrokerProfile, ...]:
    return tuple(p for p in all_brokers() if exchange in p.supported_exchanges)


def brokers_supporting_order_type(
    order_type: OrderType,
) -> tuple[BrokerProfile, ...]:
    return tuple(p for p in all_brokers() if order_type in p.order_types)


def paper_brokers() -> tuple[BrokerProfile, ...]:
    """Return only paper/sandbox brokers."""

    return tuple(p for p in all_brokers() if p.is_paper)


def live_brokers() -> tuple[BrokerProfile, ...]:
    """Return only live (real-money) brokers."""

    return tuple(p for p in all_brokers() if not p.is_paper)


class BrokerCannotExecuteError(Exception):
    """Raised when a broker doesn't support the requested operation."""

    def __init__(self, broker: Broker, reason: str) -> None:
        super().__init__(f"broker {broker.value!r} cannot execute: {reason}")
        self.broker = broker
        self.reason = reason


def assert_can_execute(
    *,
    broker: Broker,
    asset_class: AssetClass,
    order_type: OrderType,
    exchange: Exchange | None = None,
) -> None:
    """Validate that a broker supports an order before submission.

    Raises `BrokerCannotExecuteError` on any unsupported facet.
    The executor's pre-submit gate calls this; it never silently
    drops an order request.
    """

    profile = broker_profile(broker)
    if asset_class not in profile.asset_classes:
        raise BrokerCannotExecuteError(broker, f"asset_class {asset_class.value!r} not supported")
    if order_type not in profile.order_types:
        raise BrokerCannotExecuteError(broker, f"order_type {order_type.value!r} not supported")
    if (
        exchange is not None
        and asset_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.OPTION)
        and exchange not in profile.supported_exchanges
    ):
        raise BrokerCannotExecuteError(broker, f"exchange {exchange.value!r} not supported")


def is_paper(broker: Broker) -> bool:
    """Convenience: True if broker is a paper/sandbox endpoint.

    The executor's "is this safe for testing?" gate calls this.
    """

    return broker_profile(broker).is_paper


def render_broker(profile: BrokerProfile) -> str:
    """Format a broker profile for ops display.

    No-secret-leak: catalogue is pure metadata; structural.
    """

    paper_marker = "📝 PAPER" if profile.is_paper else "💰 LIVE"
    asset_list = ", ".join(sorted(a.value for a in profile.asset_classes))
    order_list = ", ".join(sorted(o.value for o in profile.order_types))
    exchange_count = len(profile.supported_exchanges)
    return (
        f"🏦 {profile.display_name} ({profile.broker.value}) — {paper_marker}\n"
        f"  asset classes: {asset_list}\n"
        f"  order types: {order_list}\n"
        f"  exchanges: {exchange_count}\n"
        f"  rate limit: {profile.rate_limit_per_min}/min"
    )


__all__ = [
    "AssetClass",
    "Broker",
    "BrokerCannotExecuteError",
    "BrokerProfile",
    "OrderType",
    "all_brokers",
    "assert_can_execute",
    "broker_profile",
    "brokers_supporting_asset_class",
    "brokers_supporting_exchange",
    "brokers_supporting_order_type",
    "is_paper",
    "live_brokers",
    "paper_brokers",
    "render_broker",
]
