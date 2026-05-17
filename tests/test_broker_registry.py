"""Tests for `halal_trader.markets.broker_registry`.

Auxiliary primitive for Wave 1.B-E broker SDK adapters. Covers:
profile validation, paper-vs-real boundary, capability matrix,
assert_can_execute gate.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.markets.broker_registry import (
    AssetClass,
    Broker,
    BrokerCannotExecuteError,
    BrokerProfile,
    OrderType,
    all_brokers,
    assert_can_execute,
    broker_profile,
    brokers_supporting_asset_class,
    brokers_supporting_exchange,
    brokers_supporting_order_type,
    is_paper,
    live_brokers,
    paper_brokers,
    render_broker,
)
from halal_trader.markets.international_registry import Exchange

# --------------------------- Enum string pins --------------------------------


def test_broker_string_values_pinned() -> None:
    assert Broker.ALPACA_PAPER.value == "alpaca_paper"
    assert Broker.ALPACA_LIVE.value == "alpaca_live"
    assert Broker.BINANCE_TESTNET.value == "binance_testnet"
    assert Broker.BINANCE_LIVE.value == "binance_live"
    assert Broker.IBKR_PAPER.value == "ibkr_paper"
    assert Broker.IBKR_LIVE.value == "ibkr_live"
    assert Broker.TRADIER_SANDBOX.value == "tradier_sandbox"
    assert Broker.TRADIER_LIVE.value == "tradier_live"
    assert Broker.COINBASE_SANDBOX.value == "coinbase_sandbox"
    assert Broker.COINBASE_LIVE.value == "coinbase_live"
    assert Broker.SAXO_SIM.value == "saxo_sim"
    assert Broker.SAXO_LIVE.value == "saxo_live"


def test_asset_class_string_values_pinned() -> None:
    assert AssetClass.EQUITY.value == "equity"
    assert AssetClass.CRYPTO.value == "crypto"
    assert AssetClass.OPTION.value == "option"
    assert AssetClass.ETF.value == "etf"
    assert AssetClass.FX.value == "fx"
    assert AssetClass.COMMODITY.value == "commodity"


def test_order_type_string_values_pinned() -> None:
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.STOP.value == "stop"
    assert OrderType.STOP_LIMIT.value == "stop_limit"
    assert OrderType.TRAILING_STOP.value == "trailing_stop"
    assert OrderType.BRACKET.value == "bracket"
    assert OrderType.OCO.value == "oco"


# --------------------------- BrokerProfile validation ------------------------


def _profile(**overrides: object) -> BrokerProfile:
    base: dict[str, object] = {
        "broker": Broker.ALPACA_PAPER,
        "display_name": "Test",
        "is_paper": True,
        "asset_classes": frozenset({AssetClass.EQUITY}),
        "order_types": frozenset({OrderType.MARKET}),
        "supported_exchanges": frozenset({Exchange.NYSE}),
        "rate_limit_per_min": 100,
    }
    base.update(overrides)
    return BrokerProfile(**base)  # type: ignore[arg-type]


def test_profile_rejects_empty_display_name() -> None:
    with pytest.raises(ValueError, match="display_name"):
        _profile(display_name="")


def test_profile_rejects_empty_asset_classes() -> None:
    with pytest.raises(ValueError, match="asset_classes"):
        _profile(asset_classes=frozenset())


def test_profile_rejects_empty_order_types() -> None:
    with pytest.raises(ValueError, match="order_types"):
        _profile(order_types=frozenset())


def test_profile_rejects_missing_market_order() -> None:
    """Pin: every broker must support MARKET orders.

    A broker that can't place a market order can't execute the
    halt's emergency-exit flow; banning at construction is the
    safety guarantee.
    """

    with pytest.raises(ValueError, match="MARKET"):
        _profile(order_types=frozenset({OrderType.LIMIT}))


def test_profile_rejects_zero_rate_limit() -> None:
    with pytest.raises(ValueError, match="rate_limit"):
        _profile(rate_limit_per_min=0)


def test_profile_rejects_negative_rate_limit() -> None:
    with pytest.raises(ValueError, match="rate_limit"):
        _profile(rate_limit_per_min=-1)


def test_profile_is_frozen() -> None:
    p = broker_profile(Broker.ALPACA_PAPER)
    with pytest.raises(FrozenInstanceError):
        p.is_paper = False  # type: ignore[misc]


# --------------------------- registry coverage -------------------------------


def test_every_broker_has_profile() -> None:
    profiles = all_brokers()
    assert len(profiles) == len(Broker)
    seen = {p.broker for p in profiles}
    for b in Broker:
        assert b in seen


def test_all_brokers_canonical_order() -> None:
    profiles = all_brokers()
    spec_order = [p.broker for p in profiles]
    assert spec_order == list(Broker)


def test_every_profile_supports_market_orders() -> None:
    """Pin: structural — every profile constructor enforces MARKET."""

    for profile in all_brokers():
        assert OrderType.MARKET in profile.order_types


# --------------------------- paper-vs-real pin -------------------------------


def test_paper_brokers_returns_only_paper() -> None:
    for p in paper_brokers():
        assert p.is_paper is True


def test_live_brokers_returns_only_live() -> None:
    for p in live_brokers():
        assert p.is_paper is False


def test_paper_and_live_partition_is_complete() -> None:
    """Pin: every broker is either paper or live, no overlap."""

    paper = {p.broker for p in paper_brokers()}
    live = {p.broker for p in live_brokers()}
    assert paper.isdisjoint(live)
    assert paper | live == set(Broker)


def test_alpaca_paper_is_paper() -> None:
    assert is_paper(Broker.ALPACA_PAPER) is True


def test_alpaca_live_is_not_paper() -> None:
    """Pin: routing real money to paper-only or paper to real are
    both wrong; this gate is the load-bearing safety check."""

    assert is_paper(Broker.ALPACA_LIVE) is False


def test_binance_testnet_is_paper() -> None:
    assert is_paper(Broker.BINANCE_TESTNET) is True


def test_binance_live_is_not_paper() -> None:
    assert is_paper(Broker.BINANCE_LIVE) is False


def test_ibkr_paper_is_paper() -> None:
    assert is_paper(Broker.IBKR_PAPER) is True


# --------------------------- asset class capability --------------------------


def test_alpaca_supports_equity_and_etf() -> None:
    p = broker_profile(Broker.ALPACA_PAPER)
    assert AssetClass.EQUITY in p.asset_classes
    assert AssetClass.ETF in p.asset_classes
    assert AssetClass.CRYPTO not in p.asset_classes


def test_binance_supports_only_crypto() -> None:
    p = broker_profile(Broker.BINANCE_TESTNET)
    assert p.asset_classes == frozenset({AssetClass.CRYPTO})


def test_ibkr_supports_full_asset_set() -> None:
    p = broker_profile(Broker.IBKR_PAPER)
    assert AssetClass.EQUITY in p.asset_classes
    assert AssetClass.OPTION in p.asset_classes
    assert AssetClass.FX in p.asset_classes
    assert AssetClass.COMMODITY in p.asset_classes


def test_saxo_supports_international_exchanges() -> None:
    """Pin: Saxo is the bot's path to LSE / TSE / TADAWUL / DIFC."""

    p = broker_profile(Broker.SAXO_SIM)
    assert Exchange.LSE in p.supported_exchanges
    assert Exchange.TSE in p.supported_exchanges
    assert Exchange.TADAWUL in p.supported_exchanges
    assert Exchange.DIFC in p.supported_exchanges


def test_brokers_supporting_crypto() -> None:
    """Pin: only Binance + Coinbase support crypto."""

    crypto_brokers = brokers_supporting_asset_class(AssetClass.CRYPTO)
    crypto_set = {p.broker for p in crypto_brokers}
    assert crypto_set == {
        Broker.BINANCE_TESTNET,
        Broker.BINANCE_LIVE,
        Broker.COINBASE_SANDBOX,
        Broker.COINBASE_LIVE,
    }


def test_brokers_supporting_options() -> None:
    """Pin: IBKR / Tradier / Saxo support options; Alpaca paper doesn't."""

    option_brokers = brokers_supporting_asset_class(AssetClass.OPTION)
    option_set = {p.broker for p in option_brokers}
    assert Broker.IBKR_PAPER in option_set
    assert Broker.IBKR_LIVE in option_set
    assert Broker.TRADIER_SANDBOX in option_set
    assert Broker.TRADIER_LIVE in option_set
    assert Broker.SAXO_SIM in option_set
    assert Broker.ALPACA_PAPER not in option_set


def test_brokers_supporting_fx() -> None:
    """Pin: only IBKR + Saxo support FX in the catalogue."""

    fx_brokers = brokers_supporting_asset_class(AssetClass.FX)
    fx_set = {p.broker for p in fx_brokers}
    assert fx_set == {
        Broker.IBKR_PAPER,
        Broker.IBKR_LIVE,
        Broker.SAXO_SIM,
        Broker.SAXO_LIVE,
    }


def test_brokers_supporting_tadawul() -> None:
    """Pin: only Saxo supports TADAWUL in the catalogue (the route for
    Saudi operators trading Aramco etc)."""

    saxo_brokers = brokers_supporting_exchange(Exchange.TADAWUL)
    assert {p.broker for p in saxo_brokers} == {
        Broker.SAXO_SIM,
        Broker.SAXO_LIVE,
    }


def test_brokers_supporting_nyse() -> None:
    """NYSE brokers: Alpaca + IBKR + Tradier (US equities)."""

    nyse_brokers = brokers_supporting_exchange(Exchange.NYSE)
    nyse_set = {p.broker for p in nyse_brokers}
    assert Broker.ALPACA_PAPER in nyse_set
    assert Broker.IBKR_PAPER in nyse_set
    assert Broker.TRADIER_SANDBOX in nyse_set
    assert Broker.BINANCE_TESTNET not in nyse_set


# --------------------------- order type capability ---------------------------


def test_brokers_supporting_oco() -> None:
    """Pin: OCO orders are supported by Binance + IBKR; not Alpaca."""

    oco_brokers = brokers_supporting_order_type(OrderType.OCO)
    oco_set = {p.broker for p in oco_brokers}
    assert Broker.BINANCE_TESTNET in oco_set
    assert Broker.IBKR_PAPER in oco_set
    assert Broker.ALPACA_PAPER not in oco_set


def test_brokers_supporting_bracket() -> None:
    """Pin: BRACKET orders supported by Alpaca + IBKR."""

    bracket_brokers = brokers_supporting_order_type(OrderType.BRACKET)
    bracket_set = {p.broker for p in bracket_brokers}
    assert Broker.ALPACA_PAPER in bracket_set
    assert Broker.IBKR_PAPER in bracket_set
    assert Broker.BINANCE_TESTNET not in bracket_set


# --------------------------- assert_can_execute ------------------------------


def test_assert_can_execute_alpaca_market_equity_nyse() -> None:
    """Happy path: Alpaca + market + equity + NYSE works."""

    assert_can_execute(
        broker=Broker.ALPACA_PAPER,
        asset_class=AssetClass.EQUITY,
        order_type=OrderType.MARKET,
        exchange=Exchange.NYSE,
    )


def test_assert_can_execute_rejects_alpaca_crypto() -> None:
    """Pin: Alpaca doesn't do crypto."""

    with pytest.raises(BrokerCannotExecuteError, match="asset_class"):
        assert_can_execute(
            broker=Broker.ALPACA_PAPER,
            asset_class=AssetClass.CRYPTO,
            order_type=OrderType.MARKET,
        )


def test_assert_can_execute_rejects_alpaca_oco() -> None:
    """Pin: Alpaca doesn't support OCO orders."""

    with pytest.raises(BrokerCannotExecuteError, match="order_type"):
        assert_can_execute(
            broker=Broker.ALPACA_PAPER,
            asset_class=AssetClass.EQUITY,
            order_type=OrderType.OCO,
        )


def test_assert_can_execute_rejects_alpaca_lse() -> None:
    """Pin: Alpaca doesn't trade on LSE."""

    with pytest.raises(BrokerCannotExecuteError, match="exchange"):
        assert_can_execute(
            broker=Broker.ALPACA_PAPER,
            asset_class=AssetClass.EQUITY,
            order_type=OrderType.MARKET,
            exchange=Exchange.LSE,
        )


def test_assert_can_execute_crypto_skips_exchange_check() -> None:
    """Pin: exchange check only applies to equities-like asset classes.

    Crypto trades on the broker's matching engine, not an MIC exchange.
    """

    # Binance has empty supported_exchanges but supports crypto orders
    assert_can_execute(
        broker=Broker.BINANCE_TESTNET,
        asset_class=AssetClass.CRYPTO,
        order_type=OrderType.MARKET,
        exchange=Exchange.NYSE,  # ignored for crypto
    )


def test_assert_can_execute_no_exchange_arg() -> None:
    """Pin: exchange arg is optional; skip the check if not provided."""

    assert_can_execute(
        broker=Broker.ALPACA_PAPER,
        asset_class=AssetClass.EQUITY,
        order_type=OrderType.MARKET,
    )


def test_can_execute_error_carries_broker_and_reason() -> None:
    try:
        assert_can_execute(
            broker=Broker.ALPACA_PAPER,
            asset_class=AssetClass.CRYPTO,
            order_type=OrderType.MARKET,
        )
    except BrokerCannotExecuteError as e:
        assert e.broker is Broker.ALPACA_PAPER
        assert "asset_class" in e.reason


def test_assert_can_execute_saxo_tadawul_market() -> None:
    """Pin: Saxo can execute market orders on TADAWUL — the path for
    Saudi-jurisdiction halal traders."""

    assert_can_execute(
        broker=Broker.SAXO_SIM,
        asset_class=AssetClass.EQUITY,
        order_type=OrderType.MARKET,
        exchange=Exchange.TADAWUL,
    )


# --------------------------- render ------------------------------------------


def test_render_includes_display_name() -> None:
    out = render_broker(broker_profile(Broker.IBKR_PAPER))
    assert "Interactive Brokers" in out
    assert "ibkr_paper" in out


def test_render_paper_marker() -> None:
    out_paper = render_broker(broker_profile(Broker.ALPACA_PAPER))
    out_live = render_broker(broker_profile(Broker.ALPACA_LIVE))
    assert "📝 PAPER" in out_paper
    assert "💰 LIVE" in out_live


def test_render_includes_rate_limit() -> None:
    out = render_broker(broker_profile(Broker.IBKR_PAPER))
    assert "50/min" in out


def test_render_includes_asset_classes() -> None:
    out = render_broker(broker_profile(Broker.IBKR_PAPER))
    assert "equity" in out
    assert "fx" in out
    assert "option" in out


def test_render_no_secret_leak() -> None:
    """Pin: catalogue is pure metadata; structural no-secret."""

    for b in Broker:
        out = render_broker(broker_profile(b))
        assert "api_key" not in out.lower()
        assert "token" not in out.lower()
        assert "secret" not in out.lower()
        assert "password" not in out.lower()


# --------------------------- e2e flows ---------------------------------------


def test_e2e_us_equity_trader_picks_alpaca() -> None:
    """A US equity trader needs NYSE + market orders + ETF support."""

    candidates = brokers_supporting_exchange(Exchange.NYSE)
    candidate_set = {p.broker for p in candidates}
    # Alpaca paper is in there
    assert Broker.ALPACA_PAPER in candidate_set


def test_e2e_saudi_equity_trader_picks_saxo() -> None:
    """Saudi operator needs TADAWUL access — only Saxo provides it."""

    candidates = brokers_supporting_exchange(Exchange.TADAWUL)
    saxo_only = {Broker.SAXO_SIM, Broker.SAXO_LIVE}
    assert {p.broker for p in candidates} == saxo_only


def test_e2e_crypto_trader_picks_binance_or_coinbase() -> None:
    candidates = brokers_supporting_asset_class(AssetClass.CRYPTO)
    crypto_set = {p.broker for p in candidates}
    assert crypto_set == {
        Broker.BINANCE_TESTNET,
        Broker.BINANCE_LIVE,
        Broker.COINBASE_SANDBOX,
        Broker.COINBASE_LIVE,
    }


def test_e2e_paper_safety_gate() -> None:
    """Pin: a contributor wiring up the executor with `is_paper(broker)`
    can route paper-only flows to paper brokers and real flows to real
    brokers without crossing wires."""

    assert is_paper(Broker.ALPACA_PAPER) is True
    assert is_paper(Broker.ALPACA_LIVE) is False
    assert is_paper(Broker.BINANCE_TESTNET) is True
    assert is_paper(Broker.BINANCE_LIVE) is False
    assert is_paper(Broker.IBKR_PAPER) is True
    assert is_paper(Broker.IBKR_LIVE) is False
    assert is_paper(Broker.SAXO_SIM) is True
    assert is_paper(Broker.SAXO_LIVE) is False
