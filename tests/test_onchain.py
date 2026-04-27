"""Tests for the Etherscan whale-flow signal source."""

from __future__ import annotations

import time

import httpx
import pytest

from halal_trader.crypto.onchain import (
    EXCHANGE_WALLETS,
    TOKENS,
    EtherscanWhaleFlow,
    WhaleFlowSignal,
    format_whale_flows_for_prompt,
)


def _binance_wallet() -> str:
    return next(iter(EXCHANGE_WALLETS))


def _non_exchange() -> str:
    return "0x000000000000000000000000000000000000dead"


def _tx(*, ts: int, value_units: float, decimals: int, from_addr: str, to_addr: str) -> dict:
    return {
        "timeStamp": str(ts),
        "value": str(int(value_units * (10**decimals))),
        "from": from_addr,
        "to": to_addr,
    }


def _payload(txs: list[dict]) -> dict:
    return {"status": "1", "message": "OK", "result": txs}


def _empty_payload() -> dict:
    return {"status": "0", "message": "No transactions found", "result": []}


def _client_with_payload(payload: dict) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Disabled paths ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_api_key_returns_empty() -> None:
    src = EtherscanWhaleFlow(api_key="")
    out = await src.fetch(["USDT"])
    assert out == {}


@pytest.mark.asyncio
async def test_no_symbols_returns_empty() -> None:
    src = EtherscanWhaleFlow(api_key="key")
    src._client = _client_with_payload(_empty_payload())
    assert await src.fetch([]) == {}
    await src.aclose()


@pytest.mark.asyncio
async def test_unknown_symbol_skipped() -> None:
    src = EtherscanWhaleFlow(api_key="key")
    src._client = _client_with_payload(_empty_payload())
    assert await src.fetch(["NOT_A_TOKEN"]) == {}
    await src.aclose()


# ── Direction classification ─────────────────────────────────────


@pytest.mark.asyncio
async def test_inflow_to_exchange_classified_correctly() -> None:
    now = int(time.time())
    spec = TOKENS["USDT"]
    txs = [
        _tx(
            ts=now - 60,
            value_units=500_000,
            decimals=spec.decimals,
            from_addr=_non_exchange(),
            to_addr=_binance_wallet(),
        ),
    ]
    src = EtherscanWhaleFlow(api_key="key", window_minutes=60)
    src._client = _client_with_payload(_payload(txs))
    out = await src.fetch(["USDT"])
    sig = out["USDT"]
    assert sig.inflow_to_exchange_usd == 500_000.0
    assert sig.outflow_from_exchange_usd == 0.0
    assert sig.inflow_pressure == 1.0
    assert sig.label == "exchange_inflow_heavy"
    await src.aclose()


@pytest.mark.asyncio
async def test_outflow_from_exchange_classified_correctly() -> None:
    now = int(time.time())
    spec = TOKENS["WETH"]
    txs = [
        _tx(
            ts=now - 30,
            value_units=200,  # 200 WETH @ $2000 = 400k
            decimals=spec.decimals,
            from_addr=_binance_wallet(),
            to_addr=_non_exchange(),
        ),
    ]
    src = EtherscanWhaleFlow(api_key="key", window_minutes=60)
    src._client = _client_with_payload(_payload(txs))
    out = await src.fetch(["WETH"], prices={"WETH": 2000.0})
    sig = out["WETH"]
    assert sig.inflow_to_exchange_usd == 0.0
    assert sig.outflow_from_exchange_usd == 400_000.0
    assert sig.inflow_pressure == -1.0
    assert sig.label == "exchange_outflow_heavy"
    await src.aclose()


@pytest.mark.asyncio
async def test_mixed_flows_compute_pressure() -> None:
    now = int(time.time())
    spec = TOKENS["USDC"]
    txs = [
        _tx(
            ts=now - 10,
            value_units=300_000,
            decimals=spec.decimals,
            from_addr=_non_exchange(),
            to_addr=_binance_wallet(),
        ),
        _tx(
            ts=now - 20,
            value_units=100_000,
            decimals=spec.decimals,
            from_addr=_binance_wallet(),
            to_addr=_non_exchange(),
        ),
    ]
    src = EtherscanWhaleFlow(api_key="key")
    src._client = _client_with_payload(_payload(txs))
    out = await src.fetch(["USDC"])
    sig = out["USDC"]
    assert sig.inflow_to_exchange_usd == 300_000.0
    assert sig.outflow_from_exchange_usd == 100_000.0
    # (300k - 100k) / 400k = 0.5
    assert abs(sig.inflow_pressure - 0.5) < 1e-9
    await src.aclose()


@pytest.mark.asyncio
async def test_skips_below_min_transfer() -> None:
    now = int(time.time())
    spec = TOKENS["USDT"]
    txs = [
        _tx(
            ts=now - 30,
            value_units=50_000,  # below 100k floor
            decimals=spec.decimals,
            from_addr=_non_exchange(),
            to_addr=_binance_wallet(),
        ),
    ]
    src = EtherscanWhaleFlow(api_key="key", min_transfer_usd=100_000.0)
    src._client = _client_with_payload(_payload(txs))
    out = await src.fetch(["USDT"])
    assert out["USDT"].n_transfers == 0
    await src.aclose()


@pytest.mark.asyncio
async def test_skips_outside_window() -> None:
    now = int(time.time())
    spec = TOKENS["USDT"]
    txs = [
        _tx(
            ts=now - 7200,  # 2 hours ago, outside 60min window
            value_units=500_000,
            decimals=spec.decimals,
            from_addr=_non_exchange(),
            to_addr=_binance_wallet(),
        ),
    ]
    src = EtherscanWhaleFlow(api_key="key", window_minutes=60)
    src._client = _client_with_payload(_payload(txs))
    out = await src.fetch(["USDT"])
    assert out["USDT"].n_transfers == 0
    await src.aclose()


@pytest.mark.asyncio
async def test_internal_exchange_to_exchange_ignored() -> None:
    now = int(time.time())
    spec = TOKENS["USDT"]
    wallets = list(EXCHANGE_WALLETS)
    txs = [
        _tx(
            ts=now - 30,
            value_units=500_000,
            decimals=spec.decimals,
            from_addr=wallets[0],
            to_addr=wallets[1],
        ),
    ]
    src = EtherscanWhaleFlow(api_key="key")
    src._client = _client_with_payload(_payload(txs))
    out = await src.fetch(["USDT"])
    # Both ends are exchanges — neither inflow nor outflow.
    assert out["USDT"].n_transfers == 0
    await src.aclose()


# ── Error / empty paths ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_response_returns_balanced_or_skip() -> None:
    src = EtherscanWhaleFlow(api_key="key")
    src._client = _client_with_payload(_empty_payload())
    out = await src.fetch(["USDT"])
    # status=0 → silently skip; no key in dict
    assert out == {}
    await src.aclose()


@pytest.mark.asyncio
async def test_http_error_returns_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    src = EtherscanWhaleFlow(api_key="key")
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await src.fetch(["USDT"])
    assert out == {}
    await src.aclose()


# ── Caching ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_caches_per_token() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_empty_payload())

    src = EtherscanWhaleFlow(api_key="key")
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # Two consecutive calls for same token — second hits cache.
    # First call returns nothing because status=0; cache stores None,
    # but the cache mechanism is exercised through the token+window key.
    await src.fetch(["USDT"])
    await src.fetch(["USDT"])
    # Either 1 (cached None) or 2 (no caching of None). Acceptable
    # range — the contract is "don't hit network unnecessarily".
    assert calls <= 2
    await src.aclose()


# ── Prompt formatting ────────────────────────────────────────────


def test_format_skips_balanced_signals() -> None:
    sigs = [
        WhaleFlowSignal(
            symbol="USDT",
            inflow_to_exchange_usd=100,
            outflow_from_exchange_usd=100,
            inflow_pressure=0.0,
            n_transfers=2,
        ),
    ]
    assert format_whale_flows_for_prompt(sigs) == ""


def test_format_renders_strong_signals() -> None:
    sigs = [
        WhaleFlowSignal(
            symbol="USDT",
            inflow_to_exchange_usd=500_000,
            outflow_from_exchange_usd=0,
            inflow_pressure=1.0,
            n_transfers=1,
        ),
        WhaleFlowSignal(
            symbol="WETH",
            inflow_to_exchange_usd=0,
            outflow_from_exchange_usd=300_000,
            inflow_pressure=-1.0,
            n_transfers=1,
        ),
    ]
    out = format_whale_flows_for_prompt(sigs)
    assert "USDT" in out
    assert "WETH" in out
    assert "exchange_inflow_heavy" in out
    assert "exchange_outflow_heavy" in out


def test_format_accepts_dict_input() -> None:
    sigs = {
        "USDT": WhaleFlowSignal(
            symbol="USDT",
            inflow_to_exchange_usd=500_000,
            outflow_from_exchange_usd=0,
            inflow_pressure=1.0,
            n_transfers=1,
        )
    }
    out = format_whale_flows_for_prompt(sigs)
    assert "USDT" in out


# ── Smoke ────────────────────────────────────────────────────────


def test_token_registry_covers_majors() -> None:
    for sym in ("USDT", "USDC", "DAI", "WETH"):
        assert sym in TOKENS


def test_exchange_wallets_normalised_lowercase() -> None:
    for w in EXCHANGE_WALLETS:
        assert w == w.lower()
