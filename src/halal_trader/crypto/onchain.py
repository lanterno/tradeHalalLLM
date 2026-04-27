"""On-chain whale-flow signal via Etherscan (free tier).

Big stablecoin transfers to centralised exchanges and big native-token
withdrawals from exchanges are two of the cleanest non-price signals
on Ethereum. The classical reads:

* **Stablecoin → exchange (inflow).** Capital is moving onto the
  trading floor. Often precedes a buy.
* **Token → exchange (inflow).** Holder is preparing to sell. Often
  precedes a sell.
* **Token ← exchange (outflow).** Holder is taking custody / staking.
  Bullish in aggregate.

This module gives the rest of the bot a ``WhaleFlowSignal`` per pair:
net token flow over the last ``window_minutes``, scored as
``inflow_pressure`` in [-1, +1] where positive = sell pressure
incoming, negative = withdrawal / supply leaving.

Design choices:
* **Free Etherscan tier** (5 req/sec). 1 request per pair per cycle is
  fine; we cache for 5 minutes.
* **Static "exchange wallet" list** for the major venues — extending
  it is one dict edit. Wallet labels rarely change.
* **Empty key disables.** The hub stays empty and prompt section
  remains an ``""`` so the prompt builder can elide it.
* No HTTP retries beyond the timeout — failed cycles get fresh data
  on the next pass, no point amplifying load.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_API_BASE = "https://api.etherscan.io/api"
_CACHE_TTL_S = 5 * 60  # 5 minutes


# ── Token + exchange-wallet registry ──────────────────────────────


@dataclass(frozen=True)
class TokenSpec:
    """One ERC-20 we know how to watch."""

    symbol: str  # 'USDT', 'USDC', 'WETH', ...
    contract: str  # ERC-20 contract address (lowercase 0x…)
    decimals: int  # token's decimals (used to convert wei → human units)


# Curated list of the most-used ERC-20s for whale-flow detection.
# Extending this is one dict entry per token.
TOKENS: dict[str, TokenSpec] = {
    "USDT": TokenSpec(
        symbol="USDT",
        contract="0xdac17f958d2ee523a2206206994597c13d831ec7",
        decimals=6,
    ),
    "USDC": TokenSpec(
        symbol="USDC",
        contract="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        decimals=6,
    ),
    "DAI": TokenSpec(
        symbol="DAI",
        contract="0x6b175474e89094c44da98b954eedeac495271d0f",
        decimals=18,
    ),
    "WETH": TokenSpec(
        symbol="WETH",
        contract="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        decimals=18,
    ),
}

# Big centralised-exchange hot wallets, lowercased. Source: public
# Etherscan labels and well-known holders. The set is small on purpose
# — we want true positives over coverage. Extending is one address per
# venue.
EXCHANGE_WALLETS: set[str] = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 14
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 15
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 16
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 17
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",  # Coinbase 1
    "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase 2
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",  # OKX
    "0x236f9f97e0e62388479bf9e5ba4889e46b0273c3",  # OKX 2
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",  # Kraken 1
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13",  # Kraken 2
}


# ── Signal ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WhaleFlowSignal:
    """Net on-chain flow for one token over the recent window."""

    symbol: str
    inflow_to_exchange_usd: float
    outflow_from_exchange_usd: float
    inflow_pressure: float  # [-1, +1]; +1 = pure inflow (sell pressure)
    n_transfers: int

    @property
    def label(self) -> str:
        if self.inflow_pressure > 0.5:
            return "exchange_inflow_heavy"
        if self.inflow_pressure < -0.5:
            return "exchange_outflow_heavy"
        return "balanced"


# ── Fetcher ───────────────────────────────────────────────────────


@dataclass
class _CacheEntry:
    fetched_at: float
    signal: WhaleFlowSignal | None


@dataclass
class EtherscanWhaleFlow:
    """Pulls whale-flow signals from Etherscan's free ERC-20 transfer API.

    Per-token cycle:
    1. Hit ``api?module=account&action=tokentx&contractaddress=...``.
    2. Bucket the last N transfers by direction relative to known
       exchange wallets.
    3. Compute USD-equivalent flow (stablecoins are ~$1; native tokens
       use a passed-in price).
    4. Score ``inflow_pressure`` in [-1, +1].

    The class accepts an optional injected ``httpx.AsyncClient`` so
    tests use a transport mock; in production it owns one.
    """

    api_key: str
    window_minutes: int = 60
    min_transfer_usd: float = 100_000.0  # below this, ignore as noise
    _client: Any | None = None
    _cache: dict[str, _CacheEntry] = field(default_factory=dict)

    async def fetch(
        self,
        symbols: Sequence[str],
        *,
        prices: dict[str, float] | None = None,
    ) -> dict[str, WhaleFlowSignal]:
        """Return ``{symbol: WhaleFlowSignal}`` for tokens we can watch.

        ``prices`` is a USD price per non-stablecoin (e.g. ``{"WETH":
        2400.0}``); stablecoins are treated as ~$1 regardless. Symbols
        outside our registry are silently skipped.
        """
        if not symbols or not self.api_key:
            return {}
        prices = prices or {}
        out: dict[str, WhaleFlowSignal] = {}
        for symbol in symbols:
            sym = symbol.upper()
            spec = TOKENS.get(sym)
            if spec is None:
                continue
            try:
                sig = await self._fetch_for_token(spec, prices.get(sym, 1.0))
            except Exception as exc:  # noqa: BLE001
                logger.debug("etherscan whale-flow failed for %s: %s", sym, exc)
                continue
            if sig is not None:
                out[sym] = sig
        return out

    async def _fetch_for_token(self, spec: TokenSpec, price_usd: float) -> WhaleFlowSignal | None:
        cache_key = f"{spec.symbol}:{self.window_minutes}"
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached.fetched_at) < _CACHE_TTL_S:
            return cached.signal

        client = await self._get_client()
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": spec.contract,
            "page": 1,
            "offset": 100,  # max 100 most-recent transfers
            "sort": "desc",
            "apikey": self.api_key,
        }
        resp = await client.get(_API_BASE, params=params)
        if resp.status_code != 200:
            logger.debug("etherscan returned %d for %s", resp.status_code, spec.symbol)
            return None
        body = resp.json()
        if str(body.get("status")) != "1":
            # status=0 with message "No transactions found" is normal for
            # tokens with no recent activity; everything else is an error.
            return None

        cutoff_ts = int(time.time()) - self.window_minutes * 60
        inflow_usd = 0.0
        outflow_usd = 0.0
        n = 0
        for tx in body.get("result", []) or []:
            try:
                ts = int(tx.get("timeStamp", 0))
            except TypeError, ValueError:
                continue
            if ts < cutoff_ts:
                break  # results are sort=desc
            try:
                value_units = int(tx.get("value", 0)) / (10**spec.decimals)
            except TypeError, ValueError:
                continue
            usd = value_units * price_usd
            if usd < self.min_transfer_usd:
                continue
            from_addr = str(tx.get("from", "")).lower()
            to_addr = str(tx.get("to", "")).lower()
            if to_addr in EXCHANGE_WALLETS and from_addr not in EXCHANGE_WALLETS:
                inflow_usd += usd
                n += 1
            elif from_addr in EXCHANGE_WALLETS and to_addr not in EXCHANGE_WALLETS:
                outflow_usd += usd
                n += 1

        total = inflow_usd + outflow_usd
        pressure = (inflow_usd - outflow_usd) / total if total > 0 else 0.0
        sig = WhaleFlowSignal(
            symbol=spec.symbol,
            inflow_to_exchange_usd=inflow_usd,
            outflow_from_exchange_usd=outflow_usd,
            inflow_pressure=pressure,
            n_transfers=n,
        )
        self._cache[cache_key] = _CacheEntry(fetched_at=time.monotonic(), signal=sig)
        return sig

    async def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


# ── Prompt formatting ─────────────────────────────────────────────


def format_whale_flows_for_prompt(
    signals: Iterable[WhaleFlowSignal] | dict[str, WhaleFlowSignal],
) -> str:
    """Compact one-block summary of meaningful flows for the LLM prompt.

    Empty input → empty string so the prompt builder can elide.
    Ignores 'balanced' signals (low information value).
    """
    if isinstance(signals, dict):
        signals = list(signals.values())
    rows = [s for s in signals if s.label != "balanced"]
    if not rows:
        return ""
    lines = ["On-chain whale flows (Etherscan, last hour):"]
    rows.sort(key=lambda s: abs(s.inflow_pressure), reverse=True)
    for s in rows[:5]:
        sign = "+" if s.inflow_pressure >= 0 else ""
        lines.append(
            f"  {s.symbol:<6} pressure={sign}{s.inflow_pressure:.2f} "
            f"(in ${s.inflow_to_exchange_usd:,.0f}, "
            f"out ${s.outflow_from_exchange_usd:,.0f}, "
            f"n={s.n_transfers}) — {s.label}"
        )
    return "\n".join(lines)
