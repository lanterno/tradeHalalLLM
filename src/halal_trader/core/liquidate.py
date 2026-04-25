"""Panic-button auto-liquidation for the operator kill-switch.

Used by `halal-trader halt --close-all`. Connects to the broker(s),
places market-SELL on every open position above the dust threshold, then
returns a result list the CLI can display.

Design notes
------------
* Crypto: Binance has no "close everything" call. We iterate the account
  balances, filter to the configured trading pairs (so we don't fire
  SELLs on assets the bot didn't open — e.g. user-deposited BNB), and
  place a market-SELL per asset. Notional below `_DUST_NOTIONAL_USD` is
  skipped on the broker side (the exchange would reject it anyway), so
  we drop it before placing.
* Stock: Alpaca exposes `close_all_positions(cancel_orders=True)` which
  is exactly what we want. One call.
* Both paths are best-effort. Failures are surfaced individually rather
  than aborting the rest of the close-all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from halal_trader.crypto.exchange import DUST_NOTIONAL_USD as _DUST_NOTIONAL_USD

logger = logging.getLogger(__name__)


@dataclass
class LiquidationResult:
    market: str  # 'crypto' | 'stocks'
    symbol: str
    quantity: float
    status: str  # 'closed' | 'skipped' | 'error'
    detail: str = ""


async def liquidate_crypto(broker: Any, configured_pairs: list[str]) -> list[LiquidationResult]:
    """Place market-SELL on every tracked asset that has a non-dust balance."""
    results: list[LiquidationResult] = []

    tracked_bases = {p.upper().removesuffix("USDT").removesuffix("BUSD") for p in configured_pairs}

    try:
        balances = await broker.get_balances()
    except Exception as exc:
        logger.error("liquidate_crypto: failed to fetch balances: %s", exc)
        return [
            LiquidationResult(
                market="crypto",
                symbol="*",
                quantity=0.0,
                status="error",
                detail=f"get_balances failed: {exc}",
            )
        ]

    for balance in balances:
        asset = balance.asset.upper()
        if asset in {"USDT", "BUSD", "USDC"} or asset not in tracked_bases:
            continue
        free = float(balance.free)
        if free <= 0:
            continue

        symbol = f"{asset}USDT"
        try:
            price = await broker.get_ticker_price(symbol)
        except Exception as exc:
            results.append(
                LiquidationResult(
                    market="crypto",
                    symbol=symbol,
                    quantity=free,
                    status="error",
                    detail=f"price fetch failed: {exc}",
                )
            )
            continue

        if price <= 0 or free * price < _DUST_NOTIONAL_USD:
            results.append(
                LiquidationResult(
                    market="crypto",
                    symbol=symbol,
                    quantity=free,
                    status="skipped",
                    detail=f"dust below ${_DUST_NOTIONAL_USD:.0f}",
                )
            )
            continue

        try:
            qty = broker.round_quantity(symbol, free)
        except Exception as exc:
            results.append(
                LiquidationResult(
                    market="crypto",
                    symbol=symbol,
                    quantity=free,
                    status="error",
                    detail=f"round_quantity failed: {exc}",
                )
            )
            continue

        if qty <= 0:
            results.append(
                LiquidationResult(
                    market="crypto",
                    symbol=symbol,
                    quantity=free,
                    status="skipped",
                    detail="quantity rounded to zero (lot size)",
                )
            )
            continue

        try:
            await broker.place_order(
                symbol=symbol,
                side="SELL",
                quantity=qty,
                order_type="MARKET",
            )
            results.append(
                LiquidationResult(
                    market="crypto",
                    symbol=symbol,
                    quantity=qty,
                    status="closed",
                )
            )
        except Exception as exc:
            results.append(
                LiquidationResult(
                    market="crypto",
                    symbol=symbol,
                    quantity=qty,
                    status="error",
                    detail=str(exc),
                )
            )

    return results


async def liquidate_stocks(broker: Any) -> list[LiquidationResult]:
    """Close every open Alpaca position via the broker's batch endpoint."""
    try:
        await broker.close_all_positions()
    except Exception as exc:
        return [
            LiquidationResult(
                market="stocks",
                symbol="*",
                quantity=0.0,
                status="error",
                detail=str(exc),
            )
        ]

    try:
        positions = await broker.get_all_positions()
    except Exception:
        positions = []

    return [
        LiquidationResult(
            market="stocks",
            symbol=p.symbol,
            quantity=float(p.qty),
            status="closed",
            detail="batch close_all_positions",
        )
        for p in positions
    ]
