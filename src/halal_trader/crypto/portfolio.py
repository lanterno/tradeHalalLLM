"""Crypto portfolio and P&L tracking."""

import logging
from datetime import UTC, datetime
from typing import Any

from halal_trader.core.portfolio import BasePortfolioTracker
from halal_trader.crypto.exchange import BinanceClient
from halal_trader.db.models import CryptoTrade
from halal_trader.domain.models import CryptoBalance
from halal_trader.domain.ports import TradeRepository

logger = logging.getLogger(__name__)


class CryptoPortfolioTracker(BasePortfolioTracker):
    """Tracks crypto portfolio state and daily P&L."""

    _DEFAULT_EQUITY: float = 10_000.0
    _label: str = "Crypto "

    def __init__(
        self,
        broker: BinanceClient,
        repo: TradeRepository,
        *,
        daily_loss_limit: float,
    ) -> None:
        super().__init__(repo, daily_loss_limit=daily_loss_limit)
        self._broker = broker

    # ── Hook implementations ───────────────────────────────────

    async def _get_equity(self, **kwargs: Any) -> float:
        account = kwargs.get("account")
        if account is None:
            account = await self._broker.get_account()
        return account.total_balance_usdt or self._DEFAULT_EQUITY

    async def _get_today_trades(self) -> list[dict[str, Any]]:
        return await self._repo.get_today_crypto_trades()

    async def _persist_day_start(self, equity: float) -> None:
        await self._repo.start_crypto_day(equity)

    async def _persist_day_end(
        self, equity: float, pnl: float, count: int
    ) -> None:
        await self._repo.end_crypto_day(
            ending_equity=equity,
            realized_pnl=pnl,
            trades_count=count,
        )

    # ── Crypto-specific methods ────────────────────────────────

    async def get_open_trades(self) -> list[CryptoTrade]:
        """Return buy trades that haven't been closed yet."""
        return await self._repo.get_open_crypto_trades()

    async def get_balances_summary(self) -> list[CryptoBalance]:
        """Get all current balances."""
        return await self._broker.get_balances()

    def format_positions_for_prompt(
        self,
        balances: list[CryptoBalance],
        configured_pairs: list[str] | None = None,
        open_trades: list[CryptoTrade] | None = None,
        current_prices: dict[str, float] | None = None,
    ) -> str:
        """Format current balances with entry price, unrealized P&L, and hold duration."""
        if configured_pairs:
            relevant_assets = {
                p.upper().removesuffix("USDT").removesuffix("BUSD")
                for p in configured_pairs
            }
            relevant_assets.add("USDT")
        else:
            relevant_assets = None

        trade_by_asset: dict[str, CryptoTrade] = {}
        if open_trades:
            for t in open_trades:
                asset = t.pair.upper().removesuffix("USDT").removesuffix("BUSD")
                trade_by_asset[asset] = t

        now = datetime.now(UTC)
        lines = []
        for b in balances:
            if b.free + b.locked <= 0:
                continue
            if relevant_assets and b.asset not in relevant_assets:
                continue
            if b.asset == "USDT":
                lines.append(f"  USDT (cash): {b.free:.2f} (locked: {b.locked:.2f})")
                continue

            line = f"  {b.asset}: {b.free:.8f}"
            trade = trade_by_asset.get(b.asset)
            if trade and trade.entry_price:
                price = (current_prices or {}).get(f"{b.asset}USDT")
                entry_str = f"entry: ${trade.entry_price:,.2f}"
                pnl_str = ""
                if price:
                    unrealized = (price - trade.entry_price) * b.free
                    pnl_pct = (price - trade.entry_price) / trade.entry_price * 100
                    pnl_str = f", unrealized: ${unrealized:+,.2f} ({pnl_pct:+.1f}%)"
                held_str = ""
                if trade.timestamp:
                    ts = trade.timestamp
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    held_min = (now - ts).total_seconds() / 60
                    if held_min < 60:
                        held_str = f", held: {held_min:.0f}m"
                    else:
                        held_str = f", held: {held_min / 60:.1f}h"
                line += f" ({entry_str}{pnl_str}{held_str})"
            elif b.locked > 0:
                line += f" (locked: {b.locked:.8f})"

            lines.append(line)

        return "\n".join(lines) if lines else "No open positions."
