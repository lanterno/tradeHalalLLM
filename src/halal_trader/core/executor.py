"""Base executor – shared sells-first-then-buys execution flow."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from halal_trader.domain.ports import TradeRepository

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Abstract base for trade executors (stocks and crypto).

    Subclasses provide broker-specific buy/sell logic; the base class
    handles the common sells-first-then-buys orchestration and
    max-positions enforcement.
    """

    def __init__(
        self,
        repo: TradeRepository,
        *,
        max_position_pct: float,
        max_simultaneous_positions: int,
    ) -> None:
        self._repo = repo
        self._max_position_pct = max_position_pct
        self._max_simultaneous_positions = max_simultaneous_positions

    async def _execute_plan_common(
        self,
        plan: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Shared execution flow: sells first, then buys with position cap."""
        results: list[dict[str, Any]] = []

        for decision in self._get_sells(plan):
            result = await self._execute_sell(decision, **kwargs)
            results.append(result)

        open_count = await self._get_current_position_count(**kwargs)

        for decision in self._get_buys(plan):
            if open_count >= self._max_simultaneous_positions:
                msg = (
                    f"Max simultaneous positions ({self._max_simultaneous_positions}) "
                    f"reached — skipping BUY {decision.symbol}"
                )
                logger.warning(msg)
                results.append({
                    "symbol": decision.symbol,
                    "action": "buy",
                    "status": "rejected",
                    "reason": msg,
                })
                continue
            result = await self._execute_buy(decision, **kwargs)
            if result.get("status") in ("submitted", "filled"):
                open_count += 1
            results.append(result)

        return results

    @abstractmethod
    def _get_sells(self, plan: Any) -> list[Any]:
        """Extract sell decisions from the plan."""
        ...

    @abstractmethod
    def _get_buys(self, plan: Any) -> list[Any]:
        """Extract buy decisions from the plan."""
        ...

    @abstractmethod
    async def _get_current_position_count(self, **kwargs: Any) -> int:
        """Return the number of currently open positions."""
        ...

    @abstractmethod
    async def _execute_buy(self, decision: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute a single buy order and return a result dict."""
        ...

    @abstractmethod
    async def _execute_sell(self, decision: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute a single sell order and return a result dict."""
        ...
