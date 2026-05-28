"""Portfolio view the policy reads + a hypothetical shadow book.

``PortfolioState`` is what ``deltas`` consults: current weight, *effective*
weight (filled + in-flight, so a working order doesn't trigger a duplicate —
R-14), whether we hold the asset (for hysteresis), and whether an order is
already working. :class:`ShadowPortfolio` is a pure hypothetical book used in
Phase 3: it tracks the weights the engine *would* hold from its own proposals,
has no real open orders, so effective == current.
"""

from __future__ import annotations

from typing import Protocol


class PortfolioState(Protocol):
    def weight(self, asset: str) -> float: ...
    def effective_weight(self, asset: str) -> float: ...  # filled + pending
    def holds(self, asset: str) -> bool: ...
    def has_open_order(self, asset: str) -> bool: ...
    def gross_exposure(self) -> float: ...


class ShadowPortfolio:
    """Hypothetical long-only book in weight space (no real orders)."""

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}

    def weight(self, asset: str) -> float:
        return self._weights.get(asset, 0.0)

    def effective_weight(self, asset: str) -> float:
        return self.weight(asset)  # no in-flight orders in shadow

    def holds(self, asset: str) -> bool:
        return self.weight(asset) > 0.0

    def has_position(self, asset: str) -> bool:
        # PositionSource alias for holds — lets the belief updater treat the
        # hypothetical book as "held" so the shadow exercises the live INV-7
        # lapsed-compliance forced-exit path (set_compliance reads this).
        return self.holds(asset)

    def has_open_order(self, asset: str) -> bool:
        return False

    def gross_exposure(self) -> float:
        return sum(self._weights.values())

    def set_weight(self, asset: str, weight: float) -> None:
        """Apply a (proposed) target — the shadow book moves to it."""
        if weight <= 0.0:
            self._weights.pop(asset, None)
        else:
            self._weights[asset] = weight
