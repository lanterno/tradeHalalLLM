"""Per-table repositories.

Each subset of related data-access methods lives behind a focused
``Protocol`` in ``protocols.py``, implemented by a ``*RepoImpl`` class
under ``db/repos/<table>.py``. :class:`RepoBundle` is the composition
root that hands them out: build one with :meth:`RepoBundle.from_engine`
and depend on the narrowest protocol you need rather than the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from halal_trader.db.repos.protocols import (
    DailyRecommendationRepo,
    HalalScreeningRepo,
    LlmDecisionRepo,
    StockHalalCacheRepo,
    StockPnlRepo,
    StrategyAdjustmentRepo,
    TradeRepo,
    WebAuditRepo,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True, slots=True)
class RepoBundle:
    """Typed bundle of per-table repos.

    Each field is a narrow Protocol satisfied by a dedicated
    ``*RepoImpl`` class. Consumers should accept the field type they
    actually use (e.g. ``TradeRepo``) rather than the full
    bundle.
    """

    trades: TradeRepo
    stock_pnl: StockPnlRepo
    stock_halal_cache: StockHalalCacheRepo
    halal_screening: HalalScreeningRepo
    daily_recommendations: DailyRecommendationRepo
    web_audit: WebAuditRepo
    llm_decisions: LlmDecisionRepo
    strategy_adjustments: StrategyAdjustmentRepo

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> RepoBundle:
        """Build a bundle directly from an engine (no Repository needed)."""
        from halal_trader.db.repos.daily_recommendations import (
            DailyRecommendationRepoImpl,
        )
        from halal_trader.db.repos.halal_screening import HalalScreeningRepoImpl
        from halal_trader.db.repos.llm_decisions import LlmDecisionRepoImpl
        from halal_trader.db.repos.stock_halal_cache import StockHalalCacheRepoImpl
        from halal_trader.db.repos.stock_pnl import StockPnlRepoImpl
        from halal_trader.db.repos.strategy_adjustments import StrategyAdjustmentRepoImpl
        from halal_trader.db.repos.trades import TradeRepoImpl
        from halal_trader.db.repos.web_audit import WebAuditRepoImpl

        return cls(
            trades=TradeRepoImpl(engine),
            stock_pnl=StockPnlRepoImpl(engine),
            stock_halal_cache=StockHalalCacheRepoImpl(engine),
            halal_screening=HalalScreeningRepoImpl(engine),
            daily_recommendations=DailyRecommendationRepoImpl(engine),
            web_audit=WebAuditRepoImpl(engine),
            llm_decisions=LlmDecisionRepoImpl(engine),
            strategy_adjustments=StrategyAdjustmentRepoImpl(engine),
        )


__all__ = [
    "DailyRecommendationRepo",
    "HalalScreeningRepo",
    "LlmDecisionRepo",
    "RepoBundle",
    "StockHalalCacheRepo",
    "StockPnlRepo",
    "StrategyAdjustmentRepo",
    "TradeRepo",
    "WebAuditRepo",
]
