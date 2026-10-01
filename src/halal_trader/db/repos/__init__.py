"""Per-table repositories.

Wave D split the monolithic ``Repository`` into per-table protocols:
each subset of related methods lives behind a focused ``Protocol``,
and :class:`RepoBundle` is the composition root that hands them out.

Every protocol field on :class:`RepoBundle` resolves to a dedicated
implementation (each ≤90 lines, mypy strict clean) under
``db/repos/<table>.py``. The bot's composition root and tests both
build a bundle via :meth:`RepoBundle.from_engine`; consumers depend
on the narrowest protocol they need rather than the full bundle.

Round-4 wave 0.D dropped the now-unused ``RepoBundle.from_repository``
shim — every call site builds from the engine directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from halal_trader.db.repos.protocols import (
    DailyRecommendationRepo,
    HalalScreeningRepo,
    IndicatorSnapshotRepo,
    LlmDecisionRepo,
    PurificationRepo,
    RuntimeConfigRepo,
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
    runtime_config: RuntimeConfigRepo
    daily_recommendations: DailyRecommendationRepo
    web_audit: WebAuditRepo
    indicator_snapshots: IndicatorSnapshotRepo
    llm_decisions: LlmDecisionRepo
    purification: PurificationRepo
    strategy_adjustments: StrategyAdjustmentRepo

    @classmethod
    def from_engine(cls, engine: "AsyncEngine") -> "RepoBundle":
        """Build a bundle directly from an engine (no Repository needed)."""
        from halal_trader.db.repos.daily_recommendations import (
            DailyRecommendationRepoImpl,
        )
        from halal_trader.db.repos.halal_screening import HalalScreeningRepoImpl
        from halal_trader.db.repos.indicator_snapshots import IndicatorSnapshotRepoImpl
        from halal_trader.db.repos.llm_decisions import LlmDecisionRepoImpl
        from halal_trader.db.repos.purification import PurificationRepoImpl
        from halal_trader.db.repos.runtime_config import RuntimeConfigRepoImpl
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
            runtime_config=RuntimeConfigRepoImpl(engine),
            daily_recommendations=DailyRecommendationRepoImpl(engine),
            web_audit=WebAuditRepoImpl(engine),
            indicator_snapshots=IndicatorSnapshotRepoImpl(engine),
            llm_decisions=LlmDecisionRepoImpl(engine),
            purification=PurificationRepoImpl(engine),
            strategy_adjustments=StrategyAdjustmentRepoImpl(engine),
        )


__all__ = [
    "DailyRecommendationRepo",
    "HalalScreeningRepo",
    "IndicatorSnapshotRepo",
    "LlmDecisionRepo",
    "PurificationRepo",
    "RepoBundle",
    "RuntimeConfigRepo",
    "StockHalalCacheRepo",
    "StockPnlRepo",
    "StrategyAdjustmentRepo",
    "TradeRepo",
    "WebAuditRepo",
]
