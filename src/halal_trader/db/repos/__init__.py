"""Per-table repositories.

Wave D introduces a per-table split of the monolithic ``Repository``:
each subset of related methods lives behind a focused ``Protocol``,
and ``RepoBundle`` is the composition root that hands them out.

For now the implementation continues to back onto the existing
``Repository`` class — splitting the implementation across multiple
files is the next mechanical pass once the Protocols stabilise. The
key win today is that callers can now type-hint ``TradeRepo`` and
``WebAuditRepo`` rather than the whole 956-line surface, so a test
that needs only one helper can mock just that protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from halal_trader.db.repos.protocols import (
    HalalCacheRepo,
    HalalScreeningRepo,
    IndicatorSnapshotRepo,
    LlmDecisionRepo,
    PairPauseRepo,
    PnlRepo,
    PurificationRepo,
    ResearchJobRepo,
    RuntimeConfigRepo,
    TradeRepo,
    WebAuditRepo,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from halal_trader.db.repository import Repository


@dataclass(frozen=True, slots=True)
class RepoBundle:
    """Typed bundle of per-table repos.

    Today every field is the same ``Repository`` instance; the typed
    fields exist so consumers depend on the narrower Protocol rather
    than the full class. Splitting the implementation across files
    is the natural next step once the Protocol shapes are settled.
    """

    trades: TradeRepo
    pnl: PnlRepo
    halal_cache: HalalCacheRepo
    halal_screening: HalalScreeningRepo
    runtime_config: RuntimeConfigRepo
    research_jobs: ResearchJobRepo
    web_audit: WebAuditRepo
    indicator_snapshots: IndicatorSnapshotRepo
    llm_decisions: LlmDecisionRepo
    purification: PurificationRepo
    pair_pauses: PairPauseRepo

    @classmethod
    def from_engine(cls, engine: "AsyncEngine") -> "RepoBundle":
        """Build a bundle backed by one ``Repository`` instance."""
        from halal_trader.db.repository import Repository

        repo = Repository(engine)
        return cls.from_repository(repo)

    @classmethod
    def from_repository(cls, repo: "Repository") -> "RepoBundle":
        return cls(
            trades=repo,
            pnl=repo,
            halal_cache=repo,
            halal_screening=repo,
            runtime_config=repo,
            research_jobs=repo,
            web_audit=repo,
            indicator_snapshots=repo,
            llm_decisions=repo,
            purification=repo,
            pair_pauses=repo,
        )


__all__ = [
    "HalalCacheRepo",
    "HalalScreeningRepo",
    "IndicatorSnapshotRepo",
    "LlmDecisionRepo",
    "PairPauseRepo",
    "PnlRepo",
    "PurificationRepo",
    "RepoBundle",
    "ResearchJobRepo",
    "RuntimeConfigRepo",
    "TradeRepo",
    "WebAuditRepo",
]
