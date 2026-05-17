"""LLM decision audit repository — every prompt/response and cost row.

Wave D extraction. Records token counts + cost per call so the
dashboard can plot daily spend, and stashes the raw response for
post-hoc inspection of any cycle. Matching protocol in ``protocols.py``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import LlmDecision


class LlmDecisionRepoImpl:
    """Concrete implementation of :class:`LlmDecisionRepo`."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def record_decision(
        self,
        provider: str,
        model: str,
        prompt_summary: str | None = None,
        raw_response: str | None = None,
        parsed_action: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        execution_ms: int | None = None,
        thinking: str | None = None,
        prompt_version: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> int:
        decision = LlmDecision(
            provider=provider,
            model=model,
            prompt_summary=prompt_summary,
            raw_response=raw_response,
            parsed_action=parsed_action,
            symbols=symbols,
            execution_ms=execution_ms,
            thinking=thinking,
            prompt_version=prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
        )
        async with AsyncSession(self._engine) as session:
            session.add(decision)
            await session.commit()
            await session.refresh(decision)
            assert decision.id is not None
            return decision.id

    async def get_recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with AsyncSession(self._engine) as session:
            statement = select(LlmDecision).order_by(col(LlmDecision.timestamp).desc()).limit(limit)
            results = await session.exec(statement)
            return [r.model_dump() for r in results.all()]
