"""Per-table repository protocols.

Each protocol is a typed view over the part of the data layer one
caller needs. Tests mock the narrow Protocol; real code passes the
full ``Repository`` (which structurally satisfies every protocol
via duck typing).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Protocol


class TradeRepo(Protocol):
    async def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = ...,
        order_id: str | None = ...,
        status: str = ...,
        llm_reasoning: str | None = ...,
        submitted_at: datetime | None = ...,
        filled_at: datetime | None = ...,
        filled_price: float | None = ...,
        filled_quantity: float | None = ...,
        halal_screening_id: int | None = ...,
        stop_loss: float | None = ...,
        target_price: float | None = ...,
    ) -> int: ...

    async def get_today_trades(self) -> list[dict[str, Any]]: ...
    async def get_recent_trades(self, limit: int = ...) -> list[dict[str, Any]]: ...
    async def close_trade(self, trade_id: int, exit_price: float, exit_reason: str) -> None: ...


class PnlRepo(Protocol):
    async def start_crypto_day(self, starting_equity: float) -> None: ...
    async def end_crypto_day(
        self, *, ending_equity: float, realized_pnl: float, trades_count: int
    ) -> None: ...
    async def get_crypto_pnl_history(self, limit: int = ...) -> list[dict[str, Any]]: ...


class HalalCacheRepo(Protocol):
    async def get_crypto_halal_status(self, symbol: str) -> str | None: ...
    async def get_crypto_halal_symbols(self) -> list[str]: ...
    async def cache_crypto_halal_status(
        self,
        symbol: str,
        compliance: str,
        category: str | None = ...,
        market_cap: float | None = ...,
        screening_criteria: dict[str, Any] | None = ...,
    ) -> None: ...


class HalalScreeningRepo(Protocol):
    async def record_halal_screening(
        self,
        *,
        symbol: str,
        asset_class: str,
        source: str,
        decision: str,
        criteria: dict[str, Any] | None = ...,
        cache_hit: bool = ...,
    ) -> int: ...

    async def get_halal_screening(self, screening_id: int) -> dict[str, Any] | None: ...


class RuntimeConfigRepo(Protocol):
    async def set_runtime_config(
        self, key: str, value: Any, *, set_by: str | None = ...
    ) -> None: ...
    async def delete_runtime_config(self, key: str) -> bool: ...
    async def list_runtime_config(self) -> dict[str, Any]: ...


class ResearchJobRepo(Protocol):
    async def enqueue_research_job(
        self, *, kind: str, params: dict[str, Any], name: str | None = ...
    ) -> int: ...

    async def update_research_job(
        self,
        job_id: int,
        *,
        status: str,
        result: dict[str, Any] | None = ...,
        error: str | None = ...,
    ) -> None: ...

    async def get_research_job(self, job_id: int) -> dict[str, Any] | None: ...
    async def list_research_jobs(self, limit: int = ...) -> list[dict[str, Any]]: ...
    async def pin_research_job(self, job_id: int, pinned: bool) -> bool: ...


class WebAuditRepo(Protocol):
    async def begin_web_action(
        self,
        *,
        actor: str,
        method: str,
        path: str,
        payload: str | None = ...,
    ) -> int: ...

    async def complete_web_action(
        self, action_id: int, *, status_code: int, error: str | None = ...
    ) -> None: ...

    async def get_recent_web_actions(self, limit: int = ...) -> list[dict[str, Any]]: ...

    async def delete_old_web_actions(self, *, older_than: timedelta) -> int: ...


class IndicatorSnapshotRepo(Protocol):
    async def record_indicator_snapshot(
        self, *, trade_id: int, pair: str, indicators: dict[str, Any]
    ) -> int: ...

    async def get_indicator_snapshots(
        self, *, limit: int = ..., labelled_only: bool = ...
    ) -> list[dict[str, Any]]: ...


class LlmDecisionRepo(Protocol):
    async def record_decision(
        self,
        provider: str,
        model: str,
        prompt_summary: str | None = ...,
        raw_response: str | None = ...,
        parsed_action: dict[str, Any] | None = ...,
        symbols: list[str] | None = ...,
        execution_ms: int | None = ...,
        thinking: str | None = ...,
        prompt_version: str | None = ...,
        input_tokens: int | None = ...,
        output_tokens: int | None = ...,
        cache_read_tokens: int | None = ...,
        cache_write_tokens: int | None = ...,
        cost_usd: float | None = ...,
    ) -> int: ...

    async def get_recent_decisions(self, limit: int = ...) -> list[dict[str, Any]]: ...


class PurificationRepo(Protocol):
    async def record_purification(
        self,
        *,
        symbol: str,
        dividend_usd: float,
        haram_pct: float,
        purification_usd: float,
        notes: str | None = ...,
    ) -> int: ...

    async def mark_purification_paid(
        self, entry_id: int, paid_at: datetime | None = ...
    ) -> bool: ...

    async def get_outstanding_purification(self) -> list[dict[str, Any]]: ...
    async def get_purification_totals(self) -> dict[str, float]: ...


class PairPauseRepo(Protocol):
    async def pause_pair(
        self, pair: str, *, set_by: str | None = ..., reason: str | None = ...
    ) -> None: ...

    async def resume_pair(self, pair: str) -> bool: ...
    async def list_pair_pauses(self) -> list[dict[str, Any]]: ...
    async def get_paused_pairs(self) -> set[str]: ...
