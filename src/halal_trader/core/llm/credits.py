"""What the LLM account can still pay for, checked each evening (research/daily.py).

The spend meter (spend.py) caps what this app spends; it cannot see what is
left to spend. OpenRouter bills a prepaid account balance, and the key's
monthly limit is only a ceiling on top of it: on 2026-10-09 the $30 balance
ran out with $39 of the key's $50 limit unused, and every GLM call (strategy,
classifier, shadow) began failing with 402. So each evening the account's
balance and the key's remaining limit are read; the lower of the two is what
can still be spent, and it is compared with the last week's metered pace.

The reading is kept in the ``llm.credits`` heartbeat (the Operations page
shows it); a balance under a week of spending, or under a floor, is reported
through the evening run's alert.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.http import request

logger = logging.getLogger(__name__)

LLM_CREDITS = "llm.credits"  # heartbeat component
LOW_DAYS = 7  # alert when what is left covers less than this many days
FLOOR_USD = 2.0  # ... or is under this, whatever the pace
PACE_DAYS = 7


@dataclass(frozen=True, slots=True)
class Credits:
    balance_usd: float | None  # the prepaid account: credits bought less used
    key_remaining_usd: float | None  # the key's monthly limit left (None: no limit)
    pace_usd_per_day: float  # metered spend, all consumers, over PACE_DAYS
    checked_at: str

    @property
    def available_usd(self) -> float | None:
        known = [v for v in (self.balance_usd, self.key_remaining_usd) if v is not None]
        return min(known) if known else None

    @property
    def days_left(self) -> float | None:
        a = self.available_usd
        return a / self.pace_usd_per_day if a is not None and self.pace_usd_per_day > 0 else None

    def problem(self) -> str | None:
        a = self.available_usd
        if a is None or (a >= FLOOR_USD and a >= LOW_DAYS * self.pace_usd_per_day):
            return None
        days = self.days_left
        pace_note = f" (about {days:.1f} days at the last week's pace)" if days is not None else ""
        return (
            f"LLM credit low: ${a:.2f} left on OpenRouter{pace_note}; add credits at "
            "openrouter.ai/settings/credits or every GLM call will fail"
        )


async def pace(engine: AsyncEngine, today: date) -> float:
    """Mean metered LLM spend per day over the last PACE_DAYS (UTC days, all consumers)."""
    async with engine.connect() as conn:
        total = (
            await conn.execute(
                text("SELECT coalesce(sum(spent_usd), 0) FROM llm_spend WHERE day > :d"),
                {"d": today - timedelta(days=PACE_DAYS)},
            )
        ).scalar()
    return float(total or 0) / PACE_DAYS


async def read(
    base_url: str, api_key: str, *, client: httpx.AsyncClient | None = None
) -> tuple[float | None, float | None]:
    """(account balance, key's remaining limit) in USD from OpenRouter."""
    own = client is None
    c = client or httpx.AsyncClient(timeout=20.0)
    headers = {"Authorization": f"Bearer {api_key}"}
    root = base_url.rstrip("/")
    try:
        credits = await request(
            c, "GET", f"{root}/credits", label="openrouter", retries=2, headers=headers
        )
        key = await request(c, "GET", f"{root}/key", label="openrouter", retries=2, headers=headers)
    finally:
        if own:
            await c.aclose()
    credits.raise_for_status()
    key.raise_for_status()
    cd: dict[str, Any] = credits.json().get("data") or {}
    kd: dict[str, Any] = key.json().get("data") or {}
    balance = (
        float(cd["total_credits"]) - float(cd["total_usage"])
        if cd.get("total_credits") is not None and cd.get("total_usage") is not None
        else None
    )
    remaining = kd.get("limit_remaining")
    return balance, float(remaining) if remaining is not None else None


async def check(
    engine: AsyncEngine, settings: Any, today: date, *, client: httpx.AsyncClient | None = None
) -> list[str]:
    """Read and record the LLM credit; the problems to alert (empty when fine).

    Only an OpenRouter endpoint publishes a balance; another endpoint is skipped.
    """
    from halal_trader.core.heartbeat import beat

    glm = settings.llm.glm
    if not glm.api_key or "openrouter.ai" not in glm.base_url:
        return []
    balance, remaining = await read(glm.base_url, glm.api_key, client=client)
    c = Credits(
        balance_usd=balance,
        key_remaining_usd=remaining,
        pace_usd_per_day=await pace(engine, today),
        checked_at=datetime.now(UTC).isoformat(),
    )
    detail = {**asdict(c), "available_usd": c.available_usd, "days_left": c.days_left}
    await beat(engine, LLM_CREDITS, detail)
    problem = c.problem()
    if problem:
        logger.error(problem)
    return [problem] if problem else []
