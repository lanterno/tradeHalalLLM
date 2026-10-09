"""The evening LLM-credit check: the account balance, not only the key's limit."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.heartbeat import read_beat
from halal_trader.core.llm.credits import LLM_CREDITS, Credits, check

TODAY = date(2026, 10, 9)


def _credits(balance: float | None, remaining: float | None, pace: float) -> Credits:
    return Credits(balance, remaining, pace, "2026-10-09T20:30:00+00:00")


def test_the_lower_of_balance_and_key_limit_is_what_is_left() -> None:
    c = _credits(0.10, 39.81, 0.5)  # 2026-10-09: the balance ran out first
    assert c.available_usd == 0.10
    assert c.days_left == pytest.approx(0.2)
    assert c.problem() is not None and "$0.10 left" in c.problem()  # type: ignore[operator]


def test_a_week_of_spending_left_is_fine_and_less_is_not() -> None:
    assert _credits(20.0, None, 1.0).problem() is None  # 20 days
    low = _credits(5.0, 30.0, 1.0).problem()  # 5 days < 7
    assert low is not None and "about 5.0 days" in low
    assert _credits(1.0, 50.0, 0.0).problem() is not None  # under the floor, whatever the pace
    assert _credits(None, None, 1.0).problem() is None  # nothing published: nothing to judge


def _openrouter(balance: tuple[float, float], remaining: float | None) -> httpx.AsyncClient:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer sk-or-test"
        if req.url.path.endswith("/credits"):
            data = {"total_credits": balance[0], "total_usage": balance[1]}
            return httpx.Response(200, json={"data": data})
        return httpx.Response(200, json={"data": {"limit_remaining": remaining}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _settings(base_url: str = "https://openrouter.ai/api/v1") -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(glm=SimpleNamespace(api_key="sk-or-test", base_url=base_url))
    )


async def test_the_evening_check_records_the_credit_and_reports_a_low_one(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO llm_spend (day, consumer, calls, spent_usd) VALUES "
                "('2026-10-08', 'stock', 100, 3.5), ('2026-10-09', 'shadow', 50, 3.5)"
            )
        )
    async with _openrouter((30.0, 29.9), 39.81) as client:
        problems = await check(engine, _settings(), TODAY, client=client)
    assert len(problems) == 1 and "$0.10 left" in problems[0]
    b = await read_beat(engine, LLM_CREDITS)
    assert b is not None and b.detail is not None
    assert b.detail["available_usd"] == pytest.approx(0.1)
    assert b.detail["pace_usd_per_day"] == pytest.approx(1.0)  # $7 over 7 days

    async with _openrouter((60.0, 29.9), 39.81) as client:
        assert await check(engine, _settings(), TODAY, client=client) == []


async def test_another_endpoint_is_not_checked(engine: AsyncEngine) -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        assert (
            await check(engine, _settings("https://api.z.ai/api/paas/v4"), TODAY, client=client)
            == []
        )
