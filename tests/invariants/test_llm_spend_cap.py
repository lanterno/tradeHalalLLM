"""INVARIANT: one daily LLM spend total, across processes and restarts, with a real stop.

The old LLMBudget was never constructed, kept its total in memory, and saw
only the strategy LLM, while the classifier, recommendation and the shadow's
news scorer spent on the same key unmetered.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.core.llm import spend
from halal_trader.core.llm.spend import BudgetExhausted, SpendMeter


@pytest.fixture(autouse=True)
def _no_global_meter():  # type: ignore[no-untyped-def]
    yield
    spend.install(None)


async def test_consumers_and_restarts_share_one_total(engine: AsyncEngine) -> None:
    stock = SpendMeter(engine, consumer="stock", cap_usd=10)
    shadow = SpendMeter(engine, consumer="shadow", cap_usd=10)
    await stock.record(Decimal("1.25"))
    await shadow.record(Decimal("0.75"))

    restarted = SpendMeter(engine, consumer="stock", cap_usd=10)  # a new process

    assert await restarted.spent_today() == Decimal("2.00")


async def test_observe_mode_alerts_once_and_never_blocks(engine: AsyncEngine) -> None:
    alert = AsyncMock()
    meter = SpendMeter(engine, consumer="stock", cap_usd=1, alert=alert)

    await meter.record(Decimal("0.85"))  # past 80%
    await meter.record(Decimal("0.30"))  # past the cap
    await meter.record(Decimal("0.30"))  # again: no second alert
    await meter.check()  # observe: never raises

    kinds = [c.args[0] for c in alert.await_args_list]
    assert kinds == ["llm.budget_warning", "llm.budget_exhausted"]


async def test_enforce_mode_refuses_calls_until_the_next_utc_day(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    meter = SpendMeter(engine, consumer="stock", cap_usd=1, enforce=True)
    await meter.check()  # under the cap: allowed
    await meter.record(Decimal("1.10"))

    with pytest.raises(BudgetExhausted):
        await meter.check()

    import halal_trader.core.llm.spend as spend_mod

    monkeypatch.setattr(spend_mod, "_today", lambda: date(2099, 1, 1))  # a new UTC day
    await meter.check()  # allowed again


async def test_an_enforced_cap_also_binds_a_freshly_started_process(engine: AsyncEngine) -> None:
    await SpendMeter(engine, consumer="shadow", cap_usd=1).record(Decimal("1.50"))
    fresh = SpendMeter(engine, consumer="stock", cap_usd=1, enforce=True)

    with pytest.raises(BudgetExhausted):
        await fresh.check()


async def test_glm_refuses_before_any_http_call_when_exhausted(engine: AsyncEngine) -> None:
    from types import SimpleNamespace

    from halal_trader.core.llm.glm import GLMLLM

    await SpendMeter(engine, consumer="stock", cap_usd=1).record(Decimal("2"))
    spend.install(SpendMeter(engine, consumer="stock", cap_usd=1, enforce=True))
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    create = AsyncMock()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(BudgetExhausted):
        await llm.generate("hello")

    create.assert_not_awaited()


async def test_every_glm_call_is_metered(engine: AsyncEngine) -> None:
    from types import SimpleNamespace

    from halal_trader.core.llm.glm import GLMLLM

    meter = SpendMeter(engine, consumer="stock", cap_usd=100)
    spend.install(meter)
    llm = GLMLLM(model="z-ai/glm-5.2", api_key="sk-test")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            cost=None,
        ),
    )
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )

    await llm.generate("hello")

    assert await meter.spent_today() > 0
