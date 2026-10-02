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


async def test_monthly_pools_keep_research_and_live_apart(engine: AsyncEngine) -> None:
    alert = AsyncMock()
    stock = SpendMeter(engine, consumer="stock", cap_usd=0, monthly_cap_usd=25, enforce=True)
    shadow = SpendMeter(engine, consumer="shadow", cap_usd=0, monthly_cap_usd=25, enforce=True)
    research = SpendMeter(
        engine, consumer="research", cap_usd=0, monthly_cap_usd=15, enforce=True, alert=alert
    )
    await stock.record(Decimal("20"))
    await shadow.record(Decimal("5.5"))  # the live pool is spent between the two of them

    with pytest.raises(BudgetExhausted, match="live"):
        await stock.check()
    with pytest.raises(BudgetExhausted, match="live"):
        await SpendMeter(
            engine, consumer="shadow", cap_usd=0, monthly_cap_usd=25, enforce=True
        ).check()
    await research.check()  # research has its own pool: still allowed

    await research.record(Decimal("12.5"))  # past 80% of 15
    assert [c.args[0] for c in alert.await_args_list] == ["llm.budget_warning.research"]
    await research.check()


def test_each_consumer_gets_its_pools_budget() -> None:
    caps = {"live_usd": 25.0, "research_usd": 15.0}
    assert spend.monthly_cap_for("stock", **caps) == 25.0
    assert spend.monthly_cap_for("shadow", **caps) == 25.0
    assert spend.monthly_cap_for("research", **caps) == 15.0
    assert spend.monthly_cap_for("other", **caps) == 0.0


async def test_a_metered_block_charges_its_own_pool_and_leaves_the_process_meter(
    engine: AsyncEngine,
) -> None:
    import asyncio

    stock = SpendMeter(engine, consumer="stock", cap_usd=0)
    research = SpendMeter(engine, consumer="research", cap_usd=0)
    spend.install(stock)

    async def call() -> None:
        await spend.after_call(Decimal("0.10"))

    with spend.metered(research):
        await asyncio.gather(call(), call())  # tasks started inside inherit it
    await call()  # outside: the process meter again

    async with engine.connect() as conn:
        from sqlalchemy import text

        rows = dict((await conn.execute(text("SELECT consumer, spent_usd FROM llm_spend"))).all())
    assert rows == {"research": Decimal("0.200000"), "stock": Decimal("0.100000")}
