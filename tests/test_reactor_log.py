"""The reactor's decision record: one row per acted-on catalyst, never raising."""

from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.trading import reactor_log


def _event() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="NVDA",
        title="Nvidia wins a hyperscaler deal",
        classification=SimpleNamespace(score=0.93, tag="contract", rationale="big"),
    )


async def test_a_shadow_entry_and_an_observation_are_recorded(engine: AsyncEngine) -> None:
    shadow = {
        "status": "shadow",
        "quantity": 50,
        "price": 200.0,
        "intraday_change": 0.005,
        "stop_loss": 184.0,
    }
    await reactor_log.record(engine, _event(), shadow, "👻 Shadow: would BUY 50 NVDA")
    await reactor_log.record(engine, _event(), None, "Observation only — market closed")
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT status, placed, quantity, price, reason FROM reactor_decisions "
                    "ORDER BY id"
                )
            )
        ).all()
    assert [(r.status, r.placed, r.quantity, r.price) for r in rows] == [
        ("shadow", False, 50.0, 200.0),
        ("observed", False, None, None),
    ]
    assert rows[1].reason == "Observation only — market closed"


async def test_a_failed_write_does_not_raise() -> None:
    class Broken:
        def begin(self):  # type: ignore[no-untyped-def]
            raise ConnectionError("db down")

    await reactor_log.record(Broken(), _event(), {"status": "shadow"}, "")  # type: ignore[arg-type]
