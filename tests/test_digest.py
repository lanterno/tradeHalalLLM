"""The weekly digest builds from the database alone and reads as plain lines."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.notifications.digest import build


async def test_the_digest_covers_each_section_even_with_no_history(engine: AsyncEngine) -> None:
    settings = SimpleNamespace(
        zakat=SimpleNamespace(hawl_hijri="09-01"),
        llm=SimpleNamespace(monthly_live_usd=25.0, monthly_research_usd=15.0),
        core=SimpleNamespace(paper=True),
    )
    message = await build(engine, settings, today=date(2026, 10, 9))
    assert message.startswith("<b>Halal Trader")
    for section in ("Core account", "Halal:", "Next zakat", "LLM this month"):
        assert section in message
    assert "&#x27;" not in message
