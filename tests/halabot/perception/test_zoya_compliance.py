"""ZoyaComplianceSource — maps screens → compliance.verdict, transient-safe."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from halabot.perception.sources.zoya_compliance import ZoyaComplianceSource
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType

CLOCK = FakeClock(datetime(2026, 5, 28, 12, 0, tzinfo=UTC))


class FakeScreener:
    def __init__(self, results: dict[str, dict]):
        self._results = results
        self.calls: list[str] = []

    async def screen_stock(self, symbol: str) -> dict:
        self.calls.append(symbol)
        r = self._results.get(symbol)
        if r is None:
            raise RuntimeError("boom")  # transport failure
        return {"symbol": symbol, **r}


async def _universe_factory(symbols):
    async def universe():
        return list(symbols)

    return universe


async def _collect(source: ZoyaComplianceSource) -> list[Event]:
    out: list[Event] = []

    async def emit(e: Event) -> None:
        out.append(e)

    await source.poll_once(emit)
    return out


@pytest.mark.asyncio
async def test_maps_each_status_to_a_verdict():
    screener = FakeScreener(
        {
            "NVDA": {"compliance": "halal", "detail": "NVIDIA — COMPLIANT"},
            "BANK": {"compliance": "not_halal", "detail": "Bank — NON_COMPLIANT"},
        }
    )
    universe = await _universe_factory(["NVDA", "BANK"])
    src = ZoyaComplianceSource(screener, universe, CLOCK, per_symbol_spacing_s=0.0)
    events = await _collect(src)
    by_asset = {e.asset: e for e in events}
    assert all(e.type == EventType.COMPLIANCE_VERDICT for e in events)
    assert by_asset["NVDA"].payload["status"] == "halal"
    assert by_asset["NVDA"].payload["transient_error"] is False
    assert by_asset["BANK"].payload["status"] == "not_halal"


@pytest.mark.asyncio
async def test_screen_failure_is_transient_no_verdict():
    """A transport failure maps to transient_error=True (a NO-VERDICT, INV-2)."""
    screener = FakeScreener({"NVDA": {"compliance": "halal"}})  # FOO missing → raises
    universe = await _universe_factory(["NVDA", "FOO"])
    src = ZoyaComplianceSource(screener, universe, CLOCK, per_symbol_spacing_s=0.0)
    events = await _collect(src)
    foo = next(e for e in events if e.asset == "FOO")
    assert foo.payload["transient_error"] is True


@pytest.mark.asyncio
async def test_error_flag_from_screener_marks_transient():
    screener = FakeScreener(
        {"NVDA": {"compliance": "doubtful", "detail": "API error 500", "error": True}}
    )
    universe = await _universe_factory(["NVDA"])
    src = ZoyaComplianceSource(screener, universe, CLOCK, per_symbol_spacing_s=0.0)
    events = await _collect(src)
    assert events[0].payload["transient_error"] is True


@pytest.mark.asyncio
async def test_no_dedup_reemits_to_refresh_freshness():
    """Verdicts are re-emitted every poll so screened_at stays fresh (INV-7)."""
    screener = FakeScreener({"NVDA": {"compliance": "halal"}})
    universe = await _universe_factory(["NVDA"])
    src = ZoyaComplianceSource(screener, universe, CLOCK, per_symbol_spacing_s=0.0)
    first = await _collect(src)
    second = await _collect(src)
    assert len(first) == 1 and len(second) == 1  # not deduped
