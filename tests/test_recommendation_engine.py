"""Tests for the daily halal recommendation engine (advisory)."""

from __future__ import annotations

from typing import Any

import pytest

from halal_trader.recommendation.engine import DailyRecommendationEngine


def _bars(base: float, slope: float, n: int = 60) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": f"2026-04-{(i % 28) + 1:02d}",
            "open": base + slope * i,
            "high": base + slope * i + 2,
            "low": base + slope * i - 2,
            "close": base + slope * i + 1,
            "volume": 1000 + i,
        }
        for i in range(n)
    ]


class _FakeBroker:
    def __init__(self, *, bars_by_symbol: dict[str, list] | None = None, default_n: int = 60):
        self._bars = bars_by_symbol or {}
        self._default_n = default_n

    async def get_stock_bars(self, symbol: str, days: int = 60, timeframe: str = "1Day"):
        if symbol in self._bars:
            return self._bars[symbol]
        return _bars(100.0, 0.5 if symbol == "NVDA" else 0.0, self._default_n)


class _FakeRepo:
    def __init__(self) -> None:
        self.saved: dict[str, Any] | None = None

    async def save_recommendation(self, rec: dict[str, Any]) -> int:
        self.saved = rec
        return 42


class _FakeLLM:
    model = "fake-llm"

    def __init__(self, response: dict[str, Any]):
        self._response = response
        self.last_prompt: str | None = None

    async def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        self.last_prompt = prompt
        return self._response


def _engine(llm: _FakeLLM, broker: _FakeBroker | None = None, universe=None):
    from unittest.mock import MagicMock

    settings = MagicMock()
    return DailyRecommendationEngine(
        broker=broker or _FakeBroker(),
        repo=_FakeRepo(),
        settings=settings,
        llm=llm,
        universe=universe or ["AAPL", "NVDA", "MSFT"],
    )


@pytest.mark.asyncio
async def test_generate_picks_and_persists():
    llm = _FakeLLM(
        {
            "symbol": "NVDA",
            "conviction": 0.82,
            "thesis": "Strong uptrend above EMAs.",
            "halal_note": "Semiconductors — real productive asset, AAOIFI compliant.",
            "suggested_entry": 130.0,
            "suggested_target": 145.0,
            "suggested_stop": 124.0,
            "catalysts": "AI demand",
            "risks": "valuation",
        }
    )
    eng = _engine(llm)
    rec = await eng.generate()

    assert rec["symbol"] == "NVDA"
    assert rec["conviction"] == 0.82
    assert rec["id"] == 42
    assert rec["universe_size"] == 3  # all 3 had enough bars
    assert rec["prompt_version"].startswith("recommendation.daily.system@")
    assert rec["candidates"]["NVDA"]["price"] is not None
    # the universe is shown to the model
    assert "NVDA" in llm.last_prompt and "AAPL" in llm.last_prompt
    # cross-sectional factor core ran: scores merged + leaders in the prompt
    assert "factor_score" in rec["candidates"]["NVDA"]
    assert "factor leaders" in llm.last_prompt.lower()


@pytest.mark.asyncio
async def test_quant_range_bands_reach_prompt_and_candidates(monkeypatch):
    import halal_trader.quant.calibration as qcal

    monkeypatch.setattr(qcal, "load_default_artifact", lambda: None)
    llm = _FakeLLM(
        {
            "symbol": "NVDA",
            "conviction": 0.6,
            "thesis": "t",
            "halal_note": "h",
            "suggested_entry": 130.0,
            "suggested_target": 140.0,
            "suggested_stop": 125.0,
        }
    )
    eng = _engine(llm)
    rec = await eng.generate()
    nvda = rec["candidates"]["NVDA"]
    # 60 fake bars → HAR refuses → current Yang-Zhang estimate still bands.
    assert nvda["band5d_lo"] < nvda["price"] < nvda["band5d_hi"]
    assert nvda["rng1d_pct"] > 0
    qb = nvda["quant_bands"]
    assert qb["calibrated"] is False
    assert qb["calibration_version"] is None
    assert qb["5"]["source"] == "yz_current"
    assert qb["5"]["low"] < qb["5"]["high"]
    # The prompt shows the band and explains its (uncalibrated) semantics.
    assert "band5d=" in llm.last_prompt
    assert "UNCALIBRATED" in llm.last_prompt


@pytest.mark.asyncio
async def test_market_regime_block_leads_the_prompt(monkeypatch):
    import halal_trader.quant.regime as qregime
    from halal_trader.quant.regime import RegimeReading

    async def _fake_regime():
        return RegimeReading(
            regime="risk_off", r_slow=1.05, r_fast=1.1, fast_inverted=True, vix=28.0
        )

    monkeypatch.setattr(qregime, "fetch_vix_term_structure", _fake_regime)
    llm = _FakeLLM(
        {
            "symbol": "NVDA",
            "conviction": 0.6,
            "thesis": "t",
            "halal_note": "h",
            "suggested_entry": 130.0,
            "suggested_target": 140.0,
            "suggested_stop": 125.0,
        }
    )
    eng = _engine(llm)
    await eng.generate()
    # The regime block leads the prompt (before the candidate table).
    assert "MARKET REGIME: RISK-OFF" in llm.last_prompt
    assert llm.last_prompt.index("MARKET REGIME") < llm.last_prompt.index("Candidate universe")


@pytest.mark.asyncio
async def test_regime_outage_does_not_break_generation(monkeypatch):
    import halal_trader.quant.regime as qregime

    async def _no_regime():
        return None

    monkeypatch.setattr(qregime, "fetch_vix_term_structure", _no_regime)
    llm = _FakeLLM(
        {
            "symbol": "NVDA",
            "conviction": 0.6,
            "thesis": "t",
            "halal_note": "h",
            "suggested_entry": 130.0,
            "suggested_target": 140.0,
            "suggested_stop": 125.0,
        }
    )
    eng = _engine(llm)
    rec = await eng.generate()
    assert rec["symbol"] == "NVDA"
    assert "MARKET REGIME" not in llm.last_prompt


@pytest.mark.asyncio
async def test_calibration_artifact_flips_prompt_semantics(monkeypatch):
    import halal_trader.quant.calibration as qcal
    from halal_trader.quant.calibration import CalibrationArtifact, HorizonCalibration

    artifact = CalibrationArtifact(
        version="zcal-test",
        created_at="2026-07-13T00:00:00+00:00",
        target_coverage=0.8,
        horizons={
            1: HorizonCalibration(z=1.9, n=100, target_coverage=0.8),
            5: HorizonCalibration(z=2.1, n=100, target_coverage=0.8),
        },
        symbols=("AAPL",),
    )
    monkeypatch.setattr(qcal, "load_default_artifact", lambda: artifact)
    llm = _FakeLLM(
        {
            "symbol": "NVDA",
            "conviction": 0.6,
            "thesis": "t",
            "halal_note": "h",
            "suggested_entry": 130.0,
            "suggested_target": 140.0,
            "suggested_stop": 125.0,
        }
    )
    eng = _engine(llm)
    rec = await eng.generate()
    qb = rec["candidates"]["NVDA"]["quant_bands"]
    assert qb["calibrated"] is True
    assert qb["calibration_version"] == "zcal-test"
    assert qb["5"]["z"] == pytest.approx(2.1)
    assert "coverage-calibrated" in llm.last_prompt
    assert "zcal-test" in llm.last_prompt
    assert "UNCALIBRATED" not in llm.last_prompt


@pytest.mark.asyncio
async def test_rejects_symbol_outside_universe():
    llm = _FakeLLM({"symbol": "TSLA", "conviction": 0.9, "thesis": "x", "halal_note": "y"})
    eng = _engine(llm)
    with pytest.raises(ValueError, match="not in the candidate universe"):
        await eng.generate()


@pytest.mark.asyncio
async def test_validate_clamps_bad_levels_and_conviction():
    # stop ABOVE entry + target BELOW entry + conviction > 1 → all repaired.
    llm = _FakeLLM(
        {
            "symbol": "AAPL",
            "conviction": 1.7,
            "thesis": "t",
            "halal_note": "h",
            "suggested_entry": 100.0,
            "suggested_target": 90.0,  # invalid (<= entry)
            "suggested_stop": 105.0,  # invalid (>= entry)
        }
    )
    eng = _engine(llm)
    rec = await eng.generate()
    assert rec["conviction"] == 1.0  # clamped
    assert rec["suggested_stop"] < rec["suggested_entry"]  # repaired below
    assert rec["suggested_target"] > rec["suggested_entry"]  # repaired above


@pytest.mark.asyncio
async def test_skips_symbols_with_insufficient_bars():
    # MSFT has too few bars → excluded from candidates; pick must be a kept name.
    broker = _FakeBroker(bars_by_symbol={"MSFT": _bars(50.0, 0.1, n=5)})
    llm = _FakeLLM(
        {
            "symbol": "NVDA",
            "conviction": 0.6,
            "thesis": "t",
            "halal_note": "h",
            "suggested_entry": 130.0,
            "suggested_target": 140.0,
            "suggested_stop": 125.0,
        }
    )
    eng = _engine(llm, broker=broker)
    rec = await eng.generate()
    assert "MSFT" not in rec["candidates"]
    assert rec["universe_size"] == 2  # AAPL + NVDA


@pytest.mark.asyncio
async def test_raises_when_no_candidates():
    broker = _FakeBroker(default_n=3)  # every symbol short → no candidates
    llm = _FakeLLM({"symbol": "NVDA", "conviction": 0.6, "thesis": "t", "halal_note": "h"})
    eng = _engine(llm, broker=broker)
    with pytest.raises(RuntimeError, match="no candidate market data"):
        await eng.generate()
