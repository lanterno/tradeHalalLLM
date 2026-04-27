"""Tests for the Yahoo options-IV surface."""

from __future__ import annotations

import httpx
import pytest

from halal_trader.trading.options_iv import (
    OptionsIVSnapshot,
    YahooOptionsIV,
    format_options_iv_for_prompt,
)


def _opt(strike: float, iv: float, volume: int = 0, oi: int = 0) -> dict:
    return {
        "strike": strike,
        "impliedVolatility": iv,
        "volume": volume,
        "openInterest": oi,
    }


def _payload(spot: float, calls: list[dict], puts: list[dict]) -> dict:
    return {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": spot},
                    "options": [{"calls": calls, "puts": puts}],
                }
            ]
        }
    }


def _client_with(payload: dict) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Basic fetch + reduce ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetches_atm_iv() -> None:
    spot = 100.0
    calls = [
        _opt(95, 0.30, volume=100, oi=500),
        _opt(100, 0.28, volume=200, oi=600),
        _opt(105, 0.27, volume=150, oi=400),
    ]
    puts = [
        _opt(95, 0.32, volume=120, oi=550),
        _opt(100, 0.30, volume=180, oi=650),
        _opt(105, 0.29, volume=110, oi=350),
    ]
    src = YahooOptionsIV()
    src._client = _client_with(_payload(spot, calls, puts))
    out = await src.fetch(["AAPL"])
    assert "AAPL" in out
    snap = out["AAPL"]
    assert snap.spot == 100.0
    # ATM band ±10%, mean IV across the 6 contracts
    assert 0.27 < snap.atm_iv < 0.32
    await src.aclose()


@pytest.mark.asyncio
async def test_computes_put_call_skew_when_otm_data_present() -> None:
    spot = 100.0
    calls = [
        _opt(100, 0.28),
        _opt(110, 0.25),  # OTM call (1.10x spot)
        _opt(115, 0.24),
    ]
    puts = [
        _opt(100, 0.30),
        _opt(85, 0.40),  # OTM put (0.85x spot) — high IV
        _opt(80, 0.42),
    ]
    src = YahooOptionsIV()
    src._client = _client_with(_payload(spot, calls, puts))
    out = await src.fetch(["AAPL"])
    snap = out["AAPL"]
    # Put-side IV (~0.41) > call-side (~0.245) → positive skew
    assert snap.put_call_skew > 0.05


@pytest.mark.asyncio
async def test_aggregates_volume_and_oi() -> None:
    spot = 50.0
    calls = [_opt(50, 0.20, volume=100, oi=200), _opt(55, 0.18, volume=200, oi=300)]
    puts = [_opt(50, 0.22, volume=150, oi=250), _opt(45, 0.25, volume=300, oi=400)]
    src = YahooOptionsIV()
    src._client = _client_with(_payload(spot, calls, puts))
    out = await src.fetch(["X"])
    snap = out["X"]
    assert snap.call_volume == 300
    assert snap.put_volume == 450
    assert snap.call_open_interest == 500
    assert snap.put_open_interest == 650
    assert snap.put_call_volume_ratio == pytest.approx(1.5)


# ── Labels ───────────────────────────────────────────────────────


def test_label_elevated_iv() -> None:
    s = OptionsIVSnapshot(
        symbol="X",
        spot=100,
        atm_iv=0.65,
        put_call_skew=0.0,
        call_volume=0,
        put_volume=0,
        call_open_interest=0,
        put_open_interest=0,
    )
    assert s.label == "elevated_iv"


def test_label_downside_premium() -> None:
    s = OptionsIVSnapshot(
        symbol="X",
        spot=100,
        atm_iv=0.30,
        put_call_skew=0.08,
        call_volume=100,
        put_volume=100,
        call_open_interest=0,
        put_open_interest=0,
    )
    assert s.label == "downside_premium"


def test_label_put_heavy() -> None:
    s = OptionsIVSnapshot(
        symbol="X",
        spot=100,
        atm_iv=0.30,
        put_call_skew=0.0,
        call_volume=100,
        put_volume=200,
        call_open_interest=0,
        put_open_interest=0,
    )
    assert s.label == "put_heavy_flow"


def test_label_neutral() -> None:
    s = OptionsIVSnapshot(
        symbol="X",
        spot=100,
        atm_iv=0.30,
        put_call_skew=0.01,
        call_volume=100,
        put_volume=100,
        call_open_interest=0,
        put_open_interest=0,
    )
    assert s.label == "neutral"


def test_put_call_volume_ratio_zero_calls() -> None:
    s = OptionsIVSnapshot(
        symbol="X",
        spot=100,
        atm_iv=0.30,
        put_call_skew=0.0,
        call_volume=0,
        put_volume=10,
        call_open_interest=0,
        put_open_interest=0,
    )
    assert s.put_call_volume_ratio == float("inf")


def test_put_call_volume_ratio_zero_both() -> None:
    s = OptionsIVSnapshot(
        symbol="X",
        spot=100,
        atm_iv=0.30,
        put_call_skew=0.0,
        call_volume=0,
        put_volume=0,
        call_open_interest=0,
        put_open_interest=0,
    )
    assert s.put_call_volume_ratio == 0.0


# ── Error / missing data ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_symbols_returns_empty() -> None:
    src = YahooOptionsIV()
    src._client = _client_with(_payload(100, [], []))
    assert await src.fetch([]) == {}
    await src.aclose()


@pytest.mark.asyncio
async def test_http_error_returns_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    src = YahooOptionsIV()
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = await src.fetch(["UNKNOWN"])
    assert out == {}
    await src.aclose()


@pytest.mark.asyncio
async def test_empty_chain_drops_silently() -> None:
    src = YahooOptionsIV()
    src._client = _client_with(_payload(100.0, [], []))
    out = await src.fetch(["X"])
    assert out == {}
    await src.aclose()


@pytest.mark.asyncio
async def test_zero_spot_drops_silently() -> None:
    src = YahooOptionsIV()
    src._client = _client_with(
        {
            "optionChain": {
                "result": [
                    {
                        "quote": {"regularMarketPrice": 0},
                        "options": [{"calls": [_opt(100, 0.3)], "puts": [_opt(100, 0.3)]}],
                    }
                ]
            }
        }
    )
    out = await src.fetch(["X"])
    assert out == {}
    await src.aclose()


@pytest.mark.asyncio
async def test_invalid_iv_skipped() -> None:
    spot = 100.0
    calls = [{"strike": 100, "impliedVolatility": "n/a"}, _opt(100, 0.30)]
    puts = [_opt(100, 0.32)]
    src = YahooOptionsIV()
    src._client = _client_with(_payload(spot, calls, puts))
    out = await src.fetch(["X"])
    assert "X" in out
    # Bad IV row dropped; remaining mean is between 0.30 and 0.32
    assert 0.30 <= out["X"].atm_iv <= 0.32
    await src.aclose()


# ── Caching ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_caches_per_symbol() -> None:
    calls_made = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls_made
        calls_made += 1
        return httpx.Response(
            200,
            json=_payload(100.0, [_opt(100, 0.3, volume=50)], [_opt(100, 0.3, volume=50)]),
        )

    src = YahooOptionsIV()
    src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await src.fetch(["AAPL"])
    await src.fetch(["AAPL"])
    await src.fetch(["AAPL"])
    assert calls_made == 1
    await src.aclose()


# ── Prompt formatting ────────────────────────────────────────────


def test_format_empty_returns_empty() -> None:
    assert format_options_iv_for_prompt({}) == ""


def test_format_renders_top_rows_sorted_by_iv() -> None:
    snaps = {
        "A": OptionsIVSnapshot(
            symbol="A",
            spot=100,
            atm_iv=0.20,
            put_call_skew=0.0,
            call_volume=100,
            put_volume=100,
            call_open_interest=0,
            put_open_interest=0,
        ),
        "B": OptionsIVSnapshot(
            symbol="B",
            spot=100,
            atm_iv=0.50,  # higher → first
            put_call_skew=0.05,
            call_volume=100,
            put_volume=200,
            call_open_interest=0,
            put_open_interest=0,
        ),
    }
    text = format_options_iv_for_prompt(snaps)
    assert text.index("B ") < text.index("A ")
