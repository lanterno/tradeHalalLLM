"""Tests for quant/expected_move.py — ATM-straddle expected move."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.quant.expected_move import (
    expected_move_from_chain,
    parse_occ_symbol,
)

TODAY = date(2026, 7, 6)


def _snap(bid: float, ask: float) -> dict:
    return {"latestQuote": {"bp": bid, "ap": ask}}


def _chain(spot: float, expiry_yymmdd: str, strikes_to_mid: dict[float, float]) -> dict:
    """Build a chain snapshot: at each strike a call+put whose mid == given.

    Mids are centered so the straddle at the ATM strike is exact.
    """
    out = {}
    for strike, mid in strikes_to_mid.items():
        k = f"{int(strike * 1000):08d}"
        # call mid and put mid each == mid/2 so straddle == mid at the strike
        half = mid / 2
        out[f"AAPL{expiry_yymmdd}C{k}"] = _snap(half - 0.05, half + 0.05)
        out[f"AAPL{expiry_yymmdd}P{k}"] = _snap(half - 0.05, half + 0.05)
    return out


class TestParseOcc:
    def test_parses_call_and_put(self):
        c = parse_occ_symbol("AAPL260713C00215000")
        assert c is not None
        assert c.expiry == "2026-07-13" and c.is_call and c.strike == 215.0
        p = parse_occ_symbol("AAPL260713P00220500")
        assert p is not None and not p.is_call and p.strike == 220.5

    def test_short_underlying(self):
        q = parse_occ_symbol("F260918C00012000")
        assert q is not None and q.strike == 12.0 and q.expiry == "2026-09-18"

    def test_rejects_garbage(self):
        assert parse_occ_symbol("NOTASYMBOL") is None
        assert parse_occ_symbol("AAPL260713X00215000") is None  # bad type
        assert parse_occ_symbol("AAPL261399C00215000") is None  # bad month/day


class TestExpectedMove:
    def test_picks_atm_strike_and_computes_move(self):
        spot = 100.0
        # ATM straddle (strike 100) mid = 6.0 → move = 0.85*6 = 5.1
        chain = _chain(spot, "260710", {95.0: 8.0, 100.0: 6.0, 105.0: 8.5})
        em = expected_move_from_chain(chain, spot, today=TODAY)
        assert em is not None
        assert em.atm_strike == 100.0
        assert em.straddle == pytest.approx(6.0, abs=0.02)
        assert em.move_abs == pytest.approx(5.1, abs=0.02)
        assert em.move_pct == pytest.approx(5.1, abs=0.05)
        assert em.low == pytest.approx(94.9, abs=0.05)
        assert em.high == pytest.approx(105.1, abs=0.05)
        assert em.dte == 4  # 07-06 → 07-10

    def test_nearest_future_expiry_wins_and_skips_short_dte(self):
        spot = 100.0
        # An expiry 1 day out (below min_dte) and one 8 days out.
        near = _chain(spot, "260707", {100.0: 3.0})  # dte 1 → skipped
        far = _chain(spot, "260714", {100.0: 6.0})  # dte 8 → chosen
        em = expected_move_from_chain({**near, **far}, spot, today=TODAY)
        assert em is not None
        assert em.expiry == "2026-07-14"
        assert em.straddle == pytest.approx(6.0, abs=0.02)

    def test_illiquid_atm_falls_through_to_next_strike(self):
        spot = 100.0
        chain = _chain(spot, "260710", {100.0: 6.0, 101.0: 6.4})
        # Wreck the 100-strike call with a huge spread → ATM moves to 101.
        chain["AAPL260710C00100000"] = _snap(1.0, 5.0)  # spread frac ~1.3
        em = expected_move_from_chain(chain, spot, today=TODAY)
        assert em is not None
        assert em.atm_strike == 101.0

    def test_crossed_and_zero_quotes_rejected(self):
        spot = 100.0
        chain = {
            "AAPL260710C00100000": _snap(5.0, 3.0),  # crossed
            "AAPL260710P00100000": _snap(0.0, 3.0),  # zero bid
        }
        assert expected_move_from_chain(chain, spot, today=TODAY) is None

    def test_no_common_strike_returns_none(self):
        spot = 100.0
        chain = {
            "AAPL260710C00100000": _snap(2.9, 3.1),  # only a call
            "AAPL260710C00105000": _snap(1.9, 2.1),
        }
        assert expected_move_from_chain(chain, spot, today=TODAY) is None

    def test_invalid_spot(self):
        assert (
            expected_move_from_chain(_chain(100.0, "260710", {100.0: 6.0}), 0.0, today=TODAY)
            is None
        )

    def test_realistic_alpaca_shape(self):
        # Mirror the observed snapshot shape (extra keys must be ignored).
        spot = 213.0
        chain = {
            "AAPL260714C00213000": {
                "dailyBar": {"c": 5.0},
                "latestQuote": {"ap": 5.6, "as": 20, "bp": 5.2, "bs": 7},
                "latestTrade": {"p": 5.4},
            },
            "AAPL260714P00213000": {
                "latestQuote": {"ap": 5.2, "bp": 4.8},
            },
        }
        em = expected_move_from_chain(chain, spot, today=TODAY)
        assert em is not None
        assert em.atm_strike == 213.0
        # straddle = 5.4 + 5.0 = 10.4 → move 8.84
        assert em.straddle == pytest.approx(10.4, abs=0.02)
        assert em.move_abs == pytest.approx(8.84, abs=0.02)
