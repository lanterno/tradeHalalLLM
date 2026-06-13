"""Tests for PerformanceAnalytics — metric computation from round-trip data."""

import pytest

from halal_trader.crypto.analytics import PerformanceAnalytics, PerformanceStats


def _make_rt(
    pair,
    pnl,
    pnl_pct,
    duration_minutes=10,
    exit_reason="llm_sell",
    closed_at="2025-01-01T12:00:00",
):
    return {
        "pair": pair,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "duration_minutes": duration_minutes,
        "exit_reason": exit_reason,
        "closed_at": closed_at,
    }


class FakeRepo:
    def __init__(self, round_trips):
        self._round_trips = round_trips

    async def get_completed_round_trips(self, limit=500, lookback_days=7):
        return self._round_trips


class TestPerformanceStats:
    async def test_empty_round_trips(self):
        analytics = PerformanceAnalytics(FakeRepo([]))
        stats = await analytics.compute_stats()
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0

    async def test_all_wins(self):
        trips = [
            _make_rt("BTCUSDT", 10.0, 0.01, closed_at="2025-01-01T12:00:00"),
            _make_rt("ETHUSDT", 5.0, 0.005, closed_at="2025-01-01T12:05:00"),
            _make_rt("BTCUSDT", 8.0, 0.008, closed_at="2025-01-01T12:10:00"),
        ]
        analytics = PerformanceAnalytics(FakeRepo(trips))
        stats = await analytics.compute_stats()

        assert stats.total_trades == 3
        assert stats.wins == 3
        assert stats.losses == 0
        assert stats.win_rate == 1.0
        assert stats.profit_factor == float("inf")
        assert stats.total_pnl == pytest.approx(23.0)

    async def test_mixed_results(self):
        trips = [
            _make_rt("BTCUSDT", 10.0, 0.02, closed_at="2025-01-01T12:00:00"),
            _make_rt("ETHUSDT", -3.0, -0.01, closed_at="2025-01-01T12:05:00"),
            _make_rt("BTCUSDT", -7.0, -0.015, closed_at="2025-01-01T12:10:00"),
            _make_rt("SOLUSDT", 5.0, 0.01, closed_at="2025-01-01T12:15:00"),
        ]
        analytics = PerformanceAnalytics(FakeRepo(trips))
        stats = await analytics.compute_stats()

        assert stats.total_trades == 4
        assert stats.wins == 2
        assert stats.losses == 2
        assert stats.win_rate == pytest.approx(0.5)
        assert stats.avg_win_pct == pytest.approx(0.015)
        assert stats.avg_loss_pct == pytest.approx(-0.0125)
        assert stats.profit_factor == pytest.approx(15.0 / 10.0)
        assert stats.total_pnl == pytest.approx(5.0)

    async def test_best_worst_pair(self):
        trips = [
            _make_rt("BTCUSDT", 20.0, 0.02, closed_at="2025-01-01T12:00:00"),
            _make_rt("ETHUSDT", -5.0, -0.01, closed_at="2025-01-01T12:05:00"),
            _make_rt("BTCUSDT", -2.0, -0.005, closed_at="2025-01-01T12:10:00"),
            _make_rt("SOLUSDT", -10.0, -0.03, closed_at="2025-01-01T12:15:00"),
        ]
        analytics = PerformanceAnalytics(FakeRepo(trips))
        stats = await analytics.compute_stats()

        assert stats.best_pair == "BTCUSDT"
        assert stats.best_pair_pnl == pytest.approx(18.0)
        assert stats.worst_pair == "SOLUSDT"
        assert stats.worst_pair_pnl == pytest.approx(-10.0)

    async def test_exit_reasons(self):
        t1 = "2025-01-01T12:00:00"
        t2 = "2025-01-01T12:05:00"
        t3 = "2025-01-01T12:10:00"
        t4 = "2025-01-01T12:15:00"
        trips = [
            _make_rt("BTCUSDT", 10, 0.01, exit_reason="take_profit", closed_at=t1),
            _make_rt("ETHUSDT", -3, -0.01, exit_reason="stop_loss", closed_at=t2),
            _make_rt("BTCUSDT", 5, 0.005, exit_reason="take_profit", closed_at=t3),
            _make_rt("SOLUSDT", -2, -0.005, exit_reason="llm_sell", closed_at=t4),
        ]
        analytics = PerformanceAnalytics(FakeRepo(trips))
        stats = await analytics.compute_stats()

        assert stats.by_exit_reason["take_profit"] == 2
        assert stats.by_exit_reason["stop_loss"] == 1
        assert stats.by_exit_reason["llm_sell"] == 1

    async def test_avg_hold_time(self):
        trips = [
            _make_rt("BTCUSDT", 10, 0.01, duration_minutes=20, closed_at="2025-01-01T12:00:00"),
            _make_rt("ETHUSDT", 5, 0.005, duration_minutes=40, closed_at="2025-01-01T12:05:00"),
        ]
        analytics = PerformanceAnalytics(FakeRepo(trips))
        stats = await analytics.compute_stats()
        assert stats.avg_hold_minutes == pytest.approx(30.0)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        trips = [
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:00:00"),
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:05:00"),
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:10:00"),
        ]
        dd = PerformanceAnalytics._compute_max_drawdown(trips)
        assert dd == 0.0

    def test_drawdown_after_peak(self):
        """Two +1% wins then -1.5%/-0.5% losses → the worst peak-to-trough on
        the compounded equity curve is 1 - (0.985 * 0.995) ≈ 1.99%.

        The old dollar-cumulative version normalized by the PEAK OF CUMULATIVE
        P&L (not equity) and asserted 20/20 = 100% drawdown for trades that
        individually moved at most 1.5% — the same defect that printed a fake
        'Max drawdown: 242.52%' into the live stock LLM prompt."""
        trips = [
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:00:00"),
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:05:00"),
            _make_rt("A", -15, -0.015, closed_at="2025-01-01T12:10:00"),
            _make_rt("A", -5, -0.005, closed_at="2025-01-01T12:15:00"),
        ]
        dd = PerformanceAnalytics._compute_max_drawdown(trips)
        assert dd == pytest.approx(1 - 0.985 * 0.995)

    def test_drawdown_bounded_below_one_even_when_cumulative_goes_negative(self):
        """Regression for the 242% bug: a losing streak that drives cumulative
        P&L below zero must still yield a drawdown fraction < 1, not a
        peak-relative blowup."""
        trips = [
            _make_rt("A", 50, 0.005, closed_at="2025-01-01T12:00:00"),  # peak +$50
            _make_rt("A", -60, -0.006, closed_at="2025-01-01T12:05:00"),
            _make_rt("A", -61, -0.006, closed_at="2025-01-01T12:10:00"),  # trough -$71
        ]
        dd = PerformanceAnalytics._compute_max_drawdown(trips)
        # Old math: (50 + 71) / 50 = 2.42 → printed "242%". New math: ~1.2%.
        assert 0.0 < dd < 0.05

    def test_empty_trips(self):
        dd = PerformanceAnalytics._compute_max_drawdown([])
        assert dd == 0.0


class TestStreak:
    def test_win_streak(self):
        trips = [
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:00:00"),
            _make_rt("A", 5, 0.005, closed_at="2025-01-01T12:05:00"),
            _make_rt("A", 8, 0.008, closed_at="2025-01-01T12:10:00"),
        ]
        count, stype = PerformanceAnalytics._compute_streak(trips)
        assert count == 3
        assert stype == "wins"

    def test_loss_streak(self):
        trips = [
            _make_rt("A", 10, 0.01, closed_at="2025-01-01T12:00:00"),
            _make_rt("A", -5, -0.005, closed_at="2025-01-01T12:05:00"),
            _make_rt("A", -3, -0.003, closed_at="2025-01-01T12:10:00"),
        ]
        count, stype = PerformanceAnalytics._compute_streak(trips)
        assert count == 2
        assert stype == "losses"

    def test_empty(self):
        count, stype = PerformanceAnalytics._compute_streak([])
        assert count == 0
        assert stype == ""


class TestFormatForPrompt:
    async def test_no_trades_message(self):
        analytics = PerformanceAnalytics(FakeRepo([]))
        stats = PerformanceStats()
        text = analytics.format_for_prompt(stats)
        assert "No completed trades" in text

    async def test_includes_key_fields(self):
        trips = [
            _make_rt(
                "BTCUSDT", 10, 0.01, exit_reason="take_profit", closed_at="2025-01-01T12:00:00"
            ),
            _make_rt(
                "ETHUSDT", -3, -0.01, exit_reason="stop_loss", closed_at="2025-01-01T12:05:00"
            ),
        ]
        analytics = PerformanceAnalytics(FakeRepo(trips))
        stats = await analytics.compute_stats()
        text = analytics.format_for_prompt(stats)

        assert "Win rate" in text
        assert "Profit factor" in text
        assert "BTCUSDT" in text
        assert "take_profit" in text
