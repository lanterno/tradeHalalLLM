"""Tests for web/public_leaderboard.py — Round-5 Wave 17.A."""

from __future__ import annotations

from datetime import date

import pytest

from halal_trader.web.public_leaderboard import (
    LeaderboardCell,
    LeaderboardPolicy,
    OperatorEntry,
    RiskBucket,
    TimeWindow,
    build_cell,
    build_grid,
    render_cell,
    render_grid,
)


def _entry(
    handle: str = "operator-001",
    window: TimeWindow = TimeWindow.MONTHLY,
    bucket: RiskBucket = RiskBucket.MODERATE,
    return_pct: float = 0.05,
    sharpe: float = 1.0,
    drawdown: float = 0.10,
    n_trades: int = 50,
    consents: bool = True,
    last_updated: date = date(2026, 5, 1),
) -> OperatorEntry:
    return OperatorEntry(
        handle=handle,
        window=window,
        bucket=bucket,
        return_pct=return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown,
        n_trades=n_trades,
        consents_to_publish=consents,
        last_updated=last_updated,
    )


# --- Enum + validation ----------------------------------------------------


def test_time_window_string_values():
    assert TimeWindow.WEEKLY.value == "weekly"
    assert TimeWindow.MONTHLY.value == "monthly"
    assert TimeWindow.QUARTERLY.value == "quarterly"
    assert TimeWindow.YEARLY.value == "yearly"
    assert TimeWindow.ALL_TIME.value == "all_time"


def test_risk_bucket_string_values():
    assert RiskBucket.CONSERVATIVE.value == "conservative"
    assert RiskBucket.MODERATE.value == "moderate"
    assert RiskBucket.AGGRESSIVE.value == "aggressive"


def test_default_policy():
    p = LeaderboardPolicy()
    assert p.min_cell_size == 5
    assert p.top_n_per_cell == 10
    assert p.confidence_required is True


def test_policy_zero_min_size_rejected():
    with pytest.raises(ValueError):
        LeaderboardPolicy(min_cell_size=0)


def test_policy_zero_top_n_rejected():
    with pytest.raises(ValueError):
        LeaderboardPolicy(top_n_per_cell=0)


def test_entry_empty_handle_rejected():
    with pytest.raises(ValueError):
        _entry(handle="")


def test_entry_email_handle_rejected():
    """Real names / emails are forbidden; only anonymous handles allowed."""
    with pytest.raises(ValueError):
        _entry(handle="user@example.com")


def test_entry_negative_trades_rejected():
    with pytest.raises(ValueError):
        _entry(n_trades=-1)


def test_entry_drawdown_over_one_rejected():
    with pytest.raises(ValueError):
        _entry(drawdown=1.5)


def test_entry_negative_drawdown_rejected():
    with pytest.raises(ValueError):
        _entry(drawdown=-0.1)


def test_cell_invariant_empty_when_too_small():
    """If cell_too_small=True, entries must be empty."""
    with pytest.raises(ValueError):
        LeaderboardCell(
            window=TimeWindow.WEEKLY,
            bucket=RiskBucket.MODERATE,
            entries=(_entry(),),
            cell_too_small=True,
        )


# --- k-anonymity guard ----------------------------------------------------


def test_below_threshold_returns_empty():
    """Fewer than min_cell_size entries → cell too small, no exposure."""
    entries = [_entry(handle=f"op-{i}") for i in range(3)]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    assert cell.cell_too_small is True
    assert cell.entries == ()


def test_at_threshold_publishes():
    entries = [_entry(handle=f"op-{i}") for i in range(5)]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    assert cell.cell_too_small is False
    assert len(cell.entries) == 5


def test_custom_threshold():
    entries = [_entry(handle=f"op-{i}") for i in range(3)]
    cell = build_cell(
        entries,
        window=TimeWindow.MONTHLY,
        bucket=RiskBucket.MODERATE,
        policy=LeaderboardPolicy(min_cell_size=2),
    )
    assert cell.cell_too_small is False


# --- Filtering ------------------------------------------------------------


def test_filters_by_window():
    entries = [_entry(handle=f"op-{i}", window=TimeWindow.MONTHLY) for i in range(5)]
    cell = build_cell(entries, window=TimeWindow.WEEKLY, bucket=RiskBucket.MODERATE)
    assert cell.cell_too_small is True


def test_filters_by_bucket():
    entries = [_entry(handle=f"op-{i}", bucket=RiskBucket.MODERATE) for i in range(5)]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.CONSERVATIVE)
    assert cell.cell_too_small is True


def test_filters_by_consent_default():
    entries = [_entry(handle=f"op-{i}", consents=False) for i in range(5)]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    assert cell.cell_too_small is True


def test_consent_optional_when_policy_relaxed():
    entries = [_entry(handle=f"op-{i}", consents=False) for i in range(5)]
    cell = build_cell(
        entries,
        window=TimeWindow.MONTHLY,
        bucket=RiskBucket.MODERATE,
        policy=LeaderboardPolicy(confidence_required=False),
    )
    assert cell.cell_too_small is False


# --- Ranking ------------------------------------------------------------


def test_ranks_by_return_descending():
    entries = [
        _entry(handle="A", return_pct=0.10),
        _entry(handle="B", return_pct=0.20),
        _entry(handle="C", return_pct=0.05),
        _entry(handle="D", return_pct=0.15),
        _entry(handle="E", return_pct=0.07),
    ]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    handles = [e.handle for e in cell.entries]
    assert handles == ["B", "D", "A", "E", "C"]


def test_tie_breaks_by_sharpe():
    entries = [
        _entry(handle="A", return_pct=0.10, sharpe=1.0),
        _entry(handle="B", return_pct=0.10, sharpe=2.0),  # higher sharpe
        _entry(handle="C", return_pct=0.05),
        _entry(handle="D", return_pct=0.05),
        _entry(handle="E", return_pct=0.05),
    ]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    assert cell.entries[0].handle == "B"  # higher sharpe wins tie


def test_top_n_truncates():
    entries = [_entry(handle=f"op-{i:02d}", return_pct=0.10 - i * 0.001) for i in range(20)]
    cell = build_cell(
        entries,
        window=TimeWindow.MONTHLY,
        bucket=RiskBucket.MODERATE,
        policy=LeaderboardPolicy(top_n_per_cell=3, min_cell_size=5),
    )
    assert len(cell.entries) == 3


# --- Grid -----------------------------------------------------------------


def test_build_grid_covers_all_combinations():
    entries = [_entry(handle=f"op-{i}") for i in range(5)]
    grid = build_grid(entries)
    # 5 windows × 3 buckets = 15 cells
    assert len(grid) == 15


def test_build_grid_unfilled_cells_too_small():
    entries = [_entry(handle=f"op-{i}") for i in range(5)]  # only one (window, bucket)
    grid = build_grid(entries)
    too_small = [c for c in grid if c.cell_too_small]
    assert len(too_small) == 14  # only the populated cell is full


# --- Render --------------------------------------------------------------


def test_render_cell_too_small():
    cell = LeaderboardCell(
        window=TimeWindow.WEEKLY,
        bucket=RiskBucket.MODERATE,
        entries=(),
        cell_too_small=True,
    )
    out = render_cell(cell)
    assert "insufficient" in out


def test_render_cell_with_entries():
    entries = [_entry(handle=f"op-{i}", return_pct=0.10 - i * 0.01) for i in range(5)]
    cell = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    out = render_cell(cell)
    assert "Leaderboard" in out
    assert "op-0" in out
    # Rank 1 is op-0 (highest return)
    assert out.index("op-0") < out.index("op-4")


def test_render_grid_includes_multiple_cells():
    entries = [_entry(handle=f"op-{i}") for i in range(5)]
    grid = build_grid(entries)
    out = render_grid(grid)
    assert "monthly" in out
    assert "moderate" in out


def test_render_no_secret_leak():
    entries = [_entry(handle=f"op-{i}") for i in range(5)]
    grid = build_grid(entries)
    out = render_grid(grid)
    for token in (
        "@",
        "zoom.us",
        "meet.google",
        "private_email",
        "+1-",
        "Authorization",
        "real_name",
        "address",
    ):
        assert token not in out


# --- E2E ---------------------------------------------------------------


def test_e2e_realistic_leaderboard_with_anonymity_protection():
    """A realistic mix where some buckets meet threshold, others don't."""
    entries: list[OperatorEntry] = []
    # 10 moderate-monthly operators
    for i in range(10):
        entries.append(_entry(handle=f"mod-{i}", return_pct=0.05 + i * 0.002))
    # Only 2 aggressive-yearly operators (below threshold)
    for i in range(2):
        entries.append(
            _entry(
                handle=f"agg-{i}",
                window=TimeWindow.YEARLY,
                bucket=RiskBucket.AGGRESSIVE,
                return_pct=0.30,
            )
        )
    grid = build_grid(entries)
    moderate_monthly = next(
        c for c in grid if c.window is TimeWindow.MONTHLY and c.bucket is RiskBucket.MODERATE
    )
    aggressive_yearly = next(
        c for c in grid if c.window is TimeWindow.YEARLY and c.bucket is RiskBucket.AGGRESSIVE
    )
    assert moderate_monthly.cell_too_small is False
    assert aggressive_yearly.cell_too_small is True


def test_replay_consistency():
    entries = [_entry(handle=f"op-{i}") for i in range(5)]
    a = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    b = build_cell(entries, window=TimeWindow.MONTHLY, bucket=RiskBucket.MODERATE)
    assert a == b
