"""Tests for `halal_trader.web.leaderboard` (Wave 3.H).

Covers: opt-in privacy filter, time-window filter, k-anonymity floor,
deterministic ranking, no-leak render contract, template config
forbidden-key enforcement, auto_handle stability.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.web.leaderboard import (
    DEFAULT_POLICY,
    LeaderboardMetric,
    LeaderboardPolicy,
    LeaderboardRow,
    LeaderboardWindow,
    StrategyEntry,
    StrategyTemplate,
    auto_handle,
    build_leaderboard,
    render_leaderboard,
    render_template,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- LeaderboardPolicy -------------------------------


def test_default_policy_values() -> None:
    assert DEFAULT_POLICY.min_entries_to_publish == 5
    assert DEFAULT_POLICY.top_n == 10
    assert DEFAULT_POLICY.min_sample_size == 10


def test_policy_rejects_min_entries_below_2() -> None:
    with pytest.raises(ValueError, match="min_entries_to_publish"):
        LeaderboardPolicy(min_entries_to_publish=1)


def test_policy_accepts_min_entries_at_2_lower_boundary() -> None:
    p = LeaderboardPolicy(min_entries_to_publish=2)
    assert p.min_entries_to_publish == 2


def test_policy_rejects_zero_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        LeaderboardPolicy(top_n=0)


def test_policy_rejects_negative_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        LeaderboardPolicy(top_n=-1)


def test_policy_rejects_negative_min_sample_size() -> None:
    with pytest.raises(ValueError, match="min_sample_size"):
        LeaderboardPolicy(min_sample_size=-1)


def test_policy_accepts_zero_min_sample_size() -> None:
    p = LeaderboardPolicy(min_sample_size=0)
    assert p.min_sample_size == 0


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.top_n = 99  # type: ignore[misc]


# --------------------------- StrategyEntry -----------------------------------


def _entry(**overrides: object) -> StrategyEntry:
    base: dict[str, object] = {
        "user_id": "u1",
        "strategy_id": "s1",
        "display_handle": "alpha_runner",
        "strategy_kind": "momentum",
        "opt_in": True,
        "created_at": T0 - timedelta(days=60),
        "last_traded_at": T0 - timedelta(days=1),
        "sharpe": 1.5,
        "win_rate": 0.55,
        "total_return_pct": 12.0,
        "sample_size": 50,
    }
    base.update(overrides)
    return StrategyEntry(**base)  # type: ignore[arg-type]


def test_entry_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id"):
        _entry(user_id="")


def test_entry_rejects_empty_strategy_id() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        _entry(strategy_id="")


def test_entry_rejects_empty_display_handle() -> None:
    with pytest.raises(ValueError, match="display_handle"):
        _entry(display_handle="")


def test_entry_rejects_empty_strategy_kind() -> None:
    with pytest.raises(ValueError, match="strategy_kind"):
        _entry(strategy_kind="")


def test_entry_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        _entry(created_at=datetime(2026, 5, 1))


def test_entry_rejects_naive_last_traded_at() -> None:
    with pytest.raises(ValueError, match="last_traded_at"):
        _entry(last_traded_at=datetime(2026, 5, 1))


def test_entry_rejects_win_rate_above_one() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        _entry(win_rate=1.01)


def test_entry_rejects_win_rate_below_zero() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        _entry(win_rate=-0.01)


def test_entry_accepts_win_rate_boundary_zero() -> None:
    e = _entry(win_rate=0.0)
    assert e.win_rate == 0.0


def test_entry_accepts_win_rate_boundary_one() -> None:
    e = _entry(win_rate=1.0)
    assert e.win_rate == 1.0


def test_entry_rejects_negative_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        _entry(sample_size=-1)


def test_entry_is_frozen() -> None:
    e = _entry()
    with pytest.raises(FrozenInstanceError):
        e.sharpe = 999.0  # type: ignore[misc]


# --------------------------- Enum string pins --------------------------------


def test_metric_string_values_pinned() -> None:
    assert LeaderboardMetric.SHARPE.value == "sharpe"
    assert LeaderboardMetric.WIN_RATE.value == "win_rate"
    assert LeaderboardMetric.TOTAL_RETURN_PCT.value == "total_return_pct"


def test_window_string_values_pinned() -> None:
    assert LeaderboardWindow.MONTHLY.value == "monthly"
    assert LeaderboardWindow.QUARTERLY.value == "quarterly"
    assert LeaderboardWindow.YEARLY.value == "yearly"
    assert LeaderboardWindow.ALL_TIME.value == "all_time"


# --------------------------- build_leaderboard -------------------------------


def _five_entries() -> list[StrategyEntry]:
    return [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"runner_{i}",
            sharpe=1.0 + i * 0.1,
        )
        for i in range(5)
    ]


def test_build_leaderboard_filters_opt_out() -> None:
    entries = _five_entries() + [
        _entry(user_id="u_priv", strategy_id="s_priv", opt_in=False, sharpe=10.0),
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.suppressed_below_k_anonymity is False
    assert all(row.strategy_id != "s_priv" for row in lb.rows)


def test_build_leaderboard_suppresses_below_k_anonymity() -> None:
    """Pin: 4 opted-in entries → empty leaderboard, suppressed flag set."""

    entries = [
        _entry(user_id=f"u{i}", strategy_id=f"s{i}", display_handle=f"r{i}") for i in range(4)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.suppressed_below_k_anonymity is True
    assert lb.rows == ()


def test_build_leaderboard_publishes_at_5_boundary() -> None:
    """Pin: exactly 5 opted-in entries → publishes (boundary inclusive)."""

    lb = build_leaderboard(
        _five_entries(),
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.suppressed_below_k_anonymity is False
    assert len(lb.rows) == 5


def test_build_leaderboard_filters_outside_window() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            last_traded_at=T0 - timedelta(days=120),
        )
        for i in range(5)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    # All 5 traded outside 90-day window → suppressed
    assert lb.suppressed_below_k_anonymity is True


def test_build_leaderboard_all_time_includes_old_entries() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            last_traded_at=T0 - timedelta(days=2000),
        )
        for i in range(5)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.ALL_TIME,
        now=T0,
    )
    assert len(lb.rows) == 5


def test_build_leaderboard_filters_below_min_sample_size() -> None:
    entries = _five_entries() + [
        _entry(
            user_id="u_small",
            strategy_id="s_small",
            display_handle="newbie",
            sample_size=3,  # below default 10
            sharpe=99.0,
        )
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.excluded_below_min_sample == 1
    assert all(row.strategy_id != "s_small" for row in lb.rows)


def test_build_leaderboard_ranks_by_sharpe_descending() -> None:
    entries = [
        _entry(user_id=f"u{i}", strategy_id=f"s{i}", display_handle=f"r{i}", sharpe=v)
        for i, v in enumerate([0.5, 2.5, 1.5, 0.8, 1.0])
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    sharpes = [row.metric_value for row in lb.rows]
    assert sharpes == sorted(sharpes, reverse=True)


def test_build_leaderboard_ranks_by_win_rate() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            win_rate=v,
        )
        for i, v in enumerate([0.5, 0.7, 0.55, 0.6, 0.65])
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.WIN_RATE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.rows[0].metric_value == 0.7


def test_build_leaderboard_ranks_by_total_return() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            total_return_pct=v,
        )
        for i, v in enumerate([5.0, 25.0, -3.0, 12.0, 8.0])
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.TOTAL_RETURN_PCT,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.rows[0].metric_value == 25.0
    assert lb.rows[-1].metric_value == -3.0


def test_build_leaderboard_tiebreak_older_strategy_first() -> None:
    """Pin: same Sharpe → older created_at ranks higher."""

    entries = [
        _entry(
            user_id="u_old",
            strategy_id="s_old",
            display_handle="veteran",
            created_at=T0 - timedelta(days=365),
            sharpe=1.5,
        ),
        _entry(
            user_id="u_new",
            strategy_id="s_new",
            display_handle="rookie",
            created_at=T0 - timedelta(days=10),
            sharpe=1.5,
        ),
        _entry(user_id="u3", strategy_id="s3", display_handle="r3", sharpe=1.0),
        _entry(user_id="u4", strategy_id="s4", display_handle="r4", sharpe=0.8),
        _entry(user_id="u5", strategy_id="s5", display_handle="r5", sharpe=0.6),
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    # Veteran ranks above rookie despite same Sharpe
    assert lb.rows[0].display_handle == "veteran"
    assert lb.rows[1].display_handle == "rookie"


def test_build_leaderboard_top_n_caps_to_policy() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            sharpe=float(i),
        )
        for i in range(15)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
        policy=LeaderboardPolicy(top_n=3),
    )
    assert len(lb.rows) == 3
    # Best 3 sharpes
    assert {row.metric_value for row in lb.rows} == {14.0, 13.0, 12.0}


def test_build_leaderboard_rank_starts_at_1() -> None:
    lb = build_leaderboard(
        _five_entries(),
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.rows[0].rank == 1
    assert lb.rows[-1].rank == 5


def test_build_leaderboard_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        build_leaderboard(
            [],
            metric=LeaderboardMetric.SHARPE,
            window=LeaderboardWindow.QUARTERLY,
            now=datetime(2026, 5, 1),
        )


def test_build_leaderboard_empty_input() -> None:
    lb = build_leaderboard(
        [],
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert lb.suppressed_below_k_anonymity is True
    assert lb.eligible_count == 0


def test_build_leaderboard_is_deterministic() -> None:
    entries = _five_entries()
    a = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    b = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert a == b


def test_leaderboard_row_is_anonymous() -> None:
    """Pin: LeaderboardRow has no user_id field."""

    fields = LeaderboardRow.__dataclass_fields__
    assert "user_id" not in fields
    assert "email" not in fields


def test_build_leaderboard_window_boundaries() -> None:
    """Pin: 30-day window includes entry traded exactly 30 days ago."""

    boundary = T0 - timedelta(days=30)
    entries = [
        _entry(
            user_id=f"u{i}", strategy_id=f"s{i}", display_handle=f"r{i}", last_traded_at=boundary
        )
        for i in range(5)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.MONTHLY,
        now=T0,
    )
    assert len(lb.rows) == 5


# ---------------------------- auto_handle ------------------------------------


def test_auto_handle_format() -> None:
    h = auto_handle("user_alice")
    assert h.startswith("strategist_")
    assert len(h) == len("strategist_") + 8


def test_auto_handle_is_stable() -> None:
    a = auto_handle("user_alice")
    b = auto_handle("user_alice")
    assert a == b


def test_auto_handle_different_users_different_handles() -> None:
    assert auto_handle("user_alice") != auto_handle("user_bob")


def test_auto_handle_does_not_leak_user_id() -> None:
    h = auto_handle("ahmed.elghareeb@example.com")
    assert "ahmed" not in h
    assert "elghareeb" not in h
    assert "@" not in h


def test_auto_handle_rejects_empty() -> None:
    with pytest.raises(ValueError, match="user_id"):
        auto_handle("")


def test_auto_handle_rejects_whitespace() -> None:
    with pytest.raises(ValueError, match="user_id"):
        auto_handle("   ")


# ----------------------------- StrategyTemplate ------------------------------


def test_template_rejects_empty_template_id() -> None:
    with pytest.raises(ValueError, match="template_id"):
        StrategyTemplate(
            template_id="",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(),
            created_at=T0,
        )


def test_template_rejects_empty_display_handle() -> None:
    with pytest.raises(ValueError, match="display_handle"):
        StrategyTemplate(
            template_id="t1",
            display_handle="",
            strategy_kind="momentum",
            config=(),
            created_at=T0,
        )


def test_template_rejects_empty_strategy_kind() -> None:
    with pytest.raises(ValueError, match="strategy_kind"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="",
            config=(),
            created_at=T0,
        )


def test_template_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(),
            created_at=datetime(2026, 5, 1),
        )


def test_template_rejects_user_id_in_config() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(("user_id", "u1"),),
            created_at=T0,
        )


def test_template_rejects_email_in_config() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(("email", "a@b.com"),),
            created_at=T0,
        )


def test_template_rejects_broker_api_key_in_config() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(("broker_api_key", "secret"),),
            created_at=T0,
        )


def test_template_rejects_stripe_id_in_config() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(("stripe_id", "cus_X"),),
            created_at=T0,
        )


def test_template_forbidden_check_case_insensitive() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        StrategyTemplate(
            template_id="t1",
            display_handle="alpha",
            strategy_kind="momentum",
            config=(("USER_ID", "u1"),),
            created_at=T0,
        )


def test_template_accepts_safe_config() -> None:
    t = StrategyTemplate(
        template_id="t1",
        display_handle="alpha",
        strategy_kind="momentum",
        config=(
            ("rsi_threshold", "30"),
            ("position_pct", "0.05"),
            ("stop_loss_pct", "0.02"),
        ),
        created_at=T0,
    )
    assert len(t.config) == 3


def test_template_is_frozen() -> None:
    t = StrategyTemplate(
        template_id="t1",
        display_handle="alpha",
        strategy_kind="momentum",
        config=(),
        created_at=T0,
    )
    with pytest.raises(FrozenInstanceError):
        t.template_id = "other"  # type: ignore[misc]


# --------------------------- render_leaderboard ------------------------------


def test_render_includes_metric_and_window() -> None:
    lb = build_leaderboard(
        _five_entries(),
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    assert "sharpe" in out
    assert "quarterly" in out


def test_render_shows_rank_and_handle() -> None:
    lb = build_leaderboard(
        _five_entries(),
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    assert "#1" in out
    assert "runner_" in out


def test_render_does_not_leak_user_id() -> None:
    """Pin: render output never contains user_id, email, or stripe ID."""

    entries = [
        _entry(
            user_id=f"sensitive_user_id_{i}",
            strategy_id=f"s{i}",
            display_handle=f"handle_{i}",
        )
        for i in range(5)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    assert "sensitive_user_id" not in out
    # No email-shaped substrings (text@domain.tld)
    import re

    assert not re.search(r"\w+@\w+\.\w+", out)
    assert "cus_" not in out.lower()
    assert "$" not in out


def test_render_suppressed_message_when_too_few() -> None:
    entries = [
        _entry(user_id=f"u{i}", strategy_id=f"s{i}", display_handle=f"r{i}") for i in range(3)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    assert "suppressed" in out


def test_render_win_rate_formats_as_percentage() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            win_rate=0.6,
        )
        for i in range(5)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.WIN_RATE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    assert "60.0%" in out


def test_render_total_return_signed() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"r{i}",
            total_return_pct=v,
        )
        for i, v in enumerate([5.0, 25.0, -3.0, 12.0, 8.0])
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.TOTAL_RETURN_PCT,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    # Highest is +25.0%
    assert "+25.0%" in out
    # Lowest in the cohort is -3.0%
    assert "-3.0%" in out


# ---------------------------- render_template --------------------------------


def test_render_template_includes_handle_and_kind() -> None:
    t = StrategyTemplate(
        template_id="t1",
        display_handle="alpha_runner",
        strategy_kind="momentum",
        config=(),
        created_at=T0,
    )
    out = render_template(t)
    assert "alpha_runner" in out
    assert "momentum" in out


def test_render_template_includes_config() -> None:
    t = StrategyTemplate(
        template_id="t1",
        display_handle="alpha",
        strategy_kind="momentum",
        config=(("rsi_threshold", "30"),),
        created_at=T0,
    )
    out = render_template(t)
    assert "rsi_threshold" in out
    assert "30" in out


def test_render_template_does_not_leak_secrets() -> None:
    t = StrategyTemplate(
        template_id="t1",
        display_handle="alpha",
        strategy_kind="momentum",
        config=(),
        created_at=T0,
    )
    out = render_template(t)
    assert "user_id" not in out.lower()
    assert "@" not in out
    assert "broker_api_key" not in out


# ----------------------------- e2e flows -------------------------------------


def test_e2e_mixed_opt_in_status_only_publishes_opted_in() -> None:
    entries = (
        # 6 opted-in
        [
            _entry(
                user_id=f"u_in_{i}",
                strategy_id=f"s_in_{i}",
                display_handle=f"public_{i}",
                opt_in=True,
                sharpe=1.0 + i * 0.1,
            )
            for i in range(6)
        ]
        # 4 opted-out (would have ranked very high but private)
        + [
            _entry(
                user_id=f"u_priv_{i}",
                strategy_id=f"s_priv_{i}",
                display_handle=f"private_{i}",
                opt_in=False,
                sharpe=99.0,
            )
            for i in range(4)
        ]
    )
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    assert all("private_" not in row.display_handle for row in lb.rows)
    # Top opt-in entry is sharpe 1.5, NOT 99.0
    assert lb.rows[0].metric_value < 2.0


def test_e2e_full_render() -> None:
    entries = [
        _entry(
            user_id=f"u{i}",
            strategy_id=f"s{i}",
            display_handle=f"runner_{i}",
            sharpe=1.0 + i * 0.2,
        )
        for i in range(7)
    ]
    lb = build_leaderboard(
        entries,
        metric=LeaderboardMetric.SHARPE,
        window=LeaderboardWindow.QUARTERLY,
        now=T0,
    )
    out = render_leaderboard(lb)
    # Header
    assert "🏆" in out
    assert "Leaderboard" in out
    # All ranks present
    for i in range(1, 8):
        assert f"#{i}" in out
