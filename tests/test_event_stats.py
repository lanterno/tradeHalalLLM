"""Trial statistics (events/stats.py): known answers computed by hand."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import pytest

from halabot.playbooks.records import Leg as SimLeg
from halal_trader.events.stats import (
    ClusteredMean,
    Leg,
    LegLike,
    calendar_series,
    cluster_bootstrap_ci,
    clustered_mean,
    holm,
    newey_west_mean,
    p_one_sided,
    t_quantile,
    tost,
)


def _p_df2(t: float) -> float:
    """P(T_2 >= t), in closed form: 1/2 - t / (2 sqrt(2 + t^2))."""
    return 0.5 - t / (2.0 * math.sqrt(2.0 + t * t))


# ── Student t ────────────────────────────────────────────────────


def test_t_quantiles_match_the_closed_forms() -> None:
    # df = 1 is Cauchy: tan(pi (p - 1/2)); df = 2: (2p - 1) / sqrt(2 p (1 - p)).
    assert t_quantile(0.975, 1) == pytest.approx(math.tan(math.pi * 0.475), rel=1e-10)
    for p in (0.9, 0.95, 0.975, 0.995):
        assert t_quantile(p, 2) == pytest.approx(
            (2 * p - 1) / math.sqrt(2 * p * (1 - p)), rel=1e-10
        )
    assert t_quantile(0.975, 10) == pytest.approx(2.2281388519649385, rel=1e-9)  # the table value


def test_t_quantile_is_symmetric_and_centred() -> None:
    assert t_quantile(0.5, 7) == 0.0
    assert t_quantile(0.05, 7) == pytest.approx(-t_quantile(0.95, 7), rel=1e-12)
    with pytest.raises(ValueError):
        t_quantile(1.0, 7)
    with pytest.raises(ValueError):
        t_quantile(0.9, 0)


def test_one_sided_p_halves_the_two_sided_tail() -> None:
    assert p_one_sided(1.5, 2) == pytest.approx(_p_df2(1.5), rel=1e-10)
    assert p_one_sided(-1.5, 2) == pytest.approx(1.0 - _p_df2(1.5), rel=1e-10)
    assert p_one_sided(0.0, 2) == pytest.approx(0.5)


# ── CR1 clustered mean ───────────────────────────────────────────

# Six trades on three entry dates, in basis points: A = {100, 300}, B = {-200},
# C = {400, 0, 200}. By hand: mean = 800/6 = 400/3 bp; the cluster residual
# sums are E_A = 400 - 2(400/3) = 400/3, E_B = -200 - 400/3 = -1000/3 and
# E_C = 600 - 3(400/3) = 200; sum E^2 = (160000 + 1000000 + 360000)/9 =
# 1520000/9; V = (3/2)(1520000/9)/36 = 190000/27 bp^2. The residuals
# -100/3, 500/3, -1000/3, 800/3, -400/3, 200/3 give s^2 = 2100000/9/5 =
# 140000/3 bp^2, so DEFF = V / (s^2/6) = (190000/27) / (70000/9) = 19/21.
BP = 1e-4
VALUES = [100 * BP, 300 * BP, -200 * BP, 400 * BP, 0.0, 200 * BP]
DATES = ["A", "A", "B", "C", "C", "C"]


def test_cr1_matches_the_hand_computation() -> None:
    cm = clustered_mean(VALUES, DATES)

    assert cm is not None
    variance = 190000 / 27 * BP**2
    assert (cm.n, cm.clusters, cm.df) == (6, 3, 2)
    assert cm.mean == pytest.approx(400 / 3 * BP, rel=1e-12)
    assert cm.se == pytest.approx(math.sqrt(variance), rel=1e-12)
    assert cm.t == pytest.approx((400 / 3 * BP) / math.sqrt(variance), rel=1e-12)
    assert cm.deff == pytest.approx(19 / 21, rel=1e-12)
    assert cm.p_one_sided == pytest.approx(_p_df2(cm.t), rel=1e-9)


def test_cr1_with_one_trade_per_date_is_the_iid_standard_error() -> None:
    # G/(G-1) * sum e^2 / N^2 = s^2/N when every cluster holds one trade.
    values = [0.012, -0.004, 0.031, 0.007, -0.015, 0.022, 0.001]
    cm = clustered_mean(values, list(range(len(values))))

    assert cm is not None
    assert cm.se == pytest.approx(float(np.std(values, ddof=1)) / math.sqrt(len(values)))
    assert cm.deff == pytest.approx(1.0)
    assert cm.df == len(values) - 1


def test_cr1_sees_dependence_inside_a_date() -> None:
    # The same values repeated on one date each: N grows but the evidence does not.
    single = clustered_mean([0.01, -0.02, 0.03], ["a", "b", "c"])
    repeated = clustered_mean(
        [0.01] * 5 + [-0.02] * 5 + [0.03] * 5, ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    )

    assert single is not None and repeated is not None
    assert repeated.t == pytest.approx(single.t)
    assert repeated.deff > 4.0


def test_a_negative_mean_has_a_one_sided_p_above_a_half() -> None:
    cm = clustered_mean([-v for v in VALUES], DATES)

    assert cm is not None
    assert cm.t < 0
    assert cm.p_one_sided == pytest.approx(1.0 - _p_df2(-cm.t), rel=1e-9)


def test_cr1_needs_two_dates_and_a_variance() -> None:
    assert clustered_mean([], []) is None
    assert clustered_mean([0.01, 0.02], ["d", "d"]) is None  # one cluster: df = 0
    assert clustered_mean([0.01, 0.01, 0.01], ["a", "b", "c"]) is None  # zero variance
    with pytest.raises(ValueError):
        clustered_mean([0.01, 0.02], ["a"])
    with pytest.raises(ValueError):
        clustered_mean([0.01, math.nan], ["a", "b"])


def test_the_clustered_interval_uses_t_on_g_minus_1() -> None:
    cm = clustered_mean(VALUES, DATES)

    assert cm is not None
    lo, hi = cm.ci(0.95)
    t2 = 0.95 / math.sqrt(2 * 0.975 * 0.025)  # t_{0.975, 2} in closed form
    assert lo == pytest.approx(cm.mean - t2 * cm.se, rel=1e-9)
    assert hi == pytest.approx(cm.mean + t2 * cm.se, rel=1e-9)


# ── Newey-West ───────────────────────────────────────────────────

# x = 1, 3, 2, 6: mean 3, residuals e = -2, 0, -1, 3, T = 4.
# gamma_0 = 14/4; gamma_1 = (0*-2 + -1*0 + 3*-1)/4 = -3/4;
# gamma_2 = (-1*-2 + 3*0)/4 = 1/2; gamma_3 = (3*-2)/4 = -3/2; gamma_4 = 0.
SERIES = [1.0, 3.0, 2.0, 6.0]


def test_newey_west_matches_the_hand_computation() -> None:
    # 4 lags, Bartlett weights 4/5, 3/5, 2/5, 1/5:
    # V = (3.5 + 2(0.8*-0.75 + 0.6*0.5 + 0.4*-1.5 + 0.2*0)) / 4 = 1.7/4.
    nw = newey_west_mean(SERIES)

    assert nw is not None
    assert (nw.n, nw.df) == (4, 3)
    assert nw.mean == 3.0
    assert nw.se == pytest.approx(math.sqrt(1.7 / 4), rel=1e-12)
    assert nw.t == pytest.approx(3.0 / math.sqrt(1.7 / 4), rel=1e-12)
    assert nw.p_one_sided == pytest.approx(p_one_sided(nw.t, 3), rel=1e-12)


def test_newey_west_with_fewer_lags() -> None:
    one = newey_west_mean(SERIES, lags=1)  # V = (3.5 + 2 * 0.5 * -0.75) / 4
    none = newey_west_mean(SERIES, lags=0)  # V = gamma_0 / T

    assert one is not None and none is not None
    assert one.se == pytest.approx(math.sqrt(2.75 / 4), rel=1e-12)
    assert none.se == pytest.approx(math.sqrt(3.5 / 4), rel=1e-12)


def test_newey_west_needs_two_points_and_a_variance() -> None:
    assert newey_west_mean([0.01]) is None
    assert newey_west_mean([0.01, 0.01, 0.01]) is None
    with pytest.raises(ValueError):
        newey_west_mean(SERIES, lags=-1)


# ── calendar series ──────────────────────────────────────────────

D1, D2, D3, D4 = date(2024, 3, 18), date(2024, 3, 19), date(2024, 3, 20), date(2024, 3, 21)


def test_the_calendar_series_averages_the_open_legs_of_each_session() -> None:
    legs = {
        "A:2024-03-18": [Leg(D1, 0.02), Leg(D2, -0.01)],
        "B:2024-03-19": [Leg(D2, 0.03), Leg(D4, 0.01)],
        "C:2024-03-19": [Leg(D2, 0.04)],
    }

    series = calendar_series(legs, [D4, D1, D2, D3])  # D3 holds nothing: no point

    assert [d for d, _ in series] == [D1, D2, D4]
    assert [x for _, x in series] == pytest.approx([0.02, 0.02, 0.01])


def test_the_calendar_series_takes_the_simulators_legs_as_they_are() -> None:
    # records.Leg carries the bars and the cost; its abn is z/a - 1 - (Z/A - 1) - cost.
    entry = SimLeg(D1, a=100.0, z=102.0, spy_a=400.0, spy_z=404.0, cost=0.001)  # 0.009
    held = SimLeg(D2, a=102.0, z=101.0, spy_a=404.0, spy_z=404.0)  # -1/102
    other = Leg(D2, 0.03)
    legs: dict[str, list[LegLike]] = {"A:2024-03-18": [entry, held], "B:2024-03-19": [other]}

    series = calendar_series(legs, [D1, D2])

    assert series == [
        (D1, pytest.approx(0.009, rel=1e-12)),
        (D2, pytest.approx((-1.0 / 102.0 + 0.03) / 2.0, rel=1e-12)),
    ]


def test_a_leg_off_the_calendar_or_not_finite_is_an_error() -> None:
    with pytest.raises(ValueError, match="not a session"):
        calendar_series({"A": [Leg(D3, 0.01)]}, [D1, D2])
    with pytest.raises(ValueError, match="non-finite"):
        calendar_series({"A": [Leg(D1, math.nan)]}, [D1, D2])
    with pytest.raises(ValueError, match="non-finite"):
        calendar_series({"A": [SimLeg(D1, a=1.0, z=math.nan, spy_a=1.0, spy_z=1.0)]}, [D1])


# ── Holm ─────────────────────────────────────────────────────────


def test_holm_on_the_textbook_example() -> None:
    # Four tests at 0.05: 0.005 <= 0.0125, 0.01 <= 0.0167, then 0.03 > 0.025 stops.
    assert holm({"H1": 0.01, "H2": 0.04, "H3": 0.03, "H4": 0.005}) == {
        "H1": True,
        "H2": False,
        "H3": False,
        "H4": True,
    }


def test_holm_rejects_more_than_bonferroni() -> None:
    # Bonferroni (0.05/3 = 0.0167) keeps only the first; Holm steps down through all three.
    assert holm({"a": 0.01, "b": 0.02, "c": 0.03}) == {"a": True, "b": True, "c": True}


def test_holm_stops_at_the_first_failure() -> None:
    # The second p would pass alpha/1, but the first fails alpha/2, so nothing is rejected.
    assert holm({"a": 0.03, "b": 0.03}) == {"a": False, "b": False}
    assert holm({"a": 0.025, "b": 0.05}) == {"a": True, "b": True}  # bounds are inclusive
    assert holm({"only": 0.05}) == {"only": True}
    assert holm({}) == {}


def test_holm_refuses_a_p_value_that_is_not_one() -> None:
    with pytest.raises(ValueError):
        holm({"a": math.nan})
    with pytest.raises(ValueError):
        holm({"a": 1.2})


# ── cluster bootstrap ────────────────────────────────────────────


def _mean_of(values: Sequence[float]) -> Callable[[Sequence[int]], float]:
    return lambda idx: float(np.mean([values[i] for i in idx]))


def test_the_bootstrap_resamples_whole_clusters() -> None:
    # Two dates: A = {0, 0}, B = {1, 1, 1, 1}. A resample is AA (mean 0, p 1/4),
    # AB or BA (4/6, p 1/2) or BB (1, p 1/4): the 95% interval spans [0, 1].
    values = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    clusters = ["A", "A", "B", "B", "B", "B"]

    assert cluster_bootstrap_ci(_mean_of(values), clusters) == (0.0, 1.0)


def test_the_bootstrap_is_seeded_and_follows_the_stated_scheme() -> None:
    rng = np.random.default_rng(3)
    clusters = [i // 4 for i in range(80)]  # 20 dates of 4
    values = list(rng.normal(0.001, 0.02, 80))
    stat = _mean_of(values)

    got = cluster_bootstrap_ci(stat, clusters, b=500, seed=11)

    groups = [list(range(4 * g, 4 * g + 4)) for g in range(20)]
    draws = np.random.default_rng(11).integers(0, 20, size=(500, 20))
    replay = [stat([i for g in row for i in groups[g]]) for row in draws.tolist()]
    assert got == pytest.approx(tuple(np.quantile(replay, [0.025, 0.975])), rel=1e-12)
    assert cluster_bootstrap_ci(stat, clusters, b=500, seed=11) == got
    assert cluster_bootstrap_ci(stat, clusters, b=500, seed=12) != got


def test_the_bootstrap_interval_agrees_with_cr1_on_a_large_sample() -> None:
    rng = np.random.default_rng(5)
    shocks = rng.normal(0.0, 0.02, 200)  # a common shock per date
    clusters = [g for g in range(200) for _ in range(5)]
    values = [float(shocks[g] + rng.normal(0.002, 0.01)) for g in clusters]
    cm = clustered_mean(values, clusters)

    lo, hi = cluster_bootstrap_ci(_mean_of(values), clusters)

    assert cm is not None
    cr_lo, cr_hi = cm.ci(0.95)
    assert (hi - lo) == pytest.approx(cr_hi - cr_lo, rel=0.15)
    assert lo < cm.mean < hi


def test_the_bootstrap_drops_undefined_draws_and_needs_two_clusters() -> None:
    values = [0.0, 1.0, 2.0]
    clusters = ["a", "b", "c"]

    def stat(idx: Sequence[int]) -> float:
        picked = [values[i] for i in idx]
        return math.nan if 0.0 in picked else float(np.mean(picked))

    lo, hi = cluster_bootstrap_ci(stat, clusters, b=200)
    assert 1.0 <= lo <= hi <= 2.0
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(_mean_of(values), ["a", "a", "a"])
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(lambda idx: math.nan, clusters, b=10)


# ── TOST ─────────────────────────────────────────────────────────


def _differences(mean: float, spread: float, dates: int = 30) -> tuple[list[float], list[int]]:
    """Two differences per date, mean ``mean``, symmetric spread across dates."""
    values, clusters = [], []
    for g in range(dates):
        sign = 1.0 if g % 2 else -1.0
        values += [mean + sign * spread, mean + sign * spread]
        clusters += [g, g]
    return values, clusters


def test_tost_passes_a_tight_interval_around_zero() -> None:
    values, clusters = _differences(0.0002, 0.0005)
    assert tost(values, clusters, margin=0.001)


def test_tost_fails_a_mean_beyond_the_margin_or_an_interval_too_wide() -> None:
    assert not tost(*_differences(0.0012, 0.0001), margin=0.001)
    assert not tost(*_differences(-0.0012, 0.0001), margin=0.001)
    assert not tost(*_differences(0.0, 0.02), margin=0.001)  # cannot show equivalence


def test_tost_bounds_are_inclusive() -> None:
    values, clusters = _differences(0.0003, 0.002)
    cm = clustered_mean(values, clusters)
    assert cm is not None
    upper = cm.mean + t_quantile(0.95, cm.df) * cm.se  # the 90% interval's upper end

    assert tost(values, clusters, margin=upper * (1 + 1e-9))
    assert not tost(values, clusters, margin=upper * (1 - 1e-9))


def test_tost_edge_cases() -> None:
    assert not tost([0.0, 0.0], ["d", "d"])  # one date shows nothing
    assert not tost([], [])
    assert tost([0.0005] * 4, ["a", "b", "c", "d"])  # no variance: the mean decides
    assert not tost([0.0015] * 4, ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        tost([0.0, 0.1], ["a", "b"], margin=0.0)
    with pytest.raises(ValueError):
        tost([0.0, 0.1], ["a", "b"], alpha=0.5)


def test_results_are_frozen() -> None:
    cm = clustered_mean(VALUES, DATES)
    assert isinstance(cm, ClusteredMean)
    with pytest.raises(AttributeError):
        cm.mean = 0.0
