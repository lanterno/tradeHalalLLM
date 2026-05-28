"""Significance tests for the Phase-3 promotion gate (REARCHITECTURE Part IV).

The Phase-3 gate is a *significance* test, not a fixed tiny n (fix R, "≥5
sessions" was meaningless). We do not flip to conviction-driven execution until
the shadow **significantly** beats the churning live cycle on real sessions:
materially lower churn at no worse realized P&L, with enough samples that the
difference isn't noise.

Pure-Python + deterministic (no scipy): Welch's t-test with a Student-t p-value
via the regularized incomplete beta (Numerical Recipes ``betacf``), plus Cohen's
d effect size. Everything here is unit-tested against known values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs: list[float]) -> float:
    """Sample variance (n-1). 0.0 for fewer than 2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    MAXIT, EPS, FPMIN = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided p-value for a t-statistic with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    return betai(0.5 * df, 0.5, df / (df + t * t))


@dataclass(frozen=True)
class TTestResult:
    t: float
    df: float
    p_two_sided: float
    mean_a: float
    mean_b: float

    @property
    def p_one_sided_a_greater(self) -> float:
        """p that mean_a > mean_b (one-sided). Small ⇒ a significantly exceeds b."""
        half = self.p_two_sided / 2.0
        return half if self.t > 0 else 1.0 - half


def welch_t_test(a: list[float], b: list[float]) -> TTestResult | None:
    """Welch's unequal-variance t-test. None if either side has < 2 points."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = mean(a), mean(b)
    va, vb = variance(a), variance(b)
    sa, sb = va / na, vb / nb
    denom = sa + sb
    if denom <= 0:
        return None
    t = (ma - mb) / math.sqrt(denom)
    df = denom * denom / (sa * sa / (na - 1) + sb * sb / (nb - 1))
    return TTestResult(t=t, df=df, p_two_sided=student_t_sf_two_sided(t, df), mean_a=ma, mean_b=mb)


def cohens_d(a: list[float], b: list[float]) -> float | None:
    """Pooled-SD standardized mean difference (a − b). None if undersized."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = ((na - 1) * variance(a) + (nb - 1) * variance(b)) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (mean(a) - mean(b)) / math.sqrt(pooled)


@dataclass(frozen=True)
class PromotionVerdict:
    promote: bool
    reasons: list[str]
    n_shadow: int
    n_live: int
    shadow_mean: float
    live_mean: float
    effect_size: float | None
    p_two_sided: float | None
    churn_reduction: float | None


def promotion_gate(
    shadow_returns: list[float],
    live_returns: list[float],
    *,
    churn_reduction: float | None,
    alpha: float = 0.05,
    min_n: int = 30,
    min_churn_reduction: float = 0.2,
) -> PromotionVerdict:
    """Phase-3 → Phase-4 gate (shadow-must-beat-live, statistically).

    Promote only when ALL hold:
    * enough closed samples on both sides (``min_n``),
    * the shadow churns materially less (``churn_reduction ≥ min_churn_reduction``),
    * the shadow is NOT significantly worse on realized P&L — i.e. it isn't the
      case that live significantly exceeds shadow at ``alpha`` (one-sided).

    This encodes "significantly lower churn at ≥ live P&L" without requiring the
    shadow to *beat* live P&L (lower churn at parity is already a win)."""
    reasons: list[str] = []
    tt = welch_t_test(shadow_returns, live_returns)
    d = cohens_d(shadow_returns, live_returns)
    s_mean, l_mean = mean(shadow_returns), mean(live_returns)

    enough = len(shadow_returns) >= min_n and len(live_returns) >= min_n
    if not enough:
        reasons.append(
            f"insufficient samples (shadow={len(shadow_returns)}, "
            f"live={len(live_returns)}, need {min_n})"
        )

    churn_ok = churn_reduction is not None and churn_reduction >= min_churn_reduction
    if not churn_ok:
        reasons.append(f"churn reduction {churn_reduction} < {min_churn_reduction}")

    # Shadow must not be SIGNIFICANTLY worse: reject if live mean exceeds shadow
    # mean with a significant one-sided p (live > shadow).
    not_worse = True
    if tt is not None and l_mean > s_mean:
        p_live_greater = tt.p_two_sided / 2.0
        if p_live_greater < alpha:
            not_worse = False
            reasons.append(f"shadow significantly worse on P&L (p={p_live_greater:.3f} < {alpha})")

    promote = bool(enough and churn_ok and not_worse)
    if promote:
        reasons.append("PASS: lower churn at no-worse P&L, sufficient samples")
    return PromotionVerdict(
        promote=promote,
        reasons=reasons,
        n_shadow=len(shadow_returns),
        n_live=len(live_returns),
        shadow_mean=s_mean,
        live_mean=l_mean,
        effect_size=d,
        p_two_sided=tt.p_two_sided if tt else None,
        churn_reduction=churn_reduction,
    )
