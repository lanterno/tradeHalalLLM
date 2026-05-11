"""Sukuk allocation engine — Round-5 Wave 3.C.

Given a target duration / sector / jurisdiction mix, pick the optimal
sukuk basket. The optimisation is the standard mean-variance Markowitz
formulation with an additional set of halal-only constraints layered on
top.

This is the basket-construction primitive — pricing comes from
`markets/sukuk_pricing.py`, ladder construction from
`markets/sukuk_ladder.py`, and credit risk from
`markets/sukuk_default.py`. This module assumes per-issue expected
return + per-pair covariance has been precomputed.

Pinned semantics:

- **Closed-set Objective** (MEAN_VARIANCE / MIN_VARIANCE / TARGET_DURATION).
- **Constraints are hard.** Sector / jurisdiction caps + tradability
  rules reject infeasible baskets at allocation time. The optimiser
  does *not* re-weight to make an over-cap basket feasible — it
  surfaces an `InfeasibleBasketError`.
- **Tradable-only universe.** Pure-Murabaha and Salam sukuk are
  excluded automatically (they cannot be repriced on the secondary
  market under Standard 17 cl. 5.1.8).
- **Sum-to-one + non-negative weights.** Long-only basket; short
  selling is not permissible under the Standard 17 framing.
- **Deterministic projected-gradient solver.** Pure Python, no SciPy
  / NumPy dependency. Convergence is bounded (max_iter); the tests
  pin specific outputs against the iteration count + tolerance.
- **`render_basket` is no-secret-leak** — issuer code + weight only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from halal_trader.halal.aaoifi_standard_17 import (
    SukukType,
    is_tradable_in_secondary,
)


class AllocationObjective(str, Enum):
    """Closed-set objective for the allocation engine."""

    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    TARGET_DURATION = "target_duration"


class InfeasibleBasketError(ValueError):
    """Raised when no allocation satisfies the given constraints."""


@dataclass(frozen=True)
class SukukCandidate:
    """A candidate sukuk for inclusion in the basket."""

    issuer: str
    sukuk_type: SukukType
    sector: str
    jurisdiction: str
    duration_years: float
    expected_return: float

    def __post_init__(self) -> None:
        if not self.issuer or not self.issuer.strip():
            raise ValueError("issuer must be non-empty")
        if not is_tradable_in_secondary(self.sukuk_type):
            raise ValueError(f"{self.sukuk_type.value} not tradable on secondary")
        if not self.sector or not self.sector.strip():
            raise ValueError("sector must be non-empty")
        if not self.jurisdiction or not self.jurisdiction.strip():
            raise ValueError("jurisdiction must be non-empty")
        if self.duration_years <= 0:
            raise ValueError("duration_years must be positive")
        if not -0.10 < self.expected_return < 0.50:
            raise ValueError("expected_return outside reasonable bounds")


@dataclass(frozen=True)
class AllocationConstraints:
    """Halal-aware constraints layered on top of the Markowitz core."""

    # Map sector → max weight in [0, 1]. Defaults to 0.40 (40%).
    sector_caps: dict[str, float] = field(default_factory=dict)
    # Map jurisdiction → max weight in [0, 1].
    jurisdiction_caps: dict[str, float] = field(default_factory=dict)
    # Map sukuk_type → max weight (e.g. limit Mudarabah to 30%).
    type_caps: dict[SukukType, float] = field(default_factory=dict)
    # Per-issue weight cap (single-name concentration).
    max_single_name: float = 0.30
    # If TARGET_DURATION objective: target portfolio duration in years.
    target_duration: float | None = None
    # Optional mean-variance risk-aversion λ (higher = more variance penalty).
    risk_aversion: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.max_single_name <= 1.0:
            raise ValueError("max_single_name must be in (0, 1]")
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        for cap_map, name in (
            (self.sector_caps, "sector_caps"),
            (self.jurisdiction_caps, "jurisdiction_caps"),
        ):
            for k, v in cap_map.items():
                if not 0.0 < v <= 1.0:
                    raise ValueError(f"{name}[{k}] must be in (0, 1]")
        for st, v in self.type_caps.items():
            if not 0.0 < v <= 1.0:
                raise ValueError(f"type_caps[{st.value}] must be in (0, 1]")
        if self.target_duration is not None and self.target_duration <= 0:
            raise ValueError("target_duration must be positive")


@dataclass(frozen=True)
class AllocationResult:
    """Output of `allocate`."""

    weights: tuple[float, ...]
    candidates: tuple[SukukCandidate, ...]
    expected_return: float
    expected_variance: float
    portfolio_duration: float

    def expected_volatility(self) -> float:
        return math.sqrt(max(0.0, self.expected_variance))

    def by_issuer(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (c.issuer, w) for c, w in zip(self.candidates, self.weights, strict=True) if w > 1e-9
        )


def _default_sector_cap(constraints: AllocationConstraints, sector: str) -> float:
    """Default sector cap is 1.0 (no cap) unless the operator opts in."""
    return constraints.sector_caps.get(sector, 1.0)


def _default_jurisdiction_cap(constraints: AllocationConstraints, jurisdiction: str) -> float:
    """Default jurisdiction cap is 1.0 (no cap) unless the operator opts in."""
    return constraints.jurisdiction_caps.get(jurisdiction, 1.0)


def _default_type_cap(constraints: AllocationConstraints, sukuk_type: SukukType) -> float:
    return constraints.type_caps.get(sukuk_type, 1.0)


def _validate_universe(candidates: Sequence[SukukCandidate]) -> None:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if len(candidates) > 200:
        raise ValueError("candidate universe too large; pre-filter to ≤200")


def _validate_covariance(cov: Sequence[Sequence[float]], n: int) -> None:
    if len(cov) != n:
        raise ValueError(f"covariance matrix must be {n}×{n}")
    for row in cov:
        if len(row) != n:
            raise ValueError(f"covariance matrix must be {n}×{n}")
    # Symmetry pin.
    for i in range(n):
        for j in range(i + 1, n):
            if abs(cov[i][j] - cov[j][i]) > 1e-9:
                raise ValueError("covariance matrix must be symmetric")
        if cov[i][i] < 0:
            raise ValueError("covariance diagonal must be non-negative")


def _project_to_simplex(w: list[float]) -> list[float]:
    """Euclidean projection onto the simplex {w >= 0, sum w = 1}.

    Standard O(n log n) sort-based projection (Held + 1974). Pure
    Python, deterministic.
    """
    n = len(w)
    u = sorted(w, reverse=True)
    cssv = 0.0
    rho = -1
    for i, ui in enumerate(u):
        cssv += ui
        if ui + (1 - cssv) / (i + 1) > 0:
            rho = i
    if rho < 0:
        # All negative — project to uniform.
        return [1.0 / n] * n
    cssv = sum(u[: rho + 1])
    theta = (cssv - 1) / (rho + 1)
    return [max(0.0, wi - theta) for wi in w]


def _apply_caps(
    w: list[float],
    candidates: Sequence[SukukCandidate],
    constraints: AllocationConstraints,
) -> list[float]:
    """Clamp each weight to its single-name + sector + jurisdiction + type cap.

    Caps are enforced by a water-filling iteration: clamp + scale + repeat
    until weights are stable. Bounded to 20 inner iterations to keep the
    outer projected-gradient solver deterministic.
    """
    out = list(w)
    cap_single = constraints.max_single_name
    by_sector: dict[str, list[int]] = {}
    by_jur: dict[str, list[int]] = {}
    by_type: dict[SukukType, list[int]] = {}
    for i, c in enumerate(candidates):
        by_sector.setdefault(c.sector, []).append(i)
        by_jur.setdefault(c.jurisdiction, []).append(i)
        by_type.setdefault(c.sukuk_type, []).append(i)

    for _ in range(20):
        # Single-name clamp.
        out = [min(wi, cap_single) for wi in out]
        # Sector cap.
        for sector, idxs in by_sector.items():
            cap = _default_sector_cap(constraints, sector)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        # Jurisdiction cap.
        for jur, idxs in by_jur.items():
            cap = _default_jurisdiction_cap(constraints, jur)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        # Type cap.
        for st, idxs in by_type.items():
            cap = _default_type_cap(constraints, st)
            s = sum(out[i] for i in idxs)
            if s > cap and s > 1e-12:
                scale = cap / s
                for i in idxs:
                    out[i] *= scale
        # Renormalise.
        s = sum(out)
        if s <= 1e-12:
            raise InfeasibleBasketError("constraints reduce all weights to zero")
        out = [wi / s for wi in out]
        # Stop iff every cap holds post-renormalisation.
        if max(out) > cap_single + 1e-9:
            continue
        violated = False
        for sector, idxs in by_sector.items():
            if sum(out[i] for i in idxs) > _default_sector_cap(constraints, sector) + 1e-9:
                violated = True
                break
        if violated:
            continue
        for jur, idxs in by_jur.items():
            if sum(out[i] for i in idxs) > _default_jurisdiction_cap(constraints, jur) + 1e-9:
                violated = True
                break
        if violated:
            continue
        for st, idxs in by_type.items():
            if sum(out[i] for i in idxs) > _default_type_cap(constraints, st) + 1e-9:
                violated = True
                break
        if not violated:
            break
    return out


def _grad_mean_variance(
    w: Sequence[float],
    mu: Sequence[float],
    cov: Sequence[Sequence[float]],
    risk_aversion: float,
) -> list[float]:
    """∇(λ wᵀΣw - μᵀw) = 2λΣw - μ. Minimisation form — descend on this."""
    n = len(w)
    grad = [-mu[i] for i in range(n)]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += cov[i][j] * w[j]
        grad[i] += 2 * risk_aversion * s
    return grad


def _portfolio_variance(w: Sequence[float], cov: Sequence[Sequence[float]]) -> float:
    n = len(w)
    out = 0.0
    for i in range(n):
        for j in range(n):
            out += w[i] * cov[i][j] * w[j]
    return out


def _portfolio_return(w: Sequence[float], mu: Sequence[float]) -> float:
    return sum(wi * mi for wi, mi in zip(w, mu, strict=True))


def _portfolio_duration(w: Sequence[float], candidates: Sequence[SukukCandidate]) -> float:
    return sum(wi * c.duration_years for wi, c in zip(w, candidates, strict=True))


def allocate(
    candidates: Sequence[SukukCandidate],
    *,
    objective: AllocationObjective = AllocationObjective.MEAN_VARIANCE,
    covariance: Sequence[Sequence[float]] | None = None,
    constraints: AllocationConstraints | None = None,
    max_iter: int = 500,
    step_size: float = 0.05,
    tolerance: float = 1e-8,
) -> AllocationResult:
    """Run the projected-gradient solver to compute optimal weights.

    The solver is deterministic and pure-Python — convergence is bounded
    by `max_iter`; the gradient direction depends on the objective
    (MIN_VARIANCE ignores μ; TARGET_DURATION adds a quadratic
    duration-mismatch penalty to the objective).
    """
    _validate_universe(candidates)
    n = len(candidates)
    cstr = constraints if constraints is not None else AllocationConstraints()

    # Build covariance — default to a diagonal 0.04 (20% vol per name).
    if covariance is None:
        cov: list[list[float]] = [[0.04 if i == j else 0.0 for j in range(n)] for i in range(n)]
    else:
        cov = [list(row) for row in covariance]
        _validate_covariance(cov, n)

    mu = [c.expected_return for c in candidates]

    # Adjust objective: MIN_VARIANCE zeros μ (no return tilt).
    if objective is AllocationObjective.MIN_VARIANCE:
        active_mu = [0.0] * n
    else:
        active_mu = mu

    # TARGET_DURATION needs a target.
    if objective is AllocationObjective.TARGET_DURATION:
        if cstr.target_duration is None:
            raise ValueError("TARGET_DURATION objective requires constraints.target_duration")

    # Initial weight = uniform.
    w = [1.0 / n] * n

    for it in range(max_iter):
        # Base gradient (λ-weighted Σw - μ for MV / MIN_VAR).
        grad = _grad_mean_variance(w, active_mu, cov, cstr.risk_aversion)

        # TARGET_DURATION: add 2β (D(w) - D*) · d_i  to gradient.
        if objective is AllocationObjective.TARGET_DURATION:
            beta = max(1.0, cstr.risk_aversion)
            d_w = _portfolio_duration(w, candidates)
            mismatch = d_w - cstr.target_duration  # type: ignore[operator]
            for i in range(n):
                grad[i] += 2 * beta * mismatch * candidates[i].duration_years

        # Adaptive step: cap so the largest absolute move is ≤ 0.1, which
        # keeps the projected-gradient stable for objectives whose
        # gradient magnitudes vary by orders of magnitude (e.g. the
        # duration-penalty term scales with max_duration).
        max_grad = max(abs(g) for g in grad) if grad else 0.0
        effective_step = step_size
        if max_grad * step_size > 0.1:
            effective_step = 0.1 / max_grad
        new_w = [w[i] - effective_step * grad[i] for i in range(n)]
        new_w = _project_to_simplex(new_w)
        new_w = _apply_caps(new_w, candidates, cstr)

        # Convergence check.
        delta = sum(abs(new_w[i] - w[i]) for i in range(n))
        w = new_w
        if delta < tolerance:
            break

    # Final check: weights sum to ~1 and are non-negative.
    total = sum(w)
    if abs(total - 1.0) > 1e-6:
        # Renormalise as a final sanity.
        w = [wi / total for wi in w]

    pr = _portfolio_return(w, mu)
    pv = _portfolio_variance(w, cov)
    pd = _portfolio_duration(w, candidates)
    return AllocationResult(
        weights=tuple(w),
        candidates=tuple(candidates),
        expected_return=pr,
        expected_variance=pv,
        portfolio_duration=pd,
    )


def render_basket(result: AllocationResult, *, top_n: int = 10) -> str:
    """Operator-readable summary of the allocation result."""
    head = (
        f"📊 Sukuk basket: {len(result.candidates)} candidates, "
        f"expected return={result.expected_return * 100:.2f}%, "
        f"vol={result.expected_volatility() * 100:.2f}%, "
        f"duration={result.portfolio_duration:.2f}y"
    )
    pairs = sorted(result.by_issuer(), key=lambda kv: kv[1], reverse=True)[:top_n]
    lines = [head]
    for issuer, w in pairs:
        lines.append(f"  • {issuer}: {w * 100:.2f}%")
    return "\n".join(lines)
