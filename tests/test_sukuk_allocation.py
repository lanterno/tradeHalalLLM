"""Tests for markets/sukuk_allocation.py — Round-5 Wave 3.C."""

from __future__ import annotations

import pytest

from halal_trader.halal.aaoifi_standard_17 import SukukType
from halal_trader.markets.sukuk_allocation import (
    AllocationConstraints,
    AllocationObjective,
    AllocationResult,
    SukukCandidate,
    allocate,
    render_basket,
)


def _cand(
    issuer: str = "GovOfMalaysia",
    sukuk_type: SukukType = SukukType.IJARA,
    sector: str = "sovereign",
    jurisdiction: str = "MY",
    duration: float = 5.0,
    er: float = 0.04,
) -> SukukCandidate:
    return SukukCandidate(
        issuer=issuer,
        sukuk_type=sukuk_type,
        sector=sector,
        jurisdiction=jurisdiction,
        duration_years=duration,
        expected_return=er,
    )


# --- SukukCandidate validation -------------------------------------------


def test_candidate_valid():
    c = _cand()
    assert c.issuer == "GovOfMalaysia"


def test_candidate_empty_issuer_rejected():
    with pytest.raises(ValueError):
        _cand(issuer="")


def test_candidate_non_tradable_rejected():
    with pytest.raises(ValueError):
        _cand(sukuk_type=SukukType.MURABAHA)
    with pytest.raises(ValueError):
        _cand(sukuk_type=SukukType.SALAM)


def test_candidate_empty_sector_rejected():
    with pytest.raises(ValueError):
        _cand(sector=" ")


def test_candidate_negative_duration_rejected():
    with pytest.raises(ValueError):
        _cand(duration=-1.0)


def test_candidate_unreasonable_return_rejected():
    with pytest.raises(ValueError):
        _cand(er=0.99)


def test_candidate_immutable():
    c = _cand()
    with pytest.raises(AttributeError):
        c.expected_return = 0.05  # type: ignore[misc]


# --- AllocationConstraints validation ------------------------------------


def test_constraints_default():
    cstr = AllocationConstraints()
    assert cstr.max_single_name == 0.30
    assert cstr.risk_aversion == 1.0


def test_constraints_invalid_single_name():
    with pytest.raises(ValueError):
        AllocationConstraints(max_single_name=0.0)
    with pytest.raises(ValueError):
        AllocationConstraints(max_single_name=1.5)


def test_constraints_invalid_sector_cap():
    with pytest.raises(ValueError):
        AllocationConstraints(sector_caps={"corp": 1.5})


def test_constraints_invalid_target_duration():
    with pytest.raises(ValueError):
        AllocationConstraints(target_duration=-1.0)


def test_constraints_invalid_risk_aversion():
    with pytest.raises(ValueError):
        AllocationConstraints(risk_aversion=-0.1)


# --- allocate basic shape ------------------------------------------------


def test_allocate_basic():
    cands = [_cand(issuer=f"Issuer{i}", er=0.03 + i * 0.005) for i in range(4)]
    res = allocate(cands)
    assert isinstance(res, AllocationResult)
    assert len(res.weights) == 4
    assert abs(sum(res.weights) - 1.0) < 1e-6
    for w in res.weights:
        assert w >= -1e-9


def test_allocate_empty_universe_rejected():
    with pytest.raises(ValueError):
        allocate([])


def test_allocate_universe_too_large_rejected():
    cands = [_cand(issuer=f"X{i}") for i in range(201)]
    with pytest.raises(ValueError):
        allocate(cands)


# --- Mean-variance objective ---------------------------------------------


def test_mean_variance_prefers_higher_return():
    """With diagonal cov + uniform vol, higher μ should get more weight."""
    cands = [
        _cand(issuer="A", er=0.02),
        _cand(issuer="B", er=0.06),
    ]
    res = allocate(
        cands,
        objective=AllocationObjective.MEAN_VARIANCE,
        constraints=AllocationConstraints(max_single_name=0.99, risk_aversion=0.5),
    )
    # B has higher expected return and the same risk → should dominate.
    a_w, b_w = res.weights
    assert b_w > a_w


def test_min_variance_ignores_returns():
    """MIN_VARIANCE on a diagonal cov should approach equal-weight."""
    cands = [
        _cand(issuer="A", er=0.02),
        _cand(issuer="B", er=0.20),
        _cand(issuer="C", er=0.04),
    ]
    res = allocate(
        cands,
        objective=AllocationObjective.MIN_VARIANCE,
        constraints=AllocationConstraints(max_single_name=0.99),
    )
    # Each weight should be close to 1/3.
    for w in res.weights:
        assert abs(w - 1 / 3) < 0.05


def test_target_duration_pulls_basket_to_target():
    cands = [
        _cand(issuer="Short", duration=1.0, er=0.03),
        _cand(issuer="Mid", duration=5.0, er=0.04),
        _cand(issuer="Long", duration=10.0, er=0.05),
    ]
    res_short = allocate(
        cands,
        objective=AllocationObjective.TARGET_DURATION,
        constraints=AllocationConstraints(
            max_single_name=0.99, target_duration=2.0, risk_aversion=2.0
        ),
    )
    res_long = allocate(
        cands,
        objective=AllocationObjective.TARGET_DURATION,
        constraints=AllocationConstraints(
            max_single_name=0.99, target_duration=8.0, risk_aversion=2.0
        ),
    )
    # Short-target basket has shorter duration than long-target basket.
    assert res_short.portfolio_duration < res_long.portfolio_duration


def test_target_duration_requires_target():
    cands = [_cand()]
    with pytest.raises(ValueError):
        allocate(
            cands,
            objective=AllocationObjective.TARGET_DURATION,
            constraints=AllocationConstraints(),
        )


# --- Constraints (caps) --------------------------------------------------


def test_single_name_cap_enforced():
    cands = [
        _cand(issuer="A", er=0.10),
        _cand(issuer="B", er=0.01),
    ]
    cstr = AllocationConstraints(max_single_name=0.50, risk_aversion=0.1)
    res = allocate(cands, constraints=cstr)
    for w in res.weights:
        assert w <= 0.50 + 1e-6


def test_sector_cap_enforced():
    cands = [
        _cand(issuer="A", sector="energy", er=0.10),
        _cand(issuer="B", sector="energy", er=0.10),
        _cand(issuer="C", sector="sovereign", er=0.02),
    ]
    cstr = AllocationConstraints(
        sector_caps={"energy": 0.30, "sovereign": 0.99},
        max_single_name=0.99,
        risk_aversion=0.1,
    )
    res = allocate(cands, constraints=cstr)
    energy_w = res.weights[0] + res.weights[1]
    assert energy_w <= 0.30 + 1e-6


def test_jurisdiction_cap_enforced():
    cands = [
        _cand(issuer="A", jurisdiction="SA", er=0.10),
        _cand(issuer="B", jurisdiction="SA", er=0.09),
        _cand(issuer="C", jurisdiction="MY", er=0.02),
    ]
    cstr = AllocationConstraints(
        jurisdiction_caps={"SA": 0.40, "MY": 0.99},
        max_single_name=0.99,
        risk_aversion=0.1,
    )
    res = allocate(cands, constraints=cstr)
    sa_w = res.weights[0] + res.weights[1]
    assert sa_w <= 0.40 + 1e-6


def test_type_cap_enforced():
    cands = [
        _cand(issuer="A", sukuk_type=SukukType.MUDARABAH, er=0.10),
        _cand(issuer="B", sukuk_type=SukukType.MUDARABAH, er=0.09),
        _cand(issuer="C", sukuk_type=SukukType.IJARA, er=0.02),
    ]
    cstr = AllocationConstraints(
        type_caps={SukukType.MUDARABAH: 0.20},
        max_single_name=0.99,
        risk_aversion=0.1,
    )
    res = allocate(cands, constraints=cstr)
    mud_w = res.weights[0] + res.weights[1]
    assert mud_w <= 0.20 + 1e-6


# --- Covariance handling -------------------------------------------------


def test_custom_covariance_accepted():
    cands = [_cand(issuer=f"X{i}") for i in range(3)]
    cov = [
        [0.04, 0.01, 0.0],
        [0.01, 0.04, 0.0],
        [0.0, 0.0, 0.04],
    ]
    res = allocate(cands, covariance=cov)
    assert abs(sum(res.weights) - 1.0) < 1e-6


def test_covariance_size_mismatch_rejected():
    cands = [_cand(issuer=f"X{i}") for i in range(3)]
    cov = [[0.04, 0.0], [0.0, 0.04]]  # 2x2 but 3 candidates
    with pytest.raises(ValueError):
        allocate(cands, covariance=cov)


def test_covariance_asymmetric_rejected():
    cands = [_cand(issuer=f"X{i}") for i in range(2)]
    cov = [[0.04, 0.01], [0.05, 0.04]]  # asymmetric
    with pytest.raises(ValueError):
        allocate(cands, covariance=cov)


def test_covariance_negative_diagonal_rejected():
    cands = [_cand(issuer=f"X{i}") for i in range(2)]
    cov = [[-0.01, 0.0], [0.0, 0.04]]
    with pytest.raises(ValueError):
        allocate(cands, covariance=cov)


# --- AllocationResult helpers --------------------------------------------


def test_result_volatility_helper():
    cands = [_cand()]
    res = allocate(cands)
    assert res.expected_volatility() >= 0


def test_result_by_issuer_drops_zero_weights():
    cands = [
        _cand(issuer="A", er=0.10),
        _cand(issuer="B", er=0.10),
        _cand(issuer="C", er=0.10),
    ]
    cstr = AllocationConstraints(sector_caps={"sovereign": 1.0}, max_single_name=0.99)
    res = allocate(cands, constraints=cstr)
    pairs = res.by_issuer()
    for _, w in pairs:
        assert w > 1e-9


# --- Determinism ---------------------------------------------------------


def test_allocate_deterministic():
    cands = [
        _cand(issuer="A", er=0.04),
        _cand(issuer="B", er=0.05),
        _cand(issuer="C", er=0.03),
    ]
    r1 = allocate(cands)
    r2 = allocate(cands)
    for w1, w2 in zip(r1.weights, r2.weights, strict=True):
        assert abs(w1 - w2) < 1e-9


# --- Render --------------------------------------------------------------


def test_render_basket_contains_summary():
    cands = [_cand(issuer="GovOfMalaysia"), _cand(issuer="GovOfSaudi")]
    res = allocate(cands)
    out = render_basket(res)
    assert "📊" in out
    assert "duration" in out.lower()


def test_render_basket_no_secret_leak():
    """Pin: render only mentions issuer + weight + portfolio summary."""
    cands = [_cand(issuer="GovOfMalaysia")]
    res = allocate(cands)
    out = render_basket(res)
    assert "covariance" not in out.lower()
    assert "gradient" not in out.lower()
