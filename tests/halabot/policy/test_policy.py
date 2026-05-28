"""Policy targets (gross normalization) + deltas (anti-churn, gates)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from halabot.belief.schema import BeliefState, ComplianceVerdict, Direction
from halabot.policy.policy import Policy
from halabot.policy.portfolio import ShadowPortfolio
from halabot.policy.sizing import PolicyConfig
from halabot.risk.engine import RiskState

CFG = PolicyConfig(
    conviction_entry_band=0.60,
    conviction_exit_band=0.45,
    max_weight_per_asset=0.20,
    max_gross_exposure=1.0,
    target_rebalance_threshold=0.05,
)
RISK = RiskState()


def _b(asset, conviction=0.9, *, direction=Direction.LONG_BIAS, status="halal", version=1):
    return BeliefState(
        asset=asset,
        direction=direction,
        conviction=conviction,
        halal=ComplianceVerdict(asset, status),  # type: ignore[arg-type]
        version=version,
    )


# ── targets: gross normalization (R-03) ──
def test_targets_normalize_when_gross_exceeds_cap():
    policy = Policy(CFG)
    beliefs = [_b(f"A{i}", conviction=1.0) for i in range(6)]  # 6 × 0.20 = 1.20 > 1.0
    targets = policy.targets(beliefs, ShadowPortfolio(), RISK)
    gross = sum(t.weight for t in targets)
    assert gross == pytest.approx(1.0)            # scaled down to the cap — no leverage
    assert all("normalized" in t.reason for t in targets)


def test_targets_not_normalized_under_cap():
    policy = Policy(CFG)
    beliefs = [_b("A", conviction=1.0), _b("B", conviction=1.0)]  # 0.40 gross
    targets = policy.targets(beliefs, ShadowPortfolio(), RISK)
    assert sum(t.weight for t in targets) == pytest.approx(0.40)


# ── deltas: anti-churn + gates ──
def test_delta_emitted_for_new_conviction():
    policy = Policy(CFG)
    b = _b("NVDA", conviction=0.9)
    targets = policy.targets([b], ShadowPortfolio(), RISK)
    props = policy.deltas(targets, ShadowPortfolio(), beliefs_by_asset={"NVDA": b}, risk=RISK)
    assert len(props) == 1 and props[0].side == "buy"


def test_no_delta_when_target_matches_current():
    policy = Policy(CFG)
    b = _b("NVDA", conviction=0.9)
    pf = ShadowPortfolio()
    targets = policy.targets([b], pf, RISK)
    pf.set_weight("NVDA", targets[0].weight)  # already at target
    props = policy.deltas(targets, pf, beliefs_by_asset={"NVDA": b}, risk=RISK)
    assert props == []  # stable belief → NO TRADE (anti-churn)


def test_halal_gate_blocks_buy():
    policy = Policy(CFG)
    b = _b("NVDA", conviction=0.9, status="not_halal")
    targets = policy.targets([b], ShadowPortfolio(), RISK)
    props = policy.deltas(targets, ShadowPortfolio(), beliefs_by_asset={"NVDA": b}, risk=RISK)
    assert props == []  # non-halal → no buy (INV-7)


# ── INV-7 entry freshness: a stale positive verdict fails closed ──
def _b_screened(asset, *, screened_at, conviction=0.9):
    return BeliefState(
        asset=asset,
        direction=Direction.LONG_BIAS,
        conviction=conviction,
        halal=ComplianceVerdict(asset, "halal", screened_at=screened_at),
        version=1,
    )


def test_fresh_verdict_allows_buy():
    policy = Policy(CFG)
    now = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
    b = _b_screened("NVDA", screened_at=now - timedelta(hours=1))
    targets = policy.targets([b], ShadowPortfolio(), RISK)
    props = policy.deltas(
        targets,
        ShadowPortfolio(),
        beliefs_by_asset={"NVDA": b},
        risk=RISK,
        now=now,
        compliance_ttl=timedelta(hours=24),
    )
    assert len(props) == 1 and props[0].side == "buy"


def test_stale_verdict_blocks_buy():
    policy = Policy(CFG)
    now = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
    b = _b_screened("NVDA", screened_at=now - timedelta(hours=48))  # older than TTL
    targets = policy.targets([b], ShadowPortfolio(), RISK)
    props = policy.deltas(
        targets,
        ShadowPortfolio(),
        beliefs_by_asset={"NVDA": b},
        risk=RISK,
        now=now,
        compliance_ttl=timedelta(hours=24),
    )
    assert props == []  # stale verdict → fail-closed (INV-7 entry freshness)


def test_missing_screened_at_blocks_buy_under_ttl():
    policy = Policy(CFG)
    now = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
    b = _b("NVDA", conviction=0.9, status="halal")  # screened_at=None
    targets = policy.targets([b], ShadowPortfolio(), RISK)
    props = policy.deltas(
        targets,
        ShadowPortfolio(),
        beliefs_by_asset={"NVDA": b},
        risk=RISK,
        now=now,
        compliance_ttl=timedelta(hours=24),
    )
    assert props == []  # no screened_at + TTL enforced → not fresh


def test_risk_halt_blocks_buy_but_allows_sell():
    policy = Policy(CFG)
    halted = RiskState(halted=True, reason="drawdown")
    # buy blocked
    b = _b("NVDA", conviction=0.9)
    buy = policy.deltas(
        policy.targets([b], ShadowPortfolio(), halted),
        ShadowPortfolio(), beliefs_by_asset={"NVDA": b}, risk=halted,
    )
    assert buy == []
    # sell allowed: held position, belief turned neutral → target 0 → exit
    flat = _b("NVDA", conviction=0.0, direction=Direction.NEUTRAL)
    pf = ShadowPortfolio()
    pf.set_weight("NVDA", 0.15)
    sell = policy.deltas(
        policy.targets([flat], pf, halted),
        pf, beliefs_by_asset={"NVDA": flat}, risk=halted,
    )
    assert len(sell) == 1 and sell[0].side == "sell"


def test_open_order_suppresses_duplicate():
    policy = Policy(CFG)
    b = _b("NVDA", conviction=0.9)

    class _PendingPortfolio(ShadowPortfolio):
        def has_open_order(self, asset: str) -> bool:
            return True

    pf = _PendingPortfolio()
    props = policy.deltas(
        policy.targets([b], pf, RISK), pf, beliefs_by_asset={"NVDA": b}, risk=RISK
    )
    assert props == []  # an order already working → no duplicate (R-14)


def test_kill_switch_blocks_buy():
    policy = Policy(CFG)
    b = _b("NVDA", conviction=0.9)
    props = policy.deltas(
        policy.targets([b], ShadowPortfolio(), RISK),
        ShadowPortfolio(), beliefs_by_asset={"NVDA": b}, risk=RISK, kill_switch=True,
    )
    assert props == []
