"""Tests for `halal_trader.ops.e2e_scenarios` (Wave 8.A).

Covers: scenario catalogue coverage, freshness ladder boundaries,
last-passed lookup, aggregate status, render no-secret contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from halal_trader.ops.e2e_scenarios import (
    DEFAULT_POLICY,
    FreshnessLevel,
    RequiredBroker,
    RunOutcome,
    RunRecord,
    Scenario,
    ScenarioKind,
    ScenarioPolicy,
    all_scenarios,
    build_status,
    freshness_for,
    last_passed_run,
    render_scenario,
    render_status,
    scenario,
    scenarios_for_broker,
    scenarios_for_kind,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --------------------------- Enum string pins --------------------------------


def test_scenario_kind_string_values_pinned() -> None:
    assert ScenarioKind.CYCLE.value == "cycle"
    assert ScenarioKind.ORDER.value == "order"
    assert ScenarioKind.HALT.value == "halt"
    assert ScenarioKind.RECONCILE.value == "reconcile"
    assert ScenarioKind.WEBSOCKET.value == "websocket"
    assert ScenarioKind.FAILOVER.value == "failover"


def test_required_broker_string_values_pinned() -> None:
    assert RequiredBroker.NONE.value == "none"
    assert RequiredBroker.BINANCE_TESTNET.value == "binance_testnet"
    assert RequiredBroker.ALPACA_PAPER.value == "alpaca_paper"


def test_run_outcome_string_values_pinned() -> None:
    assert RunOutcome.PASSED.value == "passed"
    assert RunOutcome.FAILED.value == "failed"
    assert RunOutcome.SKIPPED.value == "skipped"


def test_freshness_level_string_values_pinned() -> None:
    assert FreshnessLevel.FRESH.value == "fresh"
    assert FreshnessLevel.STALE.value == "stale"
    assert FreshnessLevel.CRITICAL.value == "critical"
    assert FreshnessLevel.NEVER_RUN.value == "never_run"


# --------------------------- ScenarioPolicy ----------------------------------


def test_default_policy_thresholds() -> None:
    assert DEFAULT_POLICY.fresh_threshold == timedelta(days=7)
    assert DEFAULT_POLICY.stale_threshold == timedelta(days=14)
    assert DEFAULT_POLICY.critical_threshold == timedelta(days=28)


def test_policy_rejects_zero_fresh_threshold() -> None:
    with pytest.raises(ValueError, match="fresh_threshold"):
        ScenarioPolicy(fresh_threshold=timedelta(0))


def test_policy_rejects_stale_below_fresh() -> None:
    with pytest.raises(ValueError, match="stale_threshold"):
        ScenarioPolicy(
            fresh_threshold=timedelta(days=10),
            stale_threshold=timedelta(days=5),
        )


def test_policy_rejects_critical_below_stale() -> None:
    with pytest.raises(ValueError, match="critical_threshold"):
        ScenarioPolicy(
            fresh_threshold=timedelta(days=1),
            stale_threshold=timedelta(days=2),
            critical_threshold=timedelta(days=2),
        )


def test_policy_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_POLICY.fresh_threshold = timedelta(days=1)  # type: ignore[misc]


# --------------------------- Scenario validation -----------------------------


def _scenario(**overrides: object) -> Scenario:
    base: dict[str, object] = {
        "scenario_id": "test_s",
        "kind": ScenarioKind.CYCLE,
        "required_broker": RequiredBroker.NONE,
        "description": "test description",
        "expected_outcome": "passes cleanly",
    }
    base.update(overrides)
    return Scenario(**base)  # type: ignore[arg-type]


def test_scenario_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        _scenario(scenario_id="")


def test_scenario_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        _scenario(description="")


def test_scenario_rejects_empty_expected_outcome() -> None:
    """Pin: every scenario must document its success criteria."""

    with pytest.raises(ValueError, match="expected_outcome"):
        _scenario(expected_outcome="")


def test_scenario_is_frozen() -> None:
    s = _scenario()
    with pytest.raises(FrozenInstanceError):
        s.scenario_id = "other"  # type: ignore[misc]


# --------------------------- RunRecord ---------------------------------------


def test_run_record_rejects_empty_scenario_id() -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        RunRecord(
            scenario_id="",
            outcome=RunOutcome.PASSED,
            decided_at=T0,
        )


def test_run_record_rejects_naive_decided_at() -> None:
    with pytest.raises(ValueError, match="decided_at"):
        RunRecord(
            scenario_id="s",
            outcome=RunOutcome.PASSED,
            decided_at=datetime(2026, 5, 1),
        )


def test_run_record_is_frozen() -> None:
    r = RunRecord(scenario_id="s", outcome=RunOutcome.PASSED, decided_at=T0)
    with pytest.raises(FrozenInstanceError):
        r.outcome = RunOutcome.FAILED  # type: ignore[misc]


# --------------------------- catalogue coverage ------------------------------


def test_catalogue_has_at_least_one_scenario_per_kind() -> None:
    """Pin: every ScenarioKind has at least one seed scenario."""

    seen_kinds = {s.kind for s in all_scenarios()}
    for kind in ScenarioKind:
        assert kind in seen_kinds, kind


def test_catalogue_has_binance_and_alpaca_scenarios() -> None:
    """Pin: both roadmap-named brokers have testnet/paper scenarios."""

    binance = scenarios_for_broker(RequiredBroker.BINANCE_TESTNET)
    alpaca = scenarios_for_broker(RequiredBroker.ALPACA_PAPER)
    assert len(binance) >= 1
    assert len(alpaca) >= 1


def test_catalogue_has_full_cycle_scenarios() -> None:
    """Pin: full cycle scenarios for both brokers."""

    cycles = scenarios_for_kind(ScenarioKind.CYCLE)
    cycle_ids = {s.scenario_id for s in cycles}
    assert "binance_full_cycle" in cycle_ids
    assert "alpaca_full_cycle" in cycle_ids


def test_catalogue_has_halt_scenario() -> None:
    halts = scenarios_for_kind(ScenarioKind.HALT)
    assert any(s.scenario_id == "halt_then_resume" for s in halts)


def test_catalogue_has_reconcile_scenario() -> None:
    """Pin: reconcile scenario covers the local-vs-broker drift case."""

    rec = scenarios_for_kind(ScenarioKind.RECONCILE)
    assert len(rec) >= 1
    assert any("reconc" in s.scenario_id.lower() for s in rec)


def test_scenarios_for_kind_returns_only_matching() -> None:
    cycles = scenarios_for_kind(ScenarioKind.CYCLE)
    for s in cycles:
        assert s.kind is ScenarioKind.CYCLE


def test_scenarios_for_broker_returns_only_matching() -> None:
    binance = scenarios_for_broker(RequiredBroker.BINANCE_TESTNET)
    for s in binance:
        assert s.required_broker is RequiredBroker.BINANCE_TESTNET


def test_scenario_lookup_by_id() -> None:
    s = scenario("binance_full_cycle")
    assert s.scenario_id == "binance_full_cycle"


def test_scenario_lookup_unknown_raises() -> None:
    with pytest.raises(KeyError):
        scenario("nonexistent")


def test_all_scenarios_canonical_order() -> None:
    """Pin: deterministic order (sorted by scenario_id)."""

    scenarios = all_scenarios()
    ids = [s.scenario_id for s in scenarios]
    assert ids == sorted(ids)


# --------------------------- freshness_for -----------------------------------


def test_freshness_never_run_when_none() -> None:
    assert freshness_for(None, now=T0) is FreshnessLevel.NEVER_RUN


def test_freshness_fresh_when_recent() -> None:
    last = T0 - timedelta(days=3)
    assert freshness_for(last, now=T0) is FreshnessLevel.FRESH


def test_freshness_at_7_day_boundary_is_stale() -> None:
    """Pin: 7d exactly hits inclusive stale boundary."""

    last = T0 - timedelta(days=7)
    assert freshness_for(last, now=T0) is FreshnessLevel.STALE


def test_freshness_just_below_7_days_is_fresh() -> None:
    last = T0 - timedelta(days=6, hours=23)
    assert freshness_for(last, now=T0) is FreshnessLevel.FRESH


def test_freshness_at_28_day_boundary_is_critical() -> None:
    """Pin: 28d exactly hits critical boundary inclusive."""

    last = T0 - timedelta(days=28)
    assert freshness_for(last, now=T0) is FreshnessLevel.CRITICAL


def test_freshness_just_below_28_days_is_stale() -> None:
    last = T0 - timedelta(days=27, hours=23)
    assert freshness_for(last, now=T0) is FreshnessLevel.STALE


def test_freshness_well_past_28_days_is_critical() -> None:
    last = T0 - timedelta(days=60)
    assert freshness_for(last, now=T0) is FreshnessLevel.CRITICAL


def test_freshness_custom_policy() -> None:
    """Strict policy: 1d / 2d / 4d."""

    strict = ScenarioPolicy(
        fresh_threshold=timedelta(days=1),
        stale_threshold=timedelta(days=2),
        critical_threshold=timedelta(days=4),
    )
    last = T0 - timedelta(days=3)
    assert freshness_for(last, now=T0, policy=strict) is FreshnessLevel.STALE


def test_freshness_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        freshness_for(None, now=datetime(2026, 5, 1))


def test_freshness_rejects_naive_last_passed_at() -> None:
    with pytest.raises(ValueError, match="last_passed_at"):
        freshness_for(datetime(2026, 4, 1), now=T0)


# --------------------------- last_passed_run ---------------------------------


def test_last_passed_returns_most_recent_passed() -> None:
    records = [
        RunRecord(scenario_id="s1", outcome=RunOutcome.PASSED, decided_at=T0 - timedelta(days=10)),
        RunRecord(scenario_id="s1", outcome=RunOutcome.FAILED, decided_at=T0 - timedelta(days=5)),
        RunRecord(scenario_id="s1", outcome=RunOutcome.PASSED, decided_at=T0 - timedelta(days=2)),
    ]
    last = last_passed_run(records, "s1")
    assert last is not None
    assert last.decided_at == T0 - timedelta(days=2)


def test_last_passed_ignores_failed() -> None:
    """Pin: SKIPPED and FAILED don't reset the freshness clock."""

    records = [
        RunRecord(scenario_id="s1", outcome=RunOutcome.PASSED, decided_at=T0 - timedelta(days=20)),
        RunRecord(scenario_id="s1", outcome=RunOutcome.FAILED, decided_at=T0 - timedelta(days=5)),
        RunRecord(scenario_id="s1", outcome=RunOutcome.SKIPPED, decided_at=T0 - timedelta(days=1)),
    ]
    last = last_passed_run(records, "s1")
    assert last is not None
    assert last.decided_at == T0 - timedelta(days=20)


def test_last_passed_returns_none_when_never_passed() -> None:
    records = [
        RunRecord(scenario_id="s1", outcome=RunOutcome.FAILED, decided_at=T0),
    ]
    assert last_passed_run(records, "s1") is None


def test_last_passed_filters_by_scenario_id() -> None:
    records = [
        RunRecord(scenario_id="s1", outcome=RunOutcome.PASSED, decided_at=T0 - timedelta(days=1)),
        RunRecord(scenario_id="s2", outcome=RunOutcome.PASSED, decided_at=T0 - timedelta(hours=1)),
    ]
    last = last_passed_run(records, "s1")
    assert last is not None
    assert last.scenario_id == "s1"


def test_last_passed_empty_records() -> None:
    assert last_passed_run([], "s1") is None


# --------------------------- build_status ------------------------------------


def test_build_status_empty() -> None:
    status = build_status([], records=[], now=T0)
    assert status.total_scenarios == 0
    assert status.fresh_count == 0
    assert status.stale_count == 0
    assert status.critical_count == 0
    assert status.never_run_count == 0


def test_build_status_all_never_run() -> None:
    scenarios = list(all_scenarios())
    status = build_status(scenarios, records=[], now=T0)
    assert status.never_run_count == len(scenarios)
    assert status.fresh_count == 0


def test_build_status_mixed_freshness() -> None:
    scenarios_list = list(all_scenarios())[:4]
    records = [
        RunRecord(
            scenario_id=scenarios_list[0].scenario_id,
            outcome=RunOutcome.PASSED,
            decided_at=T0 - timedelta(days=2),  # FRESH
        ),
        RunRecord(
            scenario_id=scenarios_list[1].scenario_id,
            outcome=RunOutcome.PASSED,
            decided_at=T0 - timedelta(days=10),  # STALE
        ),
        RunRecord(
            scenario_id=scenarios_list[2].scenario_id,
            outcome=RunOutcome.PASSED,
            decided_at=T0 - timedelta(days=40),  # CRITICAL
        ),
        # scenarios_list[3] never run
    ]
    status = build_status(scenarios_list, records=records, now=T0)
    assert status.total_scenarios == 4
    assert status.fresh_count == 1
    assert status.stale_count == 1
    assert status.critical_count == 1
    assert status.never_run_count == 1


def test_build_status_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now"):
        build_status([], records=[], now=datetime(2026, 5, 1))


def test_build_status_is_frozen() -> None:
    status = build_status([], records=[], now=T0)
    with pytest.raises(FrozenInstanceError):
        status.total_scenarios = 99  # type: ignore[misc]


def test_build_status_is_deterministic() -> None:
    a = build_status(all_scenarios(), records=[], now=T0)
    b = build_status(all_scenarios(), records=[], now=T0)
    assert a == b


# --------------------------- render ------------------------------------------


def test_render_scenario_includes_id_and_kind() -> None:
    s = scenario("binance_full_cycle")
    out = render_scenario(s)
    assert "binance_full_cycle" in out
    assert "cycle" in out


def test_render_scenario_includes_kind_emoji() -> None:
    s = scenario("halt_then_resume")
    out = render_scenario(s)
    # halt → 🛑
    assert "🛑" in out


def test_render_scenario_includes_broker_and_description() -> None:
    s = scenario("alpaca_full_cycle")
    out = render_scenario(s)
    assert "alpaca_paper" in out
    assert "Alpaca" in out


def test_render_scenario_includes_expected_outcome() -> None:
    s = scenario("binance_full_cycle")
    out = render_scenario(s)
    assert "expected:" in out
    assert "halt" in out.lower()


def test_render_scenario_no_secret_leak() -> None:
    """Pin: catalogue is pure metadata; structural no-secret."""

    for s in all_scenarios():
        out = render_scenario(s)
        assert "api_key" not in out.lower()
        assert "cus_" not in out.lower()
        assert "bearer" not in out.lower()


def test_render_status_includes_counts() -> None:
    scenarios_list = list(all_scenarios())[:2]
    status = build_status(scenarios_list, records=[], now=T0)
    out = render_status(status)
    assert "total: 2" in out
    assert "never run: 2" in out


def test_render_status_includes_freshness_emoji() -> None:
    status = build_status(all_scenarios(), records=[], now=T0)
    out = render_status(status)
    assert "✅" in out
    assert "⚠️" in out
    assert "🔴" in out
    assert "❓" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_nightly_run_lifecycle() -> None:
    """Real-world: nightly CI runs binance_full_cycle, records pass, freshness goes FRESH."""

    s = scenario("binance_full_cycle")
    records: list[RunRecord] = []

    # Day 1: scenario passes
    records.append(
        RunRecord(
            scenario_id=s.scenario_id,
            outcome=RunOutcome.PASSED,
            decided_at=T0,
        )
    )
    last = last_passed_run(records, s.scenario_id)
    assert last is not None
    level = freshness_for(last.decided_at, now=T0 + timedelta(hours=1))
    assert level is FreshnessLevel.FRESH

    # Day 8: scenario hasn't run, freshness goes STALE
    level = freshness_for(last.decided_at, now=T0 + timedelta(days=8))
    assert level is FreshnessLevel.STALE

    # Day 30: scenario hasn't run, freshness goes CRITICAL
    level = freshness_for(last.decided_at, now=T0 + timedelta(days=30))
    assert level is FreshnessLevel.CRITICAL


def test_e2e_skipped_doesnt_help_freshness() -> None:
    """Pin: SKIPPED runs don't reset the staleness clock.

    A scenario that's been SKIPPED for the last 14 days because
    Binance testnet was down should still register as STALE/CRITICAL
    because the actual test hasn't validated anything.
    """

    records = [
        RunRecord(
            scenario_id="s",
            outcome=RunOutcome.PASSED,
            decided_at=T0 - timedelta(days=30),
        ),
        # 5 SKIPPED runs in last 5 days
        *[
            RunRecord(
                scenario_id="s",
                outcome=RunOutcome.SKIPPED,
                decided_at=T0 - timedelta(days=d),
            )
            for d in range(5)
        ],
    ]
    last = last_passed_run(records, "s")
    assert last is not None
    level = freshness_for(last.decided_at, now=T0)
    # Only the PASSED record counts; 30d ago → CRITICAL
    assert level is FreshnessLevel.CRITICAL
