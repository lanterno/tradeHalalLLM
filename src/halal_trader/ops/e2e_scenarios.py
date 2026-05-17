"""End-to-end integration test scenario catalogue.

The roadmap pins Wave 8.A: "Today's integration tests stub the
broker. Add a nightly CI lane that hits real testnet endpoints
(Binance, Alpaca), runs full cycles, asserts no halt-trips, no
rejected orders, no missed fills. Catches integration drift the
unit tests can't." This module is the **pure-Python catalogue +
last-run tracker** the nightly CI lane consults to know what
scenarios to run, what outcomes to assert, and which scenarios
have gone stale and need re-running.

Picked a focused catalogue over a "single big nightly script"
approach because (a) different scenarios have different broker
dependencies (a Binance testnet scenario doesn't need Alpaca,
and vice-versa) — pinning the dependency in the scenario lets
the CI lane skip scenarios whose broker is currently unreachable
rather than fail the whole lane, (b) staleness tracking (a
scenario hasn't passed in 7 days = stale; 14 days = critical
drift risk) is a pure function of last_passed_at — surfacing it
deterministically lets the dashboard render "scenario X is
overdue" without re-deriving the logic, (c) the scenario kind
enum (CYCLE / ORDER / HALT / RECONCILE / WEBSOCKET / FAILOVER)
groups scenarios for routing — the operator dashboard can show
"all halt-related scenarios passing" as a single tile.

Pinned semantics:
- **Closed-set ScenarioKind enum.** Adding a kind is a code
  review change so the dashboard groupings can't drift.
- **Required broker is part of the scenario.** A scenario tagged
  `BINANCE_TESTNET` is skipped (not failed) when Binance testnet
  is unreachable; a scenario tagged `ALPACA_PAPER` similarly.
- **Staleness ladder: 7d fresh, 14d stale, 28d critical.** Both
  boundary days inclusive (>=); operator-tunable via
  `ScenarioPolicy`.
- **Outcome enum: PASSED / FAILED / SKIPPED.** SKIPPED is for
  broker-unreachable; doesn't reset the freshness clock since
  the scenario didn't actually validate anything.
- **Render output never includes broker API responses, account
  balances, or order IDs.** Mirrors the no-secret patterns of
  upstream waves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ScenarioKind(str, Enum):
    """Closed-set scenario categories.

    Pinned string values for JSON / DB stability. Adding a kind
    is a code review change.
    """

    CYCLE = "cycle"  # full cycle run end-to-end
    ORDER = "order"  # place / cancel / fill flow
    HALT = "halt"  # kill-switch + post-halt recovery
    RECONCILE = "reconcile"  # local-vs-broker position sync
    WEBSOCKET = "websocket"  # real-time stream connectivity
    FAILOVER = "failover"  # broker error → fallback path


class RequiredBroker(str, Enum):
    """Broker the scenario requires to run.

    `NONE` for broker-agnostic scenarios (e.g. local replay).
    `BINANCE_TESTNET` and `ALPACA_PAPER` for the two roadmap-pinned
    sandboxes.
    """

    NONE = "none"
    BINANCE_TESTNET = "binance_testnet"
    ALPACA_PAPER = "alpaca_paper"


class RunOutcome(str, Enum):
    """Per-run outcome.

    Pinned string values. SKIPPED is for "broker unreachable" —
    distinct from FAILED so the staleness clock doesn't reset.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


_DEFAULT_FRESH_THRESHOLD = timedelta(days=7)
_DEFAULT_STALE_THRESHOLD = timedelta(days=14)
_DEFAULT_CRITICAL_THRESHOLD = timedelta(days=28)


@dataclass(frozen=True)
class ScenarioPolicy:
    """Operator-tunable staleness thresholds."""

    fresh_threshold: timedelta = _DEFAULT_FRESH_THRESHOLD
    stale_threshold: timedelta = _DEFAULT_STALE_THRESHOLD
    critical_threshold: timedelta = _DEFAULT_CRITICAL_THRESHOLD

    def __post_init__(self) -> None:
        if self.fresh_threshold <= timedelta(0):
            raise ValueError("fresh_threshold must be positive")
        if self.stale_threshold <= self.fresh_threshold:
            raise ValueError("stale_threshold must exceed fresh_threshold")
        if self.critical_threshold <= self.stale_threshold:
            raise ValueError("critical_threshold must exceed stale_threshold")


DEFAULT_POLICY = ScenarioPolicy()


@dataclass(frozen=True)
class Scenario:
    """One E2E test scenario.

    `expected_outcome` is the documented expected pass behaviour
    (e.g. "cycle completes with at least 1 order placed and no
    halt trip"). Pinned non-empty so a contributor adding a new
    scenario must document the success criteria.
    """

    scenario_id: str
    kind: ScenarioKind
    required_broker: RequiredBroker
    description: str
    expected_outcome: str

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        if not self.expected_outcome or not self.expected_outcome.strip():
            raise ValueError("expected_outcome must be non-empty")


@dataclass(frozen=True)
class RunRecord:
    """Audit row for one scenario run."""

    scenario_id: str
    outcome: RunOutcome
    decided_at: datetime
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")


# Canonical seed catalogue. Operators extend via the registry +
# composition pattern (the actual broker-touching test bodies
# live in tests/integration/, this is just the index).
_SEED_SCENARIOS: dict[str, Scenario] = {
    "binance_full_cycle": Scenario(
        scenario_id="binance_full_cycle",
        kind=ScenarioKind.CYCLE,
        required_broker=RequiredBroker.BINANCE_TESTNET,
        description="Full crypto cycle on Binance testnet with halal universe",
        expected_outcome=(
            "cycle completes within asyncio.wait_for(interval*2); "
            "no halt trip; no rejected orders; at least one indicator snapshot recorded"
        ),
    ),
    "binance_order_lifecycle": Scenario(
        scenario_id="binance_order_lifecycle",
        kind=ScenarioKind.ORDER,
        required_broker=RequiredBroker.BINANCE_TESTNET,
        description=(
            "Place market order on testnet, observe fill, "
            "record submitted/filled timestamps"
        ),
        expected_outcome=(
            "submitted_at and filled_at populated; filled_quantity matches "
            "ordered quantity; no -1013 or -2010 rejections"
        ),
    ),
    "binance_websocket_stream": Scenario(
        scenario_id="binance_websocket_stream",
        kind=ScenarioKind.WEBSOCKET,
        required_broker=RequiredBroker.BINANCE_TESTNET,
        description="Connect WebSocket to Binance testnet, receive 100+ ticks for BTCUSDT",
        expected_outcome="100+ tick events received within 60s without disconnect",
    ),
    "alpaca_full_cycle": Scenario(
        scenario_id="alpaca_full_cycle",
        kind=ScenarioKind.CYCLE,
        required_broker=RequiredBroker.ALPACA_PAPER,
        description="Full stocks cycle on Alpaca paper during market hours",
        expected_outcome=(
            "cycle runs APScheduler-driven during market hours; no halt trip; "
            "no rejected orders; orders filled within 5s of placement"
        ),
    ),
    "alpaca_order_lifecycle": Scenario(
        scenario_id="alpaca_order_lifecycle",
        kind=ScenarioKind.ORDER,
        required_broker=RequiredBroker.ALPACA_PAPER,
        description="Place limit order on Alpaca paper, observe fill, confirm reconciliation",
        expected_outcome=(
            "submitted_at populated immediately; filled_at populated by poll loop; "
            "core/reconcile.py aggregates correctly"
        ),
    ),
    "halt_then_resume": Scenario(
        scenario_id="halt_then_resume",
        kind=ScenarioKind.HALT,
        required_broker=RequiredBroker.NONE,
        description="Engage kill-switch, verify cycle refuses new entries, resume cleanly",
        expected_outcome=(
            "is_halted returns True after halt; cycle.run_cycle short-circuits; "
            "monitor still enforces SL/TP; resume restores normal cycle flow"
        ),
    ),
    "broker_5xx_failover": Scenario(
        scenario_id="broker_5xx_failover",
        kind=ScenarioKind.FAILOVER,
        required_broker=RequiredBroker.NONE,
        description="Simulate broker 5xx during cycle; verify circuit breaker engagement",
        expected_outcome=(
            "circuit breaker trips per Wave 8.B chaos engine; cycle gracefully "
            "skips affected pair; AlertSink fires once per error_type"
        ),
    ),
    "reconciliation_drift": Scenario(
        scenario_id="reconciliation_drift",
        kind=ScenarioKind.RECONCILE,
        required_broker=RequiredBroker.NONE,
        description="Inject local position state mismatch; verify reconcile detects it",
        expected_outcome=(
            "core/reconcile.py compares local positions vs broker; "
            "discrepancy logged with cycle_id and surfaced via AlertSink"
        ),
    ),
}


def all_scenarios() -> tuple[Scenario, ...]:
    """Return all seed scenarios in canonical order."""

    return tuple(_SEED_SCENARIOS[k] for k in sorted(_SEED_SCENARIOS.keys()))


def scenarios_for_kind(kind: ScenarioKind) -> tuple[Scenario, ...]:
    return tuple(s for s in all_scenarios() if s.kind is kind)


def scenarios_for_broker(broker: RequiredBroker) -> tuple[Scenario, ...]:
    return tuple(s for s in all_scenarios() if s.required_broker is broker)


def scenario(scenario_id: str) -> Scenario:
    """Return the seed scenario for the given id."""

    return _SEED_SCENARIOS[scenario_id]


class FreshnessLevel(str, Enum):
    """Computed freshness tier for a scenario run.

    `NEVER_RUN` is for scenarios with no PASSED record.
    """

    FRESH = "fresh"
    STALE = "stale"
    CRITICAL = "critical"
    NEVER_RUN = "never_run"


def freshness_for(
    last_passed_at: datetime | None,
    *,
    now: datetime,
    policy: ScenarioPolicy = DEFAULT_POLICY,
) -> FreshnessLevel:
    """Classify the freshness of a scenario based on last PASSED time.

    Boundaries inclusive (>=). The thresholds compose as a ladder:
    `now - last_passed >= critical_threshold` → CRITICAL;
    `>= stale_threshold` → STALE;
    `>= fresh_threshold` → STALE (still old enough to flag);
    otherwise → FRESH.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if last_passed_at is None:
        return FreshnessLevel.NEVER_RUN
    if last_passed_at.tzinfo is None:
        raise ValueError("last_passed_at must be timezone-aware when set")

    age = now - last_passed_at
    if age >= policy.critical_threshold:
        return FreshnessLevel.CRITICAL
    if age >= policy.fresh_threshold:
        return FreshnessLevel.STALE
    return FreshnessLevel.FRESH


def last_passed_run(records: Iterable[RunRecord], scenario_id: str) -> RunRecord | None:
    """Return the most recent PASSED RunRecord for a scenario, or None."""

    matches = [
        r for r in records if r.scenario_id == scenario_id and r.outcome is RunOutcome.PASSED
    ]
    if not matches:
        return None
    return max(matches, key=lambda r: r.decided_at)


@dataclass(frozen=True)
class CatalogueStatus:
    """Aggregate status across the catalogue at a point in time."""

    generated_at: datetime
    total_scenarios: int
    fresh_count: int
    stale_count: int
    critical_count: int
    never_run_count: int

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")


def build_status(
    scenarios: Iterable[Scenario],
    *,
    records: Iterable[RunRecord],
    now: datetime,
    policy: ScenarioPolicy = DEFAULT_POLICY,
) -> CatalogueStatus:
    """Aggregate freshness across a scenario list."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    scenario_list = list(scenarios)
    record_list = list(records)
    counts: dict[FreshnessLevel, int] = {
        FreshnessLevel.FRESH: 0,
        FreshnessLevel.STALE: 0,
        FreshnessLevel.CRITICAL: 0,
        FreshnessLevel.NEVER_RUN: 0,
    }
    for scenario_obj in scenario_list:
        last = last_passed_run(record_list, scenario_obj.scenario_id)
        last_at = last.decided_at if last is not None else None
        level = freshness_for(last_at, now=now, policy=policy)
        counts[level] += 1

    return CatalogueStatus(
        generated_at=now,
        total_scenarios=len(scenario_list),
        fresh_count=counts[FreshnessLevel.FRESH],
        stale_count=counts[FreshnessLevel.STALE],
        critical_count=counts[FreshnessLevel.CRITICAL],
        never_run_count=counts[FreshnessLevel.NEVER_RUN],
    )


_KIND_EMOJI: dict[ScenarioKind, str] = {
    ScenarioKind.CYCLE: "🔄",
    ScenarioKind.ORDER: "📋",
    ScenarioKind.HALT: "🛑",
    ScenarioKind.RECONCILE: "⚖️",
    ScenarioKind.WEBSOCKET: "📡",
    ScenarioKind.FAILOVER: "🔁",
}


_FRESHNESS_EMOJI: dict[FreshnessLevel, str] = {
    FreshnessLevel.FRESH: "✅",
    FreshnessLevel.STALE: "⚠️",
    FreshnessLevel.CRITICAL: "🔴",
    FreshnessLevel.NEVER_RUN: "❓",
}


def render_scenario(scenario_obj: Scenario) -> str:
    """Format a scenario for ops display.

    No-secret-leak: catalogue is pure metadata; structural.
    """

    emoji = _KIND_EMOJI[scenario_obj.kind]
    return (
        f"{emoji} {scenario_obj.scenario_id} ({scenario_obj.kind.value})\n"
        f"  broker: {scenario_obj.required_broker.value}\n"
        f"  desc: {scenario_obj.description}\n"
        f"  expected: {scenario_obj.expected_outcome}"
    )


def render_status(status: CatalogueStatus) -> str:
    """Format the aggregate status."""

    return (
        f"📊 E2E catalogue @ {status.generated_at.isoformat()}\n"
        f"  total: {status.total_scenarios}\n"
        f"  ✅ fresh: {status.fresh_count}\n"
        f"  ⚠️ stale: {status.stale_count}\n"
        f"  🔴 critical: {status.critical_count}\n"
        f"  ❓ never run: {status.never_run_count}"
    )


__all__ = [
    "DEFAULT_POLICY",
    "CatalogueStatus",
    "FreshnessLevel",
    "RequiredBroker",
    "RunOutcome",
    "RunRecord",
    "Scenario",
    "ScenarioKind",
    "ScenarioPolicy",
    "all_scenarios",
    "build_status",
    "freshness_for",
    "last_passed_run",
    "render_scenario",
    "render_status",
    "scenario",
    "scenarios_for_broker",
    "scenarios_for_kind",
]
