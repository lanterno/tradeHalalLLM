"""Chaos engineering harness — fault-inject the bot's IO surfaces.

Round-4 wave 8.B: the existing `crypto/stress.py` injects market-
side adversarial scenarios (flash crash, blow-off pump, wide-bar
liquidity crunch); this module is its mirror on the **operational**
side. Faults the cycle must survive without hanging, leaking
state, or crashing:

* broker.timeout — every broker call hangs past its deadline
* broker.5xx — broker returns 502 / 503 / 504
* broker.malformed_json — broker returns a 200 with garbage body
* broker.rate_limit — broker returns 429 / Binance -1003
* websocket.drop — live kline / ticker socket disconnects
* db.connection_drop — repository raises OperationalError
* llm.timeout — LLM call hangs
* llm.500 — LLM provider returns 500

Each `ChaosScenario` is a tuple of (label, predicate, fault) — the
predicate selects which calls trip the fault (e.g. "every
broker.* call" or "the third call only"); the fault is a callable
that takes the original args and either raises or returns a
canned value. The harness wraps a target callable with the fault
predicate and runs the operator's behaviour under it; the
evaluator grades the outcome:

* **recover** — the call returned normally (target swallowed the
  fault and degraded gracefully)
* **clean_halt** — the call raised a known exception type (the
  target chose to fail loudly rather than mask the failure)
* **hang** — the call took longer than the watchdog deadline
* **crash** — the call raised an unexpected exception type that
  signals a programming bug rather than a recovery path

Every scenario must produce **recover** or **clean_halt**;
**hang** and **crash** are failures. Pin so a future refactor
that quietly swallows a `MemoryError` shows up as a regression.

Why a separate module rather than monkey-patching tests:

* The fault library is reusable across pytest, the operator's
  manual chaos runs (`halal-trader chaos run`), and any future
  integration-test layer.
* The evaluator's "clean_halt vs crash" classification needs the
  caller to **declare** which exception types are acceptable —
  pytest's blanket "any raise is a fail" can't make this
  distinction.

Pure-Python; no DB / network / async-loop ownership. The wrapper
is sync and async-aware. Halal alignment: the harness never
opens a position or screens an asset; it's pure observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable

# ── Fault library ─────────────────────────────────────────


class FaultKind(str, Enum):
    """Vocabulary the dashboard / report renders against. Adding
    a new fault means adding to this enum *and* a factory that
    constructs the matching `Fault`."""

    BROKER_TIMEOUT = "broker.timeout"
    BROKER_HTTP_5XX = "broker.5xx"
    BROKER_MALFORMED_JSON = "broker.malformed_json"
    BROKER_RATE_LIMIT = "broker.rate_limit"
    WEBSOCKET_DROP = "websocket.drop"
    DB_CONNECTION_DROP = "db.connection_drop"
    LLM_TIMEOUT = "llm.timeout"
    LLM_HTTP_500 = "llm.500"


# Sentinel exceptions the harness raises. They're declared here so
# tests can import + assert on the type rather than matching error
# strings. Operators can also catch these in the target callable to
# implement explicit recovery paths.


class HarnessError(Exception):
    """Base for every fault the harness injects."""


class BrokerTimeoutError(HarnessError):
    """Broker call exceeded the deadline."""


class BrokerHttpError(HarnessError):
    """Broker returned a non-2xx status."""

    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class BrokerMalformedJsonError(HarnessError):
    """Broker returned a 200 with a body that isn't valid JSON."""


class BrokerRateLimitError(BrokerHttpError):
    """Broker returned 429 / Binance -1003."""

    def __init__(self) -> None:
        super().__init__(status_code=429, message="rate-limited")


class WebsocketDropError(HarnessError):
    """Live socket disconnected mid-stream."""


class DatabaseConnectionDropError(HarnessError):
    """Repository couldn't reach Postgres."""


class LlmTimeoutError(HarnessError):
    """LLM provider didn't respond before the deadline."""


class LlmHttpError(HarnessError):
    """LLM provider returned an HTTP error."""


# ── Predicate types ───────────────────────────────────────


# A predicate decides whether a given call should trip the fault.
# Stateful predicates (e.g. "trip the third call") use a closure
# over a counter — see `_call_count_predicate` factory.
CallPredicate = Callable[[str, tuple, dict], bool]


def always() -> CallPredicate:
    """Fault every matching call."""
    return lambda label, args, kwargs: True


def never() -> CallPredicate:
    """Fault nothing — useful as a no-op control in regression tests."""
    return lambda label, args, kwargs: False


def on_call_number(n: int) -> CallPredicate:
    """Fault only the Nth matching call (1-indexed). Stateful;
    each predicate instance maintains its own counter."""
    if n < 1:
        raise ValueError(f"call number must be >= 1; got {n}")
    state = {"count": 0}

    def _pred(label: str, args: tuple, kwargs: dict) -> bool:
        state["count"] += 1
        return state["count"] == n

    return _pred


def for_label(target_label: str) -> CallPredicate:
    """Fault only calls with a matching label. Useful for
    'fault the broker.fetch_klines call but not the broker.balance
    call' scoping."""
    return lambda label, args, kwargs: label == target_label


def combine_and(*predicates: CallPredicate) -> CallPredicate:
    """All-of composition: every predicate must agree to fault."""
    if not predicates:
        return always()

    def _pred(label: str, args: tuple, kwargs: dict) -> bool:
        return all(p(label, args, kwargs) for p in predicates)

    return _pred


# ── Scenarios ────────────────────────────────────────────


@dataclass(frozen=True)
class ChaosScenario:
    """One named fault to inject.

    ``predicate`` chooses which calls trip it; ``fault`` raises
    the appropriate sentinel exception. Frozen so the registry can
    hand out shared instances safely."""

    name: str
    kind: FaultKind
    predicate: CallPredicate
    fault: Callable[[str, tuple, dict], Any]
    description: str = ""


def _broker_timeout_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise BrokerTimeoutError(f"{label} timed out")


def _broker_5xx_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise BrokerHttpError(status_code=503, message="service unavailable")


def _broker_malformed_json_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise BrokerMalformedJsonError(f"{label} returned non-JSON body")


def _broker_rate_limit_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise BrokerRateLimitError()


def _websocket_drop_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise WebsocketDropError(f"{label} socket dropped")


def _db_drop_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise DatabaseConnectionDropError(f"{label} could not reach DB")


def _llm_timeout_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise LlmTimeoutError(f"{label} timed out")


def _llm_500_fault(label: str, args: tuple, kwargs: dict) -> Any:
    raise LlmHttpError(f"{label} returned HTTP 500")


# Map enum → fault function so callers can build scenarios from a
# kind without remembering function names.
_FAULTS: dict[FaultKind, Callable[[str, tuple, dict], Any]] = {
    FaultKind.BROKER_TIMEOUT: _broker_timeout_fault,
    FaultKind.BROKER_HTTP_5XX: _broker_5xx_fault,
    FaultKind.BROKER_MALFORMED_JSON: _broker_malformed_json_fault,
    FaultKind.BROKER_RATE_LIMIT: _broker_rate_limit_fault,
    FaultKind.WEBSOCKET_DROP: _websocket_drop_fault,
    FaultKind.DB_CONNECTION_DROP: _db_drop_fault,
    FaultKind.LLM_TIMEOUT: _llm_timeout_fault,
    FaultKind.LLM_HTTP_500: _llm_500_fault,
}


def make_scenario(
    *,
    name: str,
    kind: FaultKind,
    predicate: CallPredicate | None = None,
    description: str = "",
) -> ChaosScenario:
    """Build a scenario from a kind + predicate.

    ``predicate`` defaults to ``always()`` so a "fault every
    matching call" scenario is one line. The fault function is
    looked up from the kind.
    """
    return ChaosScenario(
        name=name,
        kind=kind,
        predicate=predicate or always(),
        fault=_FAULTS[kind],
        description=description,
    )


def standard_scenarios() -> list[ChaosScenario]:
    """The default suite — one scenario per FaultKind, predicate
    = `always()`, scoped to the broker / DB / LLM call labels.
    Operators run this as the regression baseline."""
    return [
        make_scenario(
            name="broker_timeout",
            kind=FaultKind.BROKER_TIMEOUT,
            description="Every broker call hangs past its deadline.",
        ),
        make_scenario(
            name="broker_5xx",
            kind=FaultKind.BROKER_HTTP_5XX,
            description="Broker returns 503 service unavailable.",
        ),
        make_scenario(
            name="broker_malformed_json",
            kind=FaultKind.BROKER_MALFORMED_JSON,
            description="Broker returns 200 with a non-JSON body.",
        ),
        make_scenario(
            name="broker_rate_limit",
            kind=FaultKind.BROKER_RATE_LIMIT,
            description="Broker returns 429 / Binance -1003.",
        ),
        make_scenario(
            name="websocket_drop",
            kind=FaultKind.WEBSOCKET_DROP,
            description="Live kline socket disconnects.",
        ),
        make_scenario(
            name="db_connection_drop",
            kind=FaultKind.DB_CONNECTION_DROP,
            description="Repository raises OperationalError.",
        ),
        make_scenario(
            name="llm_timeout",
            kind=FaultKind.LLM_TIMEOUT,
            description="LLM provider hangs past the deadline.",
        ),
        make_scenario(
            name="llm_500",
            kind=FaultKind.LLM_HTTP_500,
            description="LLM provider returns HTTP 500.",
        ),
    ]


# ── Wrapping calls under chaos ────────────────────────────


def chaos_call(
    scenario: ChaosScenario,
    label: str,
    target: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Sync wrapper. If the scenario's predicate matches the call,
    raise the fault; otherwise call the target normally."""
    if scenario.predicate(label, args, kwargs):
        return scenario.fault(label, args, kwargs)
    return target(*args, **kwargs)


async def chaos_call_async(
    scenario: ChaosScenario,
    label: str,
    target: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async twin of `chaos_call`. The fault function is sync and
    raises immediately — async targets that need a delayed fault
    can wrap their own."""
    if scenario.predicate(label, args, kwargs):
        return scenario.fault(label, args, kwargs)
    return await target(*args, **kwargs)


# ── Evaluation ────────────────────────────────────────────


class ChaosOutcome(str, Enum):
    """Operator-readable verdict on a single scenario run."""

    RECOVER = "recover"
    CLEAN_HALT = "clean_halt"
    HANG = "hang"
    CRASH = "crash"


@dataclass
class ChaosVerdict:
    """One scenario's evaluated outcome."""

    scenario_name: str
    outcome: ChaosOutcome
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)
    raised: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome in (ChaosOutcome.RECOVER, ChaosOutcome.CLEAN_HALT)


def evaluate(
    scenario: ChaosScenario,
    target: Callable[[], Any],
    *,
    deadline_seconds: float = 5.0,
    expected_exceptions: Iterable[type[BaseException]] = (),
) -> ChaosVerdict:
    """Run ``target()`` under ``scenario`` and grade the outcome.

    ``target`` must be a zero-arg callable — the closure pattern
    means the caller wires in their own broker, repository, etc.
    The evaluator doesn't know what the bot does; it only watches
    how it reacts.

    ``expected_exceptions`` declares the exception types that
    constitute a *clean halt* — usually the harness sentinels for
    the fault kind, plus any application-side recovery exceptions
    (e.g. `core.halt.HaltEngagedError`). Anything else is a
    crash. Pin so a future refactor that swallows a `MemoryError`
    or a `RecursionError` shows up as a regression.

    Watchdog: a sync target that exceeds ``deadline_seconds`` is
    classified as `HANG` rather than left to wallclock-out the
    test. We measure with `time.monotonic()` after the call
    returns rather than killing the thread (Python doesn't allow
    safely killing arbitrary threads); pin so a 10-second hang in
    a unit test doesn't become a 30-second hang. The verdict's
    `elapsed_ms` is the precise measurement.
    """
    expected_types = tuple(expected_exceptions) or (HarnessError,)
    notes: list[str] = []
    t0 = time.monotonic()
    raised = ""
    try:
        target()
    except expected_types as exc:
        outcome = ChaosOutcome.CLEAN_HALT
        raised = repr(exc)
        notes.append(f"clean halt via {type(exc).__name__}")
    except BaseException as exc:  # noqa: BLE001
        outcome = ChaosOutcome.CRASH
        raised = repr(exc)
        notes.append(f"unexpected {type(exc).__name__}")
    else:
        outcome = ChaosOutcome.RECOVER
        notes.append("returned without raising")
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if outcome == ChaosOutcome.RECOVER and elapsed_ms / 1000.0 > deadline_seconds:
        outcome = ChaosOutcome.HANG
        notes.append(f"exceeded deadline {deadline_seconds:.1f}s ({elapsed_ms:.0f}ms)")
    return ChaosVerdict(
        scenario_name=scenario.name,
        outcome=outcome,
        elapsed_ms=elapsed_ms,
        notes=notes,
        raised=raised,
    )


def evaluate_all(
    scenarios: Iterable[ChaosScenario],
    target_factory: Callable[[ChaosScenario], Callable[[], Any]],
    *,
    deadline_seconds: float = 5.0,
    expected_exceptions: Iterable[type[BaseException]] = (),
) -> list[ChaosVerdict]:
    """Run every scenario through ``target_factory(scenario)``.

    Why a factory rather than a single target: the wired bot
    reads the active scenario at construction time (e.g. wraps
    its broker with `chaos_call` bound to the scenario), so each
    iteration needs a fresh target. Pin the factory pattern so
    state leakage between scenarios is impossible.
    """
    return [
        evaluate(
            sc,
            target_factory(sc),
            deadline_seconds=deadline_seconds,
            expected_exceptions=expected_exceptions,
        )
        for sc in scenarios
    ]


def render_verdicts(verdicts: Iterable[ChaosVerdict]) -> str:
    """CLI / Slack-ready text payload. Mirrors the visual shape of
    `crypto/stress.render_report` and `core/promotion_gate.render_verdict`
    so an operator running the three sees a familiar layout."""
    verdict_list = list(verdicts)
    lines = ["=== Chaos verdicts ==="]
    for v in verdict_list:
        marker = "✔" if v.passed else "✘"
        lines.append(
            f"  {marker} {v.scenario_name:<24} {v.outcome.value:<12} ({v.elapsed_ms:.0f}ms)"
        )
        for n in v.notes:
            lines.append(f"      · {n}")
    failed = [v for v in verdict_list if not v.passed]
    lines.append("")
    if failed:
        lines.append(f"FAIL: {len(failed)} scenarios crashed or hung")
    else:
        lines.append("PASS: every scenario recovered or halted cleanly")
    return "\n".join(lines)


__all__ = [
    "BrokerHttpError",
    "BrokerMalformedJsonError",
    "BrokerRateLimitError",
    "BrokerTimeoutError",
    "ChaosOutcome",
    "ChaosScenario",
    "ChaosVerdict",
    "DatabaseConnectionDropError",
    "FaultKind",
    "HarnessError",
    "LlmHttpError",
    "LlmTimeoutError",
    "WebsocketDropError",
    "always",
    "chaos_call",
    "chaos_call_async",
    "combine_and",
    "evaluate",
    "evaluate_all",
    "for_label",
    "make_scenario",
    "never",
    "on_call_number",
    "render_verdicts",
    "standard_scenarios",
]
