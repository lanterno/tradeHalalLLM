"""Tests for `core/chaos.py` (chaos engineering harness).

Pins each predicate's selection logic, the fault library's
exception types, the sync + async wrappers' fault-vs-passthrough
contract, the evaluator's four-bucket classification (recover /
clean_halt / hang / crash), the watchdog deadline behaviour, and
the standard-scenarios suite.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from halal_trader.core.chaos import (
    BrokerHttpError,
    BrokerMalformedJsonError,
    BrokerRateLimitError,
    BrokerTimeoutError,
    ChaosOutcome,
    ChaosScenario,
    ChaosVerdict,
    DatabaseConnectionDropError,
    FaultKind,
    HarnessError,
    LlmHttpError,
    LlmTimeoutError,
    WebsocketDropError,
    always,
    chaos_call,
    chaos_call_async,
    combine_and,
    evaluate,
    evaluate_all,
    for_label,
    make_scenario,
    never,
    on_call_number,
    render_verdicts,
    standard_scenarios,
)

# ── predicates ───────────────────────────────────────────


def test_always_predicate_returns_true_unconditionally():
    assert always()("any", (), {}) is True


def test_never_predicate_returns_false_unconditionally():
    assert never()("any", (), {}) is False


def test_on_call_number_trips_only_the_nth_call():
    pred = on_call_number(3)
    results = [pred("x", (), {}) for _ in range(5)]
    # 1st call false, 2nd false, 3rd true, 4th false, 5th false
    assert results == [False, False, True, False, False]


def test_on_call_number_rejects_zero_or_negative():
    with pytest.raises(ValueError, match=">= 1"):
        on_call_number(0)
    with pytest.raises(ValueError, match=">= 1"):
        on_call_number(-1)


def test_for_label_matches_only_the_target_label():
    pred = for_label("broker.fetch")
    assert pred("broker.fetch", (), {}) is True
    assert pred("broker.balance", (), {}) is False


def test_combine_and_requires_all_predicates():
    pred = combine_and(for_label("broker.fetch"), on_call_number(1))
    # First "broker.fetch" call → True
    assert pred("broker.fetch", (), {}) is True
    # Second "broker.fetch" call → False (count predicate)
    assert pred("broker.fetch", (), {}) is False
    # The next call number now consumed; "broker.balance" doesn't
    # trip even though the count predicate is at 3 because the
    # label predicate fails.
    assert pred("broker.balance", (), {}) is False


def test_combine_and_with_no_predicates_falls_back_to_always():
    """Pin: empty composition means 'no constraints' → match all."""
    pred = combine_and()
    assert pred("any", (), {}) is True


# ── fault library ────────────────────────────────────────


def test_make_scenario_raises_correct_fault_kind():
    """Each FaultKind must map to its sentinel exception type. Pin
    so a refactor of the fault map can't quietly swap two kinds."""
    cases = [
        (FaultKind.BROKER_TIMEOUT, BrokerTimeoutError),
        (FaultKind.BROKER_HTTP_5XX, BrokerHttpError),
        (FaultKind.BROKER_MALFORMED_JSON, BrokerMalformedJsonError),
        (FaultKind.BROKER_RATE_LIMIT, BrokerRateLimitError),
        (FaultKind.WEBSOCKET_DROP, WebsocketDropError),
        (FaultKind.DB_CONNECTION_DROP, DatabaseConnectionDropError),
        (FaultKind.LLM_TIMEOUT, LlmTimeoutError),
        (FaultKind.LLM_HTTP_500, LlmHttpError),
    ]
    for kind, expected_type in cases:
        sc = make_scenario(name="t", kind=kind)
        with pytest.raises(expected_type):
            sc.fault("label", (), {})


def test_broker_rate_limit_is_a_broker_http_error_subclass():
    """Pin: the rate-limit class extends BrokerHttpError so a
    target catching the parent automatically handles 429s."""
    sc = make_scenario(name="t", kind=FaultKind.BROKER_RATE_LIMIT)
    try:
        sc.fault("label", (), {})
    except BrokerHttpError as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("rate limit should raise BrokerHttpError")


def test_broker_http_error_carries_status_code():
    """Pin: status_code on the exception so a recovery handler
    can switch on it."""
    sc = make_scenario(name="t", kind=FaultKind.BROKER_HTTP_5XX)
    try:
        sc.fault("label", (), {})
    except BrokerHttpError as exc:
        assert exc.status_code == 503


def test_every_harness_error_descends_from_harness_error_base():
    """Pin: target callables can catch the base `HarnessError` to
    handle any injected fault generically."""
    for cls in (
        BrokerTimeoutError,
        BrokerHttpError,
        BrokerMalformedJsonError,
        BrokerRateLimitError,
        WebsocketDropError,
        DatabaseConnectionDropError,
        LlmTimeoutError,
        LlmHttpError,
    ):
        assert issubclass(cls, HarnessError)


# ── chaos_call (sync) ────────────────────────────────────


def test_chaos_call_passes_through_when_predicate_false():
    """Pin: scenario with `never()` predicate must call the
    target normally."""
    sc = make_scenario(name="off", kind=FaultKind.BROKER_TIMEOUT, predicate=never())

    def target(x, y):
        return x + y

    assert chaos_call(sc, "label", target, 3, 4) == 7


def test_chaos_call_raises_when_predicate_true():
    sc = make_scenario(name="on", kind=FaultKind.BROKER_TIMEOUT, predicate=always())
    with pytest.raises(BrokerTimeoutError):
        chaos_call(sc, "label", lambda: 42)


def test_chaos_call_predicate_sees_label_and_args():
    """Pin: predicates can switch on the label, args, and kwargs."""
    captured = {}

    def pred(label, args, kwargs):
        captured["label"] = label
        captured["args"] = args
        captured["kwargs"] = kwargs
        return False

    sc = ChaosScenario(
        name="probe",
        kind=FaultKind.BROKER_TIMEOUT,
        predicate=pred,
        fault=lambda *a: None,
    )
    chaos_call(sc, "broker.fetch", lambda x, y=1: None, 5, y=10)
    assert captured["label"] == "broker.fetch"
    assert captured["args"] == (5,)
    assert captured["kwargs"] == {"y": 10}


# ── chaos_call_async ─────────────────────────────────────


def test_chaos_call_async_passes_through():
    sc = make_scenario(name="off", kind=FaultKind.LLM_TIMEOUT, predicate=never())

    async def target(x):
        return x * 2

    assert asyncio.run(chaos_call_async(sc, "label", target, 5)) == 10


def test_chaos_call_async_raises_on_match():
    sc = make_scenario(name="on", kind=FaultKind.LLM_TIMEOUT, predicate=always())

    async def target():
        return "ok"

    async def runner():
        await chaos_call_async(sc, "label", target)

    with pytest.raises(LlmTimeoutError):
        asyncio.run(runner())


# ── evaluator ────────────────────────────────────────────


def test_evaluate_classifies_normal_return_as_recover():
    sc = standard_scenarios()[0]

    def target():
        return "ok"

    verdict = evaluate(sc, target)
    assert verdict.outcome == ChaosOutcome.RECOVER
    assert verdict.passed
    assert "returned without raising" in verdict.notes[0]


def test_evaluate_classifies_expected_exception_as_clean_halt():
    sc = standard_scenarios()[0]

    def target():
        raise BrokerTimeoutError("planned halt")

    verdict = evaluate(sc, target, expected_exceptions=(BrokerTimeoutError,))
    assert verdict.outcome == ChaosOutcome.CLEAN_HALT
    assert verdict.passed
    assert "BrokerTimeoutError" in verdict.raised


def test_evaluate_classifies_unexpected_exception_as_crash():
    """Pin: a `MemoryError` or `RecursionError` is never a clean
    halt, even if the target meant to halt — the operator wants
    these surfaced loudly."""
    sc = standard_scenarios()[0]

    def target():
        raise MemoryError("oom")

    verdict = evaluate(sc, target, expected_exceptions=(BrokerTimeoutError,))
    assert verdict.outcome == ChaosOutcome.CRASH
    assert not verdict.passed
    assert "MemoryError" in verdict.raised


def test_evaluate_default_expected_exceptions_is_harness_error():
    """Pin: when caller doesn't declare expected_exceptions, the
    base HarnessError counts as a clean halt — every fault the
    library injects is acceptable by default."""
    sc = standard_scenarios()[0]

    def target():
        raise BrokerTimeoutError()

    verdict = evaluate(sc, target)  # no expected_exceptions kwarg
    assert verdict.outcome == ChaosOutcome.CLEAN_HALT


def test_evaluate_classifies_long_run_as_hang():
    """Pin: a target that returns normally but exceeds the
    deadline is reclassified as HANG."""
    sc = standard_scenarios()[0]

    def target():
        time.sleep(0.05)
        return "ok"

    verdict = evaluate(sc, target, deadline_seconds=0.01)
    assert verdict.outcome == ChaosOutcome.HANG
    assert not verdict.passed


def test_evaluate_records_elapsed_ms():
    sc = standard_scenarios()[0]

    def target():
        time.sleep(0.01)

    verdict = evaluate(sc, target)
    assert verdict.elapsed_ms >= 10.0


def test_evaluate_returns_chaos_verdict_with_scenario_name():
    sc = standard_scenarios()[0]
    verdict = evaluate(sc, lambda: None)
    assert isinstance(verdict, ChaosVerdict)
    assert verdict.scenario_name == sc.name


# ── evaluate_all ─────────────────────────────────────────


def test_evaluate_all_runs_every_scenario_via_factory():
    """Pin: target_factory is called once per scenario so each
    iteration sees a fresh target — state can't leak between
    scenarios."""
    factory_calls: list[str] = []

    def factory(sc: ChaosScenario):
        factory_calls.append(sc.name)
        return lambda: None

    verdicts = evaluate_all(standard_scenarios(), factory)
    assert len(verdicts) == len(standard_scenarios())
    assert factory_calls == [sc.name for sc in standard_scenarios()]


def test_evaluate_all_aggregates_each_scenarios_outcome():
    """Half the scenarios recover; half clean-halt."""

    def factory(sc: ChaosScenario):
        if sc.kind == FaultKind.BROKER_TIMEOUT:
            return lambda: (_ for _ in ()).throw(BrokerTimeoutError())
        return lambda: "ok"

    verdicts = evaluate_all(standard_scenarios(), factory)
    timeouts = [v for v in verdicts if v.scenario_name == "broker_timeout"]
    others = [v for v in verdicts if v.scenario_name != "broker_timeout"]
    assert timeouts[0].outcome == ChaosOutcome.CLEAN_HALT
    assert all(v.outcome == ChaosOutcome.RECOVER for v in others)


# ── standard_scenarios ───────────────────────────────────


def test_standard_scenarios_covers_every_fault_kind():
    """Pin: the regression suite tests every fault — adding a
    FaultKind without a scenario in standard_scenarios is a bug."""
    suite = standard_scenarios()
    kinds_in_suite = {sc.kind for sc in suite}
    assert kinds_in_suite == set(FaultKind)


def test_standard_scenarios_has_unique_names():
    """Names are render labels and dict keys downstream — must
    not collide."""
    names = [sc.name for sc in standard_scenarios()]
    assert len(names) == len(set(names))


def test_standard_scenarios_each_carries_a_description():
    for sc in standard_scenarios():
        assert sc.description, f"{sc.name} missing description"


# ── render_verdicts ──────────────────────────────────────


def test_render_marks_passing_scenarios_with_check():
    verdicts = [
        ChaosVerdict(
            scenario_name="ok",
            outcome=ChaosOutcome.RECOVER,
            elapsed_ms=12.0,
            notes=["all good"],
        ),
    ]
    text = render_verdicts(verdicts)
    assert "✔" in text
    assert "PASS" in text
    assert "ok" in text


def test_render_marks_failing_scenarios_with_cross():
    verdicts = [
        ChaosVerdict(
            scenario_name="bad",
            outcome=ChaosOutcome.CRASH,
            elapsed_ms=12.0,
            raised="MemoryError()",
        ),
    ]
    text = render_verdicts(verdicts)
    assert "✘" in text
    assert "FAIL" in text


def test_render_includes_elapsed_ms():
    verdicts = [
        ChaosVerdict(
            scenario_name="x",
            outcome=ChaosOutcome.RECOVER,
            elapsed_ms=42.5,
        ),
    ]
    text = render_verdicts(verdicts)
    assert "42" in text or "43" in text


def test_render_handles_empty_verdict_list():
    """Pin: empty list still renders a clean PASS line — never
    a `KeyError` or empty string."""
    text = render_verdicts([])
    assert "PASS" in text or "Chaos verdicts" in text


# ── verdict structure ────────────────────────────────────


def test_verdict_passed_property_matches_outcome_buckets():
    assert ChaosVerdict(scenario_name="x", outcome=ChaosOutcome.RECOVER, elapsed_ms=0).passed
    assert ChaosVerdict(scenario_name="x", outcome=ChaosOutcome.CLEAN_HALT, elapsed_ms=0).passed
    assert not ChaosVerdict(scenario_name="x", outcome=ChaosOutcome.HANG, elapsed_ms=0).passed
    assert not ChaosVerdict(scenario_name="x", outcome=ChaosOutcome.CRASH, elapsed_ms=0).passed


def test_scenario_is_immutable():
    sc = make_scenario(name="t", kind=FaultKind.BROKER_TIMEOUT)
    with pytest.raises(Exception):
        sc.name = "tampered"  # type: ignore[misc]
