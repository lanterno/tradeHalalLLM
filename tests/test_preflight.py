"""Tests for `halal_trader.ops.preflight`.

Auxiliary primitive for bot startup safety. Covers: severity ladder,
check outcome classification, fail-fast vs full-report aggregation,
exception handling in runners.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from halal_trader.ops.preflight import (
    CheckOutcome,
    CheckResult,
    CheckSeverity,
    CheckSpec,
    PreflightReport,
    critical_failures,
    render_report,
    render_result,
    run_checks,
    warnings_only,
)

# --------------------------- Enum string pins --------------------------------


def test_check_severity_string_values_pinned() -> None:
    assert CheckSeverity.INFO.value == "info"
    assert CheckSeverity.WARN.value == "warn"
    assert CheckSeverity.CRITICAL.value == "critical"


def test_check_outcome_string_values_pinned() -> None:
    assert CheckOutcome.PASSED.value == "passed"
    assert CheckOutcome.WARNED.value == "warned"
    assert CheckOutcome.FAILED.value == "failed"


# --------------------------- CheckSpec ---------------------------------------


def test_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        CheckSpec(name="", description="x", severity=CheckSeverity.INFO)


def test_spec_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="description"):
        CheckSpec(name="x", description="", severity=CheckSeverity.INFO)


def test_spec_is_frozen() -> None:
    spec = CheckSpec(
        name="db_reachable",
        description="DB connectivity",
        severity=CheckSeverity.CRITICAL,
    )
    with pytest.raises(FrozenInstanceError):
        spec.severity = CheckSeverity.WARN  # type: ignore[misc]


# --------------------------- CheckResult outcome -----------------------------


def test_passed_check_returns_passed() -> None:
    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    result = CheckResult(spec=spec, passed=True, message="all good")
    assert result.outcome is CheckOutcome.PASSED


def test_failed_critical_returns_failed() -> None:
    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    result = CheckResult(spec=spec, passed=False, message="DB unreachable")
    assert result.outcome is CheckOutcome.FAILED


def test_failed_warn_returns_warned() -> None:
    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.WARN,
    )
    result = CheckResult(spec=spec, passed=False, message="rate limit at 80%")
    assert result.outcome is CheckOutcome.WARNED


def test_failed_info_returns_passed() -> None:
    """Pin: INFO checks never fail (informational only)."""

    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.INFO,
    )
    result = CheckResult(spec=spec, passed=False, message="info note")
    assert result.outcome is CheckOutcome.PASSED


def test_result_rejects_empty_message() -> None:
    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.INFO,
    )
    with pytest.raises(ValueError, match="message"):
        CheckResult(spec=spec, passed=True, message="")


def test_result_is_frozen() -> None:
    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.INFO,
    )
    result = CheckResult(spec=spec, passed=True, message="x")
    with pytest.raises(FrozenInstanceError):
        result.passed = False  # type: ignore[misc]


# --------------------------- PreflightReport --------------------------------


def test_report_rejects_inconsistent_counts() -> None:
    """Pin: counts must equal results length."""

    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.INFO,
    )
    result = CheckResult(spec=spec, passed=True, message="x")
    with pytest.raises(ValueError, match="counts"):
        PreflightReport(
            results=(result,),
            passed_count=10,  # doesn't match 1 result
            warned_count=0,
            failed_count=0,
        )


def test_report_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="passed_count"):
        PreflightReport(
            results=(),
            passed_count=-1,
            warned_count=0,
            failed_count=0,
        )


def test_report_is_safe_to_start_when_no_failures() -> None:
    """Pin: only CRITICAL failures block startup."""

    spec_warn = CheckSpec(
        name="warn_check",
        description="x",
        severity=CheckSeverity.WARN,
    )
    spec_pass = CheckSpec(
        name="pass_check",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    warned = CheckResult(spec=spec_warn, passed=False, message="warn")
    passed = CheckResult(spec=spec_pass, passed=True, message="ok")
    report = PreflightReport(
        results=(warned, passed),
        passed_count=1,
        warned_count=1,
        failed_count=0,
    )
    assert report.is_safe_to_start is True


def test_report_not_safe_when_critical_failure() -> None:
    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    failed = CheckResult(spec=spec, passed=False, message="critical failure")
    report = PreflightReport(
        results=(failed,),
        passed_count=0,
        warned_count=0,
        failed_count=1,
    )
    assert report.is_safe_to_start is False


def test_report_is_frozen() -> None:
    report = PreflightReport(
        results=(),
        passed_count=0,
        warned_count=0,
        failed_count=0,
    )
    with pytest.raises(FrozenInstanceError):
        report.passed_count = 99  # type: ignore[misc]


# --------------------------- run_checks --------------------------------------


def _runner(passed: bool, message: str) -> Callable[[], tuple[bool, str]]:
    """Helper: build a runner closure with fixed return."""

    def runner() -> tuple[bool, str]:
        return passed, message

    return runner


def test_run_checks_empty() -> None:
    report = run_checks([])
    assert report.passed_count == 0
    assert report.warned_count == 0
    assert report.failed_count == 0
    assert report.is_safe_to_start is True


def test_run_checks_all_passing() -> None:
    checks = [
        (
            CheckSpec(
                name="db",
                description="DB reachable",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "DB reachable at localhost:5433"),
        ),
        (
            CheckSpec(
                name="vault",
                description="Vault valid",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "Vault has all required keys"),
        ),
    ]
    report = run_checks(checks)
    assert report.passed_count == 2
    assert report.failed_count == 0
    assert report.is_safe_to_start is True


def test_run_checks_one_failure() -> None:
    checks = [
        (
            CheckSpec(
                name="db",
                description="DB",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(False, "DB unreachable"),
        ),
    ]
    report = run_checks(checks)
    assert report.failed_count == 1
    assert report.is_safe_to_start is False


def test_run_checks_warning_doesnt_fail() -> None:
    """Pin: WARN-level failures don't block startup."""

    checks = [
        (
            CheckSpec(
                name="rate_limit",
                description="Rate limit healthy",
                severity=CheckSeverity.WARN,
            ),
            _runner(False, "rate limit at 80%"),
        ),
    ]
    report = run_checks(checks)
    assert report.warned_count == 1
    assert report.failed_count == 0
    assert report.is_safe_to_start is True


def test_run_checks_info_failure_passes() -> None:
    """Pin: INFO-level checks never fail; informational only."""

    checks = [
        (
            CheckSpec(
                name="version_info",
                description="Bot version",
                severity=CheckSeverity.INFO,
            ),
            _runner(False, "bot version 1.0.0"),
        ),
    ]
    report = run_checks(checks)
    assert report.passed_count == 1
    assert report.failed_count == 0


def test_run_checks_runner_exception_treated_as_failure() -> None:
    """Pin: runner raising → check fails with exception message.

    The engine doesn't crash; it captures the exception so the
    report still aggregates.
    """

    def raising_runner() -> tuple[bool, str]:
        raise RuntimeError("connection timeout")

    checks = [
        (
            CheckSpec(
                name="x",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            raising_runner,
        ),
    ]
    report = run_checks(checks)
    assert report.failed_count == 1
    assert report.results[0].passed is False
    assert "connection timeout" in report.results[0].message


def test_run_checks_results_sorted_by_name() -> None:
    """Pin: deterministic ordering."""

    checks = [
        (
            CheckSpec(
                name="zulu",
                description="x",
                severity=CheckSeverity.INFO,
            ),
            _runner(True, "ok"),
        ),
        (
            CheckSpec(
                name="alpha",
                description="x",
                severity=CheckSeverity.INFO,
            ),
            _runner(True, "ok"),
        ),
        (
            CheckSpec(
                name="mike",
                description="x",
                severity=CheckSeverity.INFO,
            ),
            _runner(True, "ok"),
        ),
    ]
    report = run_checks(checks)
    names = [r.spec.name for r in report.results]
    assert names == ["alpha", "mike", "zulu"]


def test_run_checks_fail_fast_stops_on_first_critical() -> None:
    """Pin: fail_fast=True stops after first CRITICAL failure."""

    call_count = {"x": 0}

    def counting_runner(name: str, passed: bool) -> Callable[[], tuple[bool, str]]:
        def inner() -> tuple[bool, str]:
            call_count[name] = call_count.get(name, 0) + 1
            return passed, f"{name} done"

        return inner

    checks = [
        (
            CheckSpec(
                name="aaa_fails",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            counting_runner("aaa", False),
        ),
        (
            CheckSpec(
                name="bbb_passes",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            counting_runner("bbb", True),
        ),
    ]
    run_checks(checks, fail_fast=True)
    assert call_count.get("aaa") == 1
    assert call_count.get("bbb") is None  # never called


def test_run_checks_fail_fast_doesnt_stop_on_warn() -> None:
    """Pin: fail_fast only stops on CRITICAL, not WARN."""

    call_count = {"x": 0}

    def counting_runner(name: str, passed: bool) -> Callable[[], tuple[bool, str]]:
        def inner() -> tuple[bool, str]:
            call_count[name] = call_count.get(name, 0) + 1
            return passed, f"{name} done"

        return inner

    checks = [
        (
            CheckSpec(
                name="aaa_warns",
                description="x",
                severity=CheckSeverity.WARN,
            ),
            counting_runner("aaa", False),
        ),
        (
            CheckSpec(
                name="bbb_passes",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            counting_runner("bbb", True),
        ),
    ]
    run_checks(checks, fail_fast=True)
    # Both ran because the first only WARNED, not FAILED
    assert call_count.get("aaa") == 1
    assert call_count.get("bbb") == 1


def test_run_checks_no_fail_fast_runs_all() -> None:
    """Pin: default fail_fast=False runs every check for full report."""

    call_count = {"x": 0}

    def counting_runner(name: str, passed: bool) -> Callable[[], tuple[bool, str]]:
        def inner() -> tuple[bool, str]:
            call_count[name] = call_count.get(name, 0) + 1
            return passed, f"{name} done"

        return inner

    checks = [
        (
            CheckSpec(
                name="aaa_fails",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            counting_runner("aaa", False),
        ),
        (
            CheckSpec(
                name="bbb_passes",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            counting_runner("bbb", True),
        ),
    ]
    run_checks(checks)  # fail_fast defaults False
    # Both ran
    assert call_count.get("aaa") == 1
    assert call_count.get("bbb") == 1


def test_run_checks_is_deterministic() -> None:
    """Pin: same checks → same report."""

    checks = [
        (
            CheckSpec(
                name="x",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "ok"),
        ),
    ]
    a = run_checks(checks)
    b = run_checks(checks)
    assert a == b


# --------------------------- critical_failures + warnings_only ---------------


def test_critical_failures_filter() -> None:
    spec_pass = CheckSpec(
        name="p",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    spec_warn = CheckSpec(
        name="w",
        description="x",
        severity=CheckSeverity.WARN,
    )
    spec_fail = CheckSpec(
        name="f",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    passed = CheckResult(spec=spec_pass, passed=True, message="x")
    warned = CheckResult(spec=spec_warn, passed=False, message="x")
    failed = CheckResult(spec=spec_fail, passed=False, message="x")
    report = PreflightReport(
        results=(passed, warned, failed),
        passed_count=1,
        warned_count=1,
        failed_count=1,
    )
    fails = critical_failures(report)
    assert len(fails) == 1
    assert fails[0].spec.name == "f"


def test_warnings_only_filter() -> None:
    spec_warn = CheckSpec(
        name="w",
        description="x",
        severity=CheckSeverity.WARN,
    )
    spec_pass = CheckSpec(
        name="p",
        description="x",
        severity=CheckSeverity.INFO,
    )
    warned = CheckResult(spec=spec_warn, passed=False, message="warn")
    passed = CheckResult(spec=spec_pass, passed=True, message="info")
    report = PreflightReport(
        results=(warned, passed),
        passed_count=1,
        warned_count=1,
        failed_count=0,
    )
    warns = warnings_only(report)
    assert len(warns) == 1
    assert warns[0].spec.name == "w"


# --------------------------- render ------------------------------------------


def test_render_result_passed_emoji() -> None:
    spec = CheckSpec(
        name="db",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    result = CheckResult(spec=spec, passed=True, message="DB at :5433")
    out = render_result(result)
    assert "✅" in out
    assert "db" in out
    assert "DB at :5433" in out


def test_render_result_failed_emoji() -> None:
    spec = CheckSpec(
        name="db",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    result = CheckResult(spec=spec, passed=False, message="DB unreachable")
    out = render_result(result)
    assert "❌" in out


def test_render_report_all_go() -> None:
    """Pin: report with zero issues → 'ALL GO' top line."""

    checks = [
        (
            CheckSpec(
                name="x",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "ok"),
        ),
    ]
    report = run_checks(checks)
    out = render_report(report)
    assert "ALL GO" in out
    assert "✅" in out


def test_render_report_with_warnings() -> None:
    """Pin: warnings present but no failures → 'GO with N warning(s)'."""

    checks = [
        (
            CheckSpec(
                name="x",
                description="x",
                severity=CheckSeverity.WARN,
            ),
            _runner(False, "rate limit at 80%"),
        ),
    ]
    report = run_checks(checks)
    out = render_report(report)
    assert "GO with 1 warning" in out


def test_render_report_no_go() -> None:
    """Pin: critical failure → 'NO-GO' top line."""

    checks = [
        (
            CheckSpec(
                name="db",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(False, "DB unreachable"),
        ),
    ]
    report = run_checks(checks)
    out = render_report(report)
    assert "NO-GO" in out


def test_render_report_lists_failures_first() -> None:
    """Pin: critical failures section appears before warnings + passed."""

    checks = [
        (
            CheckSpec(
                name="aaa_pass",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "ok"),
        ),
        (
            CheckSpec(
                name="bbb_warn",
                description="x",
                severity=CheckSeverity.WARN,
            ),
            _runner(False, "warn issue"),
        ),
        (
            CheckSpec(
                name="ccc_fail",
                description="x",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(False, "DB issue"),
        ),
    ]
    report = run_checks(checks)
    out = render_report(report)
    failures_idx = out.index("CRITICAL FAILURES")
    warnings_idx = out.index("WARNINGS")
    passed_idx = out.index("PASSED")
    assert failures_idx < warnings_idx < passed_idx


def test_render_no_secret_leak() -> None:
    """Pin: render shows summary message; no raw stack traces / API
    responses (those are operator-side debug log)."""

    spec = CheckSpec(
        name="db",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    # Operator's runner correctly returns a high-level message
    # (not the raw exception)
    result = CheckResult(
        spec=spec,
        passed=False,
        message="DB unreachable at configured host",
    )
    out = render_result(result)
    # The high-level message is present
    assert "DB unreachable" in out
    # Rendering doesn't leak typical sensitive substrings
    assert "Traceback" not in out
    assert "at /home/" not in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_realistic_startup_sequence() -> None:
    """Real-world: bot startup runs 6 checks; 4 pass, 1 warn, 1 fail."""

    checks = [
        (
            CheckSpec(
                name="db_reachable",
                description="Postgres reachable at configured URL",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "Postgres reachable at localhost:5433"),
        ),
        (
            CheckSpec(
                name="db_schema_at_head",
                description="Alembic at head revision",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "Alembic at revision abc123"),
        ),
        (
            CheckSpec(
                name="vault_keys_valid",
                description="Vault has all required keys",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(True, "Vault has 4/4 required keys"),
        ),
        (
            CheckSpec(
                name="broker_authenticated",
                description="Broker plugin authenticated",
                severity=CheckSeverity.CRITICAL,
            ),
            _runner(False, "Broker auth returned 401"),  # critical fail
        ),
        (
            CheckSpec(
                name="alert_router_reachable",
                description="At least one alert channel reachable",
                severity=CheckSeverity.WARN,
            ),
            _runner(False, "Telegram unreachable; Slack still works"),
        ),
        (
            CheckSpec(
                name="cycle_interval_set",
                description="Cycle interval configured",
                severity=CheckSeverity.INFO,
            ),
            _runner(True, "Cycle interval: 60s"),
        ),
    ]
    report = run_checks(checks)
    assert report.passed_count == 4
    assert report.warned_count == 1
    assert report.failed_count == 1
    assert report.is_safe_to_start is False


def test_e2e_replay_consistency() -> None:
    """Same checks → same report."""

    spec = CheckSpec(
        name="x",
        description="x",
        severity=CheckSeverity.CRITICAL,
    )
    runner = _runner(True, "ok")
    a = run_checks([(spec, runner)])
    b = run_checks([(spec, runner)])
    assert a == b
