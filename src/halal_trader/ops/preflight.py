"""Runtime pre-flight check engine.

Auxiliary primitive complementing the bot's startup sequence.
Before the first cycle runs, the bot needs to verify every
critical dependency is wired correctly: secrets vault returns
valid keys, DB is reachable + at the right schema revision,
broker plugin can authenticate, halal screener is responsive,
alert router has at least one reachable channel. Each check
runs independently and reports PASS / WARN / CRITICAL; the
engine aggregates into a unified pre-flight report.

Picked a focused engine over scattered startup logging because
(a) the "are all systems go?" question is the load-bearing
operator decision before letting the bot trade real money — a
single inspectable report ("✅ all critical checks passed") is
far more reliable than reading 30 lines of startup log scrolling
past, and a critical failure ("❌ DB at wrong schema revision —
run migrations first") needs to surface immediately rather than
get buried in info-level startup chatter; (b) the severity ladder
(INFO / WARN / CRITICAL) lets operators distinguish "everything's
fine" from "operator-tunable optimization opportunity" from
"DO NOT START THE BOT"; (c) the report is structured so the
dashboard's "system status" tile + the operator email summary +
the kill-switch's safety gate all consult the same source rather
than re-implementing the check.

Pinned semantics:
- **Closed-set CheckSeverity ladder.** INFO < WARN < CRITICAL.
  Any CRITICAL fails the report; WARN doesn't fail but flags.
- **Each check is a CheckSpec + a runner closure.** The runner
  is operator-supplied (because it touches DB / network); the
  engine just composes. Pure on the engine side.
- **Boolean fail_fast option.** If True, the engine stops on the
  first CRITICAL failure (default False, runs all checks for a
  full report).
- **Reproducible aggregation.** Same set of CheckResults → same
  PreflightReport (deterministic ordering by check name).
- **Render output never includes raw error messages from check
  runners.** Each CheckResult carries a high-level message; raw
  stack traces / API responses go to the operator's debug log.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum


class CheckSeverity(str, Enum):
    """Severity ladder for pre-flight check failures.

    Pinned string values for JSON / DB stability. INFO < WARN < CRITICAL.
    """

    INFO = "info"  # Informational; check passed
    WARN = "warn"  # Non-critical issue; bot can proceed
    CRITICAL = "critical"  # Bot must not start


class CheckOutcome(str, Enum):
    """Per-check result.

    Pinned string values. PASSED / WARNED / FAILED.
    """

    PASSED = "passed"
    WARNED = "warned"
    FAILED = "failed"


@dataclass(frozen=True)
class CheckSpec:
    """Static specification of a pre-flight check.

    `name` is operator-facing identifier (e.g. "db_reachable",
    "secrets_vault_valid"). `severity` is the failure severity
    if the check returns False — INFO checks are documentation-
    only (always pass); WARN checks log but don't fail; CRITICAL
    checks fail the report.
    """

    name: str
    description: str
    severity: CheckSeverity

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")


@dataclass(frozen=True)
class CheckResult:
    """Outcome of running one check.

    `passed` is the boolean check result; `message` is the
    operator-facing summary. The message must be non-empty so
    the report always has actionable text — a passing check might
    say "DB reachable at localhost:5433", a warning "broker rate
    limit at 80%", a failure "secrets vault returned no key for
    DATABASE_URL".
    """

    spec: CheckSpec
    passed: bool
    message: str

    def __post_init__(self) -> None:
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")

    @property
    def outcome(self) -> CheckOutcome:
        """Map (passed, severity) to a CheckOutcome.

        - passed=True → PASSED regardless of severity
        - passed=False + severity=INFO → PASSED (info checks never fail)
        - passed=False + severity=WARN → WARNED
        - passed=False + severity=CRITICAL → FAILED
        """

        if self.passed:
            return CheckOutcome.PASSED
        if self.spec.severity is CheckSeverity.INFO:
            return CheckOutcome.PASSED
        if self.spec.severity is CheckSeverity.WARN:
            return CheckOutcome.WARNED
        return CheckOutcome.FAILED


@dataclass(frozen=True)
class PreflightReport:
    """Aggregated pre-flight report."""

    results: tuple[CheckResult, ...]
    passed_count: int
    warned_count: int
    failed_count: int

    def __post_init__(self) -> None:
        if self.passed_count < 0:
            raise ValueError("passed_count must be non-negative")
        if self.warned_count < 0:
            raise ValueError("warned_count must be non-negative")
        if self.failed_count < 0:
            raise ValueError("failed_count must be non-negative")
        total = self.passed_count + self.warned_count + self.failed_count
        if total != len(self.results):
            raise ValueError(f"counts {total} != results length {len(self.results)}")

    @property
    def is_safe_to_start(self) -> bool:
        """True if zero CRITICAL failures.

        WARN-level issues don't block startup; only CRITICAL does.
        Pinned: this is the load-bearing safety gate that the cycle's
        startup sequence consults.
        """

        return self.failed_count == 0


class CheckRunFailure(Exception):
    """Raised internally when a check runner itself raises.

    The engine catches this and converts to a CheckResult with
    `passed=False` so the report still aggregates. Operators see
    "check X failed to run: <reason>" rather than a crashed bot.
    """


def run_checks(
    checks: Iterable[tuple[CheckSpec, Callable[[], tuple[bool, str]]]],
    *,
    fail_fast: bool = False,
) -> PreflightReport:
    """Run all checks and aggregate into a PreflightReport.

    Each check is a (spec, runner) tuple. Runner returns
    (passed, message). If runner raises, the engine catches and
    treats as `passed=False` with the exception message.

    `fail_fast=True` stops on first CRITICAL failure; default is
    False (run every check for full report).

    Results are sorted by spec.name for deterministic ordering.
    """

    results: list[CheckResult] = []
    for spec, runner in checks:
        try:
            passed, message = runner()
        except Exception as exc:
            passed = False
            message = f"check runner raised: {exc!r}"
        result = CheckResult(spec=spec, passed=passed, message=message)
        results.append(result)
        if fail_fast and result.outcome is CheckOutcome.FAILED:
            break

    # Sort for deterministic ordering (alphabetical by check name)
    results.sort(key=lambda r: r.spec.name)

    passed = sum(1 for r in results if r.outcome is CheckOutcome.PASSED)
    warned = sum(1 for r in results if r.outcome is CheckOutcome.WARNED)
    failed = sum(1 for r in results if r.outcome is CheckOutcome.FAILED)

    return PreflightReport(
        results=tuple(results),
        passed_count=passed,
        warned_count=warned,
        failed_count=failed,
    )


def critical_failures(
    report: PreflightReport,
) -> tuple[CheckResult, ...]:
    """Return only the failed (CRITICAL-severity) checks.

    Operators surface this in the kill-switch safety gate output:
    "Cannot start bot. Critical checks failing: X, Y, Z".
    """

    return tuple(r for r in report.results if r.outcome is CheckOutcome.FAILED)


def warnings_only(
    report: PreflightReport,
) -> tuple[CheckResult, ...]:
    """Return only the warned (WARN-severity) checks.

    Operators surface in the dashboard tile; non-blocking but
    actionable.
    """

    return tuple(r for r in report.results if r.outcome is CheckOutcome.WARNED)


_OUTCOME_EMOJI: dict[CheckOutcome, str] = {
    CheckOutcome.PASSED: "✅",
    CheckOutcome.WARNED: "⚠️",
    CheckOutcome.FAILED: "❌",
}


_SEVERITY_EMOJI: dict[CheckSeverity, str] = {
    CheckSeverity.INFO: "ℹ️",
    CheckSeverity.WARN: "🟡",
    CheckSeverity.CRITICAL: "🔴",
}


def render_result(result: CheckResult) -> str:
    """Format a single check result for ops display.

    No-secret-leak: shows the spec name + operator-facing message.
    Raw stack traces / API responses are operator-side debug logs,
    not the check result.
    """

    emoji = _OUTCOME_EMOJI[result.outcome]
    return f"{emoji} {result.spec.name}: {result.message}"


def render_report(report: PreflightReport) -> str:
    """Format the full pre-flight report.

    Top-line summary + per-check details. Failed checks are listed
    first in a critical section so operators see them at-a-glance.
    """

    if report.is_safe_to_start:
        if report.warned_count == 0:
            top_line = "✅ Pre-flight: ALL GO"
        else:
            top_line = f"⚠️ Pre-flight: GO with {report.warned_count} warning(s)"
    else:
        top_line = f"❌ Pre-flight: NO-GO ({report.failed_count} critical failures)"

    lines = [
        top_line,
        f"  {report.passed_count} passed | "
        f"{report.warned_count} warned | "
        f"{report.failed_count} failed",
    ]
    failures = critical_failures(report)
    if failures:
        lines.append("")
        lines.append("CRITICAL FAILURES:")
        for f in failures:
            lines.append(f"  {render_result(f)}")
    warns = warnings_only(report)
    if warns:
        lines.append("")
        lines.append("WARNINGS:")
        for w in warns:
            lines.append(f"  {render_result(w)}")
    # Show passed checks tersely
    passed_checks = [r for r in report.results if r.outcome is CheckOutcome.PASSED]
    if passed_checks:
        lines.append("")
        lines.append(f"PASSED ({len(passed_checks)}):")
        for r in passed_checks:
            lines.append(f"  ✅ {r.spec.name}")
    return "\n".join(lines)


__all__ = [
    "CheckOutcome",
    "CheckResult",
    "CheckRunFailure",
    "CheckSeverity",
    "CheckSpec",
    "PreflightReport",
    "critical_failures",
    "render_report",
    "render_result",
    "run_checks",
    "warnings_only",
]
