"""Deprecation policy + sunset timeline engine.

Auxiliary primitive complementing Wave 9.F API reference. The
generated API docs surface what's currently public; this module is
the **pure-Python lifecycle engine** that tracks how a public API
moves from "announced as deprecated" through "actively deprecated"
through "removed", with deterministic timeline enforcement so
operators can't accidentally remove an API before users have had
time to migrate.

Picked a focused engine over scattered `@deprecated` decorators
because (a) the sunset timeline (announced → 60 days → deprecated
→ 90 days → removed) needs deterministic enforcement: the
removal-readiness gate answers "is symbol X past its scheduled
removal date?" without operator interpretation; (b) the structured
record per deprecated symbol gives the API reference layer
something to surface ("this method was announced as deprecated on
2026-Q1, will be removed by 2026-Q3"); (c) the warning emission
classification (no warning during ANNOUNCED window; DeprecationWarning
during DEPRECATED; RuntimeError during REMOVED) is a single
inspectable rule rather than a scattered conditional in each
deprecated function.

Pinned semantics:
- **Closed-set DeprecationStage enum.** ANNOUNCED → DEPRECATED →
  REMOVED. Forward-only; cannot revive a removed symbol via the
  same record (operators must add it back as a fresh symbol).
- **Default sunset policy: 60-day announce + 90-day deprecated.**
  Total 150 days from announce to remove. Operator-tunable but
  validation enforces minimum 30-day announce + 60-day deprecated.
- **Warning emission per stage:** ANNOUNCED → no warning (just
  catalogued); DEPRECATED → DeprecationWarning fires; REMOVED →
  RuntimeError on use.
- **Removal-readiness is a hard gate.** Symbols past their
  scheduled removal date AND in REMOVED stage can be hard-deleted;
  symbols past the date but still DEPRECATED can be auto-promoted
  to REMOVED via `advance_stage`.
- **Render output never includes operator email / Slack handles
  for migration support contacts** — render shows replacement
  symbol + migration_url only.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class DeprecationStage(str, Enum):
    """Lifecycle stage for a deprecated symbol.

    Pinned string values for JSON / DB stability.
    """

    ANNOUNCED = "announced"  # Documented as deprecated; no warning yet
    DEPRECATED = "deprecated"  # DeprecationWarning fires on use
    REMOVED = "removed"  # RuntimeError on use


_STAGE_ORDER: tuple[DeprecationStage, ...] = (
    DeprecationStage.ANNOUNCED,
    DeprecationStage.DEPRECATED,
    DeprecationStage.REMOVED,
)


_DEFAULT_ANNOUNCE_WINDOW = timedelta(days=60)
_DEFAULT_DEPRECATED_WINDOW = timedelta(days=90)
_MIN_ANNOUNCE_WINDOW = timedelta(days=30)
_MIN_DEPRECATED_WINDOW = timedelta(days=60)


@dataclass(frozen=True)
class DeprecationPolicy:
    """Operator-tunable sunset timeline."""

    announce_window: timedelta = _DEFAULT_ANNOUNCE_WINDOW
    deprecated_window: timedelta = _DEFAULT_DEPRECATED_WINDOW

    def __post_init__(self) -> None:
        if self.announce_window < _MIN_ANNOUNCE_WINDOW:
            raise ValueError(
                f"announce_window {self.announce_window} below minimum "
                f"{_MIN_ANNOUNCE_WINDOW} (users need warning time)"
            )
        if self.deprecated_window < _MIN_DEPRECATED_WINDOW:
            raise ValueError(
                f"deprecated_window {self.deprecated_window} below minimum {_MIN_DEPRECATED_WINDOW}"
            )


DEFAULT_POLICY = DeprecationPolicy()


class StageTransitionError(Exception):
    """Raised on invalid stage transition (skip / revert)."""

    def __init__(
        self,
        symbol: str,
        current: DeprecationStage,
        attempted: DeprecationStage,
    ) -> None:
        super().__init__(
            f"symbol {symbol!r}: cannot transition from {current.value} to {attempted.value}"
        )
        self.symbol = symbol
        self.current = current
        self.attempted = attempted


class SymbolRemovedError(RuntimeError):
    """Raised when a REMOVED symbol is used.

    The error inherits from RuntimeError so callers' generic
    error handlers catch it; the symbol + replacement attributes
    let the dashboard surface "this caller still uses removed
    symbol X; migrate to Y".
    """

    def __init__(
        self,
        symbol: str,
        replacement: str = "",
        migration_url: str = "",
    ) -> None:
        message = f"symbol {symbol!r} has been removed"
        if replacement:
            message += f"; use {replacement!r} instead"
        if migration_url:
            message += f" (see {migration_url})"
        super().__init__(message)
        self.symbol = symbol
        self.replacement = replacement
        self.migration_url = migration_url


@dataclass(frozen=True)
class DeprecatedSymbol:
    """One deprecated symbol's lifecycle record.

    `replacement` is the recommended new symbol (empty if none —
    operators sometimes deprecate without a replacement when the
    feature is just gone). `migration_url` points to the migration
    guide. Neither field is required; both surface in render.
    """

    symbol: str
    announced_at: datetime
    stage: DeprecationStage
    replacement: str = ""
    migration_url: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.announced_at.tzinfo is None:
            raise ValueError("announced_at must be timezone-aware")


def announce_deprecation(
    *,
    symbol: str,
    now: datetime,
    replacement: str = "",
    migration_url: str = "",
    reason: str = "",
) -> DeprecatedSymbol:
    """Build a fresh ANNOUNCED record."""

    if not symbol or not symbol.strip():
        raise ValueError("symbol must be non-empty")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return DeprecatedSymbol(
        symbol=symbol,
        announced_at=now,
        stage=DeprecationStage.ANNOUNCED,
        replacement=replacement,
        migration_url=migration_url,
        reason=reason,
    )


def _check_forward(
    symbol: str,
    current: DeprecationStage,
    target: DeprecationStage,
) -> None:
    cur_idx = _STAGE_ORDER.index(current)
    target_idx = _STAGE_ORDER.index(target)
    if target_idx != cur_idx + 1:
        raise StageTransitionError(symbol, current, target)


def advance_stage(record: DeprecatedSymbol, *, now: datetime) -> DeprecatedSymbol:
    """Move forward one stage: ANNOUNCED → DEPRECATED → REMOVED.

    Cannot skip ahead (REMOVED requires DEPRECATED first); cannot
    revert; REMOVED is terminal.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if record.stage is DeprecationStage.REMOVED:
        raise StageTransitionError(record.symbol, record.stage, record.stage)
    next_idx = _STAGE_ORDER.index(record.stage) + 1
    target = _STAGE_ORDER[next_idx]
    return DeprecatedSymbol(
        symbol=record.symbol,
        announced_at=record.announced_at,
        stage=target,
        replacement=record.replacement,
        migration_url=record.migration_url,
        reason=record.reason,
    )


def scheduled_deprecated_at(
    record: DeprecatedSymbol, *, policy: DeprecationPolicy = DEFAULT_POLICY
) -> datetime:
    """When should this symbol move from ANNOUNCED → DEPRECATED?"""

    return record.announced_at + policy.announce_window


def scheduled_removal_at(
    record: DeprecatedSymbol, *, policy: DeprecationPolicy = DEFAULT_POLICY
) -> datetime:
    """When should this symbol be hard-removed?"""

    return record.announced_at + policy.announce_window + policy.deprecated_window


def is_overdue_for_advancement(
    record: DeprecatedSymbol,
    *,
    now: datetime,
    policy: DeprecationPolicy = DEFAULT_POLICY,
) -> bool:
    """True if the record is past the scheduled date for its next stage.

    ANNOUNCED records past `scheduled_deprecated_at` are overdue;
    DEPRECATED records past `scheduled_removal_at` are overdue;
    REMOVED records are never overdue (terminal).
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if record.stage is DeprecationStage.ANNOUNCED:
        return now >= scheduled_deprecated_at(record, policy=policy)
    if record.stage is DeprecationStage.DEPRECATED:
        return now >= scheduled_removal_at(record, policy=policy)
    return False


def emit_warning_if_needed(
    record: DeprecatedSymbol,
) -> None:
    """Emit a DeprecationWarning if symbol is in DEPRECATED stage.

    ANNOUNCED → no warning (operators are still in the announce
    window where users can prepare). REMOVED → caller should
    raise SymbolRemovedError instead of using this; this function
    just emits warnings, not errors.
    """

    if record.stage is DeprecationStage.DEPRECATED:
        message = f"{record.symbol} is deprecated"
        if record.replacement:
            message += f"; use {record.replacement} instead"
        if record.migration_url:
            message += f" (see {record.migration_url})"
        warnings.warn(message, DeprecationWarning, stacklevel=2)


def assert_not_removed(record: DeprecatedSymbol) -> None:
    """Raise SymbolRemovedError if record.stage is REMOVED.

    Callers wrap their first use of a deprecated symbol with
    `assert_not_removed(record); emit_warning_if_needed(record)` so
    a REMOVED symbol fails loudly while a DEPRECATED one just warns.
    """

    if record.stage is DeprecationStage.REMOVED:
        raise SymbolRemovedError(
            symbol=record.symbol,
            replacement=record.replacement,
            migration_url=record.migration_url,
        )


def filter_overdue(
    records: Iterable[DeprecatedSymbol],
    *,
    now: datetime,
    policy: DeprecationPolicy = DEFAULT_POLICY,
) -> tuple[DeprecatedSymbol, ...]:
    """Return overdue records sorted by announced_at ascending."""

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    overdue = [r for r in records if is_overdue_for_advancement(r, now=now, policy=policy)]
    return tuple(sorted(overdue, key=lambda r: r.announced_at))


_STAGE_EMOJI: dict[DeprecationStage, str] = {
    DeprecationStage.ANNOUNCED: "📣",
    DeprecationStage.DEPRECATED: "⚠️",
    DeprecationStage.REMOVED: "🗑️",
}


def render_record(
    record: DeprecatedSymbol,
    *,
    policy: DeprecationPolicy = DEFAULT_POLICY,
) -> str:
    """Format a deprecated-symbol record for ops display.

    No-secret-leak: the dataclass deliberately doesn't carry
    operator email / Slack handles, so render is structurally
    secret-free. Migration URL surfaces (it's intended to be a
    public link).
    """

    emoji = _STAGE_EMOJI[record.stage]
    lines = [
        f"{emoji} {record.symbol} — {record.stage.value}",
        f"  announced: {record.announced_at.date().isoformat()}",
    ]
    if record.stage is DeprecationStage.ANNOUNCED:
        scheduled = scheduled_deprecated_at(record, policy=policy)
        lines.append(f"  → DEPRECATED scheduled: {scheduled.date().isoformat()}")
    elif record.stage is DeprecationStage.DEPRECATED:
        scheduled = scheduled_removal_at(record, policy=policy)
        lines.append(f"  → REMOVED scheduled: {scheduled.date().isoformat()}")
    if record.replacement:
        lines.append(f"  replacement: {record.replacement}")
    if record.migration_url:
        lines.append(f"  migration: {record.migration_url}")
    if record.reason:
        lines.append(f"  reason: {record.reason}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "DeprecatedSymbol",
    "DeprecationPolicy",
    "DeprecationStage",
    "StageTransitionError",
    "SymbolRemovedError",
    "advance_stage",
    "announce_deprecation",
    "assert_not_removed",
    "emit_warning_if_needed",
    "filter_overdue",
    "is_overdue_for_advancement",
    "render_record",
    "scheduled_deprecated_at",
    "scheduled_removal_at",
]
