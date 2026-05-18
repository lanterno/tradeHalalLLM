"""Settings drift detector.

Auxiliary primitive for the operator's at-a-glance "what have I
tuned away from defaults?" audit. The bot's `Settings` model has
~80+ knobs (LLM provider, broker keys, tier caps, risk thresholds,
etc.); the operator typically tunes 5-10 for their specific
deployment. This module is the **pure-Python diff engine** that
takes the canonical-defaults catalogue + the running settings
values and produces a structured drift report.

Picked a focused detector over operators reading their .env file
because (a) the .env file shows what the operator set, but not
what differs from the codebase default — a value of "0.05" might
be the default OR an aggressive override; the drift report makes
the distinction explicit; (b) magnitude classification (TINY /
MODERATE / SIGNIFICANT / EXTREME) lets the dashboard prioritize
which drifts matter — a 0.001 lr override is irrelevant; a 50%
position cap is dangerous; (c) numeric bounds catch operator
typos at audit time — a bot configured with max_position_pct=5.0
(meaning 500%) is an obvious mistake the bounds detection
surfaces.

Pinned semantics:
- **DriftMagnitude is closed-set: TINY / MODERATE / SIGNIFICANT
  / EXTREME.** Magnitude bands relative to default; for booleans
  any flip is at least MODERATE.
- **NumericBounds optional but enforced if set.** A setting with
  bounds (e.g. `max_position_pct: [0.001, 0.5]`) flags
  out-of-bounds values as EXTREME plus surfaces a warning.
- **Settings catalogue is closed.** Only catalogued settings
  appear in the drift report — operator-defined custom env vars
  are silently ignored to avoid noise.
- **Render output never includes API keys, broker secrets, or
  vault references.** Sensitive settings are explicitly tagged
  `is_secret=True` and render as `<secret>` placeholder.
- **Reproducible across runs.** Same (catalogue, current_values)
  → same DriftReport.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DriftMagnitude(str, Enum):
    """Magnitude tiers for drift classification.

    Pinned string values for JSON / DB stability.
    """

    NONE = "none"  # value matches default
    TINY = "tiny"  # within 10% of default (numeric)
    MODERATE = "moderate"  # 10-50% drift / boolean flip / string change
    SIGNIFICANT = "significant"  # 50-100% drift
    EXTREME = "extreme"  # >100% drift OR out of bounds


@dataclass(frozen=True)
class NumericBounds:
    """Optional numeric bounds for a setting."""

    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if self.max_value <= self.min_value:
            raise ValueError(f"max_value {self.max_value} must be > min_value {self.min_value}")

    def contains(self, value: float) -> bool:
        """True if value within [min, max] inclusive."""

        return self.min_value <= value <= self.max_value


@dataclass(frozen=True)
class SettingSpec:
    """Catalogued setting with default + optional bounds.

    `is_secret=True` means render output uses `<secret>` placeholder
    rather than the actual value (the comparator still runs against
    the actual value to detect drift).
    """

    name: str
    default_value: Any
    description: str
    is_secret: bool = False
    bounds: NumericBounds | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        # Bounds only make sense for numeric defaults
        if self.bounds is not None and not isinstance(self.default_value, (int, float)):
            raise ValueError(f"bounds set on non-numeric setting {self.name!r}")
        # If bounds are set, default must be within them
        if self.bounds is not None:
            if not self.bounds.contains(float(self.default_value)):
                raise ValueError(
                    f"default {self.default_value} for {self.name!r} "
                    f"out of bounds [{self.bounds.min_value}, "
                    f"{self.bounds.max_value}]"
                )


@dataclass(frozen=True)
class DriftEntry:
    """One setting's drift status."""

    name: str
    default_value: Any
    current_value: Any
    magnitude: DriftMagnitude
    out_of_bounds: bool
    is_secret: bool

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")


@dataclass(frozen=True)
class DriftReport:
    """Aggregate drift report across all catalogued settings."""

    entries: tuple[DriftEntry, ...]
    total_settings: int
    drifted_count: int
    out_of_bounds_count: int

    def __post_init__(self) -> None:
        if self.total_settings < 0:
            raise ValueError("total_settings must be non-negative")
        if self.drifted_count < 0:
            raise ValueError("drifted_count must be non-negative")
        if self.drifted_count > self.total_settings:
            raise ValueError("drifted_count cannot exceed total_settings")
        if self.out_of_bounds_count < 0:
            raise ValueError("out_of_bounds_count must be non-negative")
        if self.out_of_bounds_count > self.drifted_count:
            raise ValueError(
                "out_of_bounds_count cannot exceed drifted_count "
                "(out-of-bounds is a subset of drifted)"
            )


def _classify_numeric_drift(
    *, default: float, current: float, bounds: NumericBounds | None
) -> tuple[DriftMagnitude, bool]:
    """Classify magnitude + out-of-bounds for a numeric drift.

    Returns (magnitude, out_of_bounds).
    """

    out_of_bounds = bounds is not None and not bounds.contains(current)

    if default == current:
        return DriftMagnitude.NONE, out_of_bounds

    # Use absolute drift relative to |default|, with safe handling
    # for default == 0
    if default == 0:
        # Any non-zero current is at least MODERATE drift from 0
        # (arbitrary threshold: 1.0 → SIGNIFICANT, 10.0 → EXTREME)
        abs_current = abs(current)
        if out_of_bounds or abs_current >= 10.0:
            return DriftMagnitude.EXTREME, out_of_bounds
        if abs_current >= 1.0:
            return DriftMagnitude.SIGNIFICANT, out_of_bounds
        if abs_current >= 0.1:
            return DriftMagnitude.MODERATE, out_of_bounds
        return DriftMagnitude.TINY, out_of_bounds

    rel_drift = abs(current - default) / abs(default)
    if out_of_bounds:
        return DriftMagnitude.EXTREME, out_of_bounds
    if rel_drift > 1.0:
        return DriftMagnitude.EXTREME, False
    if rel_drift > 0.5:
        return DriftMagnitude.SIGNIFICANT, False
    if rel_drift > 0.1:
        return DriftMagnitude.MODERATE, False
    return DriftMagnitude.TINY, False


def _classify_boolean_drift(*, default: bool, current: bool) -> DriftMagnitude:
    """Boolean flips are at least MODERATE."""

    if default == current:
        return DriftMagnitude.NONE
    return DriftMagnitude.MODERATE


def _classify_string_drift(*, default: str, current: str) -> DriftMagnitude:
    """String changes default to MODERATE; empty-vs-set defaults
    to SIGNIFICANT."""

    if default == current:
        return DriftMagnitude.NONE
    # Empty → set or set → empty is significant
    if not default or not current:
        return DriftMagnitude.SIGNIFICANT
    return DriftMagnitude.MODERATE


def detect_drift_for_setting(spec: SettingSpec, current_value: Any) -> DriftEntry:
    """Compute drift for a single setting."""

    out_of_bounds = False
    if isinstance(spec.default_value, bool):
        if not isinstance(current_value, bool):
            raise TypeError(
                f"setting {spec.name!r} expected bool; got {type(current_value).__name__}"
            )
        magnitude = _classify_boolean_drift(default=spec.default_value, current=current_value)
    elif isinstance(spec.default_value, (int, float)):
        if not isinstance(current_value, (int, float)) or isinstance(current_value, bool):
            raise TypeError(
                f"setting {spec.name!r} expected numeric; got {type(current_value).__name__}"
            )
        magnitude, out_of_bounds = _classify_numeric_drift(
            default=float(spec.default_value),
            current=float(current_value),
            bounds=spec.bounds,
        )
    elif isinstance(spec.default_value, str):
        if not isinstance(current_value, str):
            raise TypeError(
                f"setting {spec.name!r} expected str; got {type(current_value).__name__}"
            )
        magnitude = _classify_string_drift(default=spec.default_value, current=current_value)
    else:
        raise TypeError(
            f"setting {spec.name!r} default has unsupported type "
            f"{type(spec.default_value).__name__}"
        )

    return DriftEntry(
        name=spec.name,
        default_value=spec.default_value,
        current_value=current_value,
        magnitude=magnitude,
        out_of_bounds=out_of_bounds,
        is_secret=spec.is_secret,
    )


def build_drift_report(
    catalogue: Iterable[SettingSpec],
    current_values: dict[str, Any],
) -> DriftReport:
    """Build the full drift report.

    Settings in the catalogue but missing from `current_values` raise
    KeyError — operators must explicitly provide every catalogued
    value (no silent defaults). Operator-defined env vars not in the
    catalogue are silently ignored (per the closed-catalogue pin).

    The returned `entries` tuple is sorted by setting name for
    deterministic display.
    """

    entries: list[DriftEntry] = []
    catalogue_list = list(catalogue)
    for spec in catalogue_list:
        if spec.name not in current_values:
            raise KeyError(f"setting {spec.name!r} required but missing from current_values")
        entries.append(detect_drift_for_setting(spec, current_values[spec.name]))

    entries.sort(key=lambda e: e.name)
    drifted = [e for e in entries if e.magnitude is not DriftMagnitude.NONE]
    out_of_bounds = [e for e in drifted if e.out_of_bounds]

    return DriftReport(
        entries=tuple(entries),
        total_settings=len(catalogue_list),
        drifted_count=len(drifted),
        out_of_bounds_count=len(out_of_bounds),
    )


def filter_drifted(
    report: DriftReport,
    *,
    minimum: DriftMagnitude = DriftMagnitude.TINY,
) -> tuple[DriftEntry, ...]:
    """Return only entries at or above the minimum magnitude.

    Magnitude ordering: NONE < TINY < MODERATE < SIGNIFICANT < EXTREME.
    """

    order = [
        DriftMagnitude.NONE,
        DriftMagnitude.TINY,
        DriftMagnitude.MODERATE,
        DriftMagnitude.SIGNIFICANT,
        DriftMagnitude.EXTREME,
    ]
    min_idx = order.index(minimum)
    return tuple(e for e in report.entries if order.index(e.magnitude) >= min_idx)


_MAGNITUDE_EMOJI: dict[DriftMagnitude, str] = {
    DriftMagnitude.NONE: "✅",
    DriftMagnitude.TINY: "🟢",
    DriftMagnitude.MODERATE: "🟡",
    DriftMagnitude.SIGNIFICANT: "🟠",
    DriftMagnitude.EXTREME: "🔴",
}


def _render_value(value: Any, *, is_secret: bool) -> str:
    if is_secret:
        return "<secret>"
    if isinstance(value, str):
        # Truncate long strings for display
        if len(value) > 40:
            return f"{value[:37]!r}..."
        return repr(value)
    return str(value)


def render_entry(entry: DriftEntry) -> str:
    """Format one drift entry for ops display.

    No-secret-leak: secrets render as `<secret>` placeholder. The
    magnitude + out-of-bounds flag still surface even for secrets so
    operators know "this is non-default" without seeing the value.
    """

    emoji = _MAGNITUDE_EMOJI[entry.magnitude]
    bounds_marker = " ⚠️ OUT OF BOUNDS" if entry.out_of_bounds else ""
    default_str = _render_value(entry.default_value, is_secret=entry.is_secret)
    current_str = _render_value(entry.current_value, is_secret=entry.is_secret)
    return (
        f"{emoji} {entry.name}: {current_str} (default: {default_str}) "
        f"— {entry.magnitude.value}{bounds_marker}"
    )


def render_report(report: DriftReport) -> str:
    """Format the full drift report.

    Shows summary counts + only the drifted entries (NONE entries
    omitted to keep the report focused on what changed).
    """

    lines = [
        "⚙️ Settings drift report",
        f"  total: {report.total_settings} settings",
        f"  drifted: {report.drifted_count}",
        f"  out of bounds: {report.out_of_bounds_count}",
    ]
    drifted = filter_drifted(report)
    if not drifted:
        lines.append("")
        lines.append("  ✅ no drift detected")
        return "\n".join(lines)
    lines.append("")
    lines.append("Drifted settings:")
    for entry in drifted:
        lines.append(f"  {render_entry(entry)}")
    return "\n".join(lines)


__all__ = [
    "DriftEntry",
    "DriftMagnitude",
    "DriftReport",
    "NumericBounds",
    "SettingSpec",
    "build_drift_report",
    "detect_drift_for_setting",
    "filter_drifted",
    "render_entry",
    "render_report",
]
