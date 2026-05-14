"""Accessibility (WCAG AA) audit engine.

The roadmap pins Wave 5.J: "Add dark mode + run a full WCAG AA
audit (alt text, color contrast, keyboard nav). Halal traders work
odd hours; dark mode is non-negotiable." This module is the
**pure-Python audit engine** that consumes a static UI manifest
(per-component descriptors of color foreground/background,
alt-text presence, keyboard reachability, focus indicators) and
returns a structured set of WCAG AA violations.

Picked a focused audit engine over a "manual checklist" approach
because (a) WCAG AA contrast ratios (4.5:1 normal, 3:1 large) are
arithmetic — running the math at CI lets us regression-pin every
component's contrast against silent theme drift, (b) the dark-mode
roadmap line means *every* color decision applies under two themes
and a pure function of (foreground, background) → ratio means we
audit both themes from the same component manifest, (c) the
acceptance criterion "operator can identify a losing trade in
under 30 seconds" requires keyboard navigability + focus indicator
+ readable contrast — auditing those properties together rather
than three separate ad-hoc scripts means a regression in any of
them is surfaced at the same place.

Pinned semantics:
- **Contrast ratio uses the WCAG 1.4.3 luminance formula.** Normal
  text needs >= 4.5:1; large text (18pt+ or 14pt+ bold) needs
  >= 3:1. UI components (icons, focus indicators) need >= 3:1
  per WCAG 1.4.11. Boundaries inclusive at 4.5 / 3.0.
- **Alt text required for non-decorative images.** Decorative
  images (purely visual flourish) are explicitly tagged
  `decorative=True` and skip the alt-text check; otherwise
  empty alt text is a 1.1.1 violation.
- **Every interactive component must be keyboard-reachable + have
  a focus indicator.** WCAG 2.1.1 + 2.4.7 — pinned together
  because failing either makes the bot's dashboard unusable for
  keyboard-only operators (a common case for senior operators).
- **Two themes audited together.** A component's `light_*` and
  `dark_*` colors are both checked; a violation in either theme
  surfaces.
- **Render output never includes operator-identifying data.**
  The audit is over UI components, not user data; mirrors no-PII
  patterns of upstream waves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


_NORMAL_TEXT_THRESHOLD = 4.5
_LARGE_TEXT_THRESHOLD = 3.0
_UI_COMPONENT_THRESHOLD = 3.0


class WcagCriterion(str, Enum):
    """The WCAG AA criteria this engine checks.

    Pinned string values for JSON / DB / Lighthouse-export stability.
    """

    TEXT_ALTERNATIVES_1_1_1 = "1.1.1"  # Non-text content has alt text
    CONTRAST_MINIMUM_1_4_3 = "1.4.3"  # 4.5:1 normal / 3:1 large
    NON_TEXT_CONTRAST_1_4_11 = "1.4.11"  # 3:1 for UI components
    KEYBOARD_2_1_1 = "2.1.1"  # All functionality via keyboard
    FOCUS_VISIBLE_2_4_7 = "2.4.7"  # Visible focus indicator


class Severity(str, Enum):
    """Violation severity. AA failures are ERROR; advisory warnings are WARN."""

    ERROR = "error"
    WARN = "warn"


class Theme(str, Enum):
    """The two themes audited together."""

    LIGHT = "light"
    DARK = "dark"


def _parse_hex(color: str) -> tuple[int, int, int]:
    """Parse a `#RRGGBB` string to (r, g, b) ints."""

    if not _HEX_RE.match(color):
        raise ValueError(f"color must be in #RRGGBB form, got {color!r}")
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return r, g, b


def _channel_luminance(channel_byte: int) -> float:
    """Per the WCAG 1.4.3 luminance formula."""

    c = channel_byte / 255.0
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    """Compute the relative luminance of a hex color per WCAG."""

    r, g, b = _parse_hex(color)
    return (
        0.2126 * _channel_luminance(r)
        + 0.7152 * _channel_luminance(g)
        + 0.0722 * _channel_luminance(b)
    )


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 1.4.3 contrast ratio, in [1.0, 21.0]."""

    l1 = relative_luminance(foreground)
    l2 = relative_luminance(background)
    lighter, darker = (l1, l2) if l1 > l2 else (l2, l1)
    return (lighter + 0.05) / (darker + 0.05)


@dataclass(frozen=True)
class TextComponent:
    """A text-bearing UI component to audit.

    `is_large_text` covers 18pt+ regular or 14pt+ bold per WCAG.
    `is_ui_component` (rather than text content) routes to the 3:1
    UI-contrast threshold of WCAG 1.4.11 (used for icons, separators,
    focus indicators).
    """

    component_id: str
    label: str
    light_foreground: str
    light_background: str
    dark_foreground: str
    dark_background: str
    is_large_text: bool = False
    is_ui_component: bool = False

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not self.label or not self.label.strip():
            raise ValueError("label must be non-empty")
        for color in (
            self.light_foreground,
            self.light_background,
            self.dark_foreground,
            self.dark_background,
        ):
            if not _HEX_RE.match(color):
                raise ValueError(f"color {color!r} must be in #RRGGBB form")


@dataclass(frozen=True)
class ImageComponent:
    """An image / icon UI component to audit.

    `decorative=True` images skip the alt-text check (e.g. a
    purely visual flourish behind text); decorative=False without
    alt text is a 1.1.1 violation.
    """

    component_id: str
    alt_text: str
    decorative: bool = False

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id must be non-empty")


@dataclass(frozen=True)
class InteractiveComponent:
    """An interactive (button / link / input) UI component to audit.

    `keyboard_reachable=True` confirms the component is in the
    tab order; `focus_indicator_visible=True` confirms a visible
    focus ring is rendered.
    """

    component_id: str
    label: str
    keyboard_reachable: bool
    focus_indicator_visible: bool

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not self.label or not self.label.strip():
            raise ValueError("label must be non-empty")


@dataclass(frozen=True)
class Violation:
    """One audit finding."""

    component_id: str
    criterion: WcagCriterion
    severity: Severity
    message: str
    theme: Theme | None = None  # None means theme-agnostic violation
    measured_value: float | None = None  # e.g. the contrast ratio
    threshold: float | None = None  # e.g. 4.5

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")


@dataclass(frozen=True)
class AuditReport:
    """The full audit-report view-model.

    `violations` is sorted by (criterion, component_id) for
    deterministic ordering across renders.
    """

    text_components_audited: int
    image_components_audited: int
    interactive_components_audited: int
    violations: tuple[Violation, ...]

    @property
    def has_errors(self) -> bool:
        return any(v.severity is Severity.ERROR for v in self.violations)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def by_criterion(self, criterion: WcagCriterion) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.criterion is criterion)


def _audit_text_component(component: TextComponent) -> list[Violation]:
    out: list[Violation] = []
    if component.is_ui_component:
        threshold = _UI_COMPONENT_THRESHOLD
        criterion = WcagCriterion.NON_TEXT_CONTRAST_1_4_11
    elif component.is_large_text:
        threshold = _LARGE_TEXT_THRESHOLD
        criterion = WcagCriterion.CONTRAST_MINIMUM_1_4_3
    else:
        threshold = _NORMAL_TEXT_THRESHOLD
        criterion = WcagCriterion.CONTRAST_MINIMUM_1_4_3

    for theme, fg, bg in (
        (Theme.LIGHT, component.light_foreground, component.light_background),
        (Theme.DARK, component.dark_foreground, component.dark_background),
    ):
        ratio = contrast_ratio(fg, bg)
        if ratio < threshold:
            out.append(
                Violation(
                    component_id=component.component_id,
                    criterion=criterion,
                    severity=Severity.ERROR,
                    message=(
                        f"{theme.value} contrast {ratio:.2f}:1 below {threshold}:1 "
                        f"for {component.label}"
                    ),
                    theme=theme,
                    measured_value=ratio,
                    threshold=threshold,
                )
            )
    return out


def _audit_image_component(component: ImageComponent) -> list[Violation]:
    if component.decorative:
        return []
    if not component.alt_text or not component.alt_text.strip():
        return [
            Violation(
                component_id=component.component_id,
                criterion=WcagCriterion.TEXT_ALTERNATIVES_1_1_1,
                severity=Severity.ERROR,
                message="non-decorative image missing alt text",
            )
        ]
    return []


def _audit_interactive_component(
    component: InteractiveComponent,
) -> list[Violation]:
    out: list[Violation] = []
    if not component.keyboard_reachable:
        out.append(
            Violation(
                component_id=component.component_id,
                criterion=WcagCriterion.KEYBOARD_2_1_1,
                severity=Severity.ERROR,
                message=f"{component.label} not reachable via keyboard",
            )
        )
    if not component.focus_indicator_visible:
        out.append(
            Violation(
                component_id=component.component_id,
                criterion=WcagCriterion.FOCUS_VISIBLE_2_4_7,
                severity=Severity.ERROR,
                message=f"{component.label} has no visible focus indicator",
            )
        )
    return out


def audit(
    *,
    text_components: Iterable[TextComponent] = (),
    image_components: Iterable[ImageComponent] = (),
    interactive_components: Iterable[InteractiveComponent] = (),
) -> AuditReport:
    """Run the full WCAG AA audit and return a report.

    Pure: deterministic for a given input. The report's
    `violations` tuple is sorted by `(criterion, component_id)`
    so dashboards render the same row order across runs.
    """

    text_list = list(text_components)
    image_list = list(image_components)
    interactive_list = list(interactive_components)

    raw: list[Violation] = []
    for tc in text_list:
        raw.extend(_audit_text_component(tc))
    for ic in image_list:
        raw.extend(_audit_image_component(ic))
    for ac in interactive_list:
        raw.extend(_audit_interactive_component(ac))

    raw.sort(key=lambda v: (v.criterion.value, v.component_id))

    return AuditReport(
        text_components_audited=len(text_list),
        image_components_audited=len(image_list),
        interactive_components_audited=len(interactive_list),
        violations=tuple(raw),
    )


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.ERROR: "🔴",
    Severity.WARN: "🟡",
}


def render_report(report: AuditReport) -> str:
    """Format the audit report for ops display.

    No-secret-leak: the engine works on UI component descriptors
    only — never sees user data — so render carries no PII risk
    by construction. The render is meant for CI logs + the
    operator dashboard's accessibility tile.
    """

    total = (
        report.text_components_audited
        + report.image_components_audited
        + report.interactive_components_audited
    )
    if report.passed:
        return f"♿ Accessibility audit ✅ — {total} components, 0 violations"

    lines = [f"♿ Accessibility audit ❌ — {total} components, {len(report.violations)} violations"]
    for v in report.violations:
        emoji = _SEVERITY_EMOJI[v.severity]
        theme_marker = f" [{v.theme.value}]" if v.theme is not None else ""
        lines.append(f"  {emoji} {v.criterion.value}{theme_marker} {v.component_id}: {v.message}")
    return "\n".join(lines)


__all__ = [
    "AuditReport",
    "ImageComponent",
    "InteractiveComponent",
    "Severity",
    "TextComponent",
    "Theme",
    "Violation",
    "WcagCriterion",
    "audit",
    "contrast_ratio",
    "relative_luminance",
    "render_report",
]
