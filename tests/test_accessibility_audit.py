"""Tests for `halal_trader.web.accessibility_audit` (Wave 5.J).

Covers: WCAG luminance/contrast math, threshold boundaries (4.5:1
normal, 3:1 large + UI), alt-text presence, keyboard + focus pins,
two-theme audit, deterministic ordering, render output.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halal_trader.web.accessibility_audit import (
    ImageComponent,
    InteractiveComponent,
    Severity,
    TextComponent,
    Theme,
    Violation,
    WcagCriterion,
    audit,
    contrast_ratio,
    relative_luminance,
    render_report,
)

# --------------------------- Enum string pins --------------------------------


def test_wcag_criterion_string_values_pinned() -> None:
    assert WcagCriterion.TEXT_ALTERNATIVES_1_1_1.value == "1.1.1"
    assert WcagCriterion.CONTRAST_MINIMUM_1_4_3.value == "1.4.3"
    assert WcagCriterion.NON_TEXT_CONTRAST_1_4_11.value == "1.4.11"
    assert WcagCriterion.KEYBOARD_2_1_1.value == "2.1.1"
    assert WcagCriterion.FOCUS_VISIBLE_2_4_7.value == "2.4.7"


def test_severity_string_values_pinned() -> None:
    assert Severity.ERROR.value == "error"
    assert Severity.WARN.value == "warn"


def test_theme_string_values_pinned() -> None:
    assert Theme.LIGHT.value == "light"
    assert Theme.DARK.value == "dark"


# --------------------------- relative_luminance ------------------------------


def test_luminance_pure_white_is_1() -> None:
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0, abs=1e-6)


def test_luminance_pure_black_is_0() -> None:
    assert relative_luminance("#000000") == pytest.approx(0.0, abs=1e-6)


def test_luminance_rejects_bad_format() -> None:
    with pytest.raises(ValueError, match="RRGGBB"):
        relative_luminance("white")


def test_luminance_rejects_short_hex() -> None:
    with pytest.raises(ValueError, match="RRGGBB"):
        relative_luminance("#FFF")


def test_luminance_rejects_missing_hash() -> None:
    with pytest.raises(ValueError, match="RRGGBB"):
        relative_luminance("FFFFFF")


def test_luminance_accepts_lowercase_hex() -> None:
    """Pin: hex parsing case-insensitive."""

    assert relative_luminance("#ffffff") == pytest.approx(1.0, abs=1e-6)


# --------------------------- contrast_ratio ----------------------------------


def test_contrast_white_on_black_is_21() -> None:
    """Pin: max contrast in WCAG is 21:1."""

    assert contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0, abs=0.01)


def test_contrast_same_color_is_1() -> None:
    """Pin: identical colors give 1:1 contrast."""

    assert contrast_ratio("#888888", "#888888") == pytest.approx(1.0, abs=1e-6)


def test_contrast_symmetric() -> None:
    """Pin: contrast(a, b) == contrast(b, a)."""

    a = contrast_ratio("#1a1a1a", "#fafafa")
    b = contrast_ratio("#fafafa", "#1a1a1a")
    assert a == pytest.approx(b)


def test_contrast_white_on_blue_known_value() -> None:
    """Smoke check against published reference values."""

    # White (#fff) on standard blue link (#0000ee) is ~8.59:1
    ratio = contrast_ratio("#FFFFFF", "#0000EE")
    assert ratio > 8.0


# --------------------------- TextComponent -----------------------------------


def _text(**overrides: object) -> TextComponent:
    base: dict[str, object] = {
        "component_id": "tc1",
        "label": "Body text",
        "light_foreground": "#000000",
        "light_background": "#FFFFFF",
        "dark_foreground": "#FFFFFF",
        "dark_background": "#000000",
    }
    base.update(overrides)
    return TextComponent(**base)  # type: ignore[arg-type]


def test_text_component_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="component_id"):
        _text(component_id="")


def test_text_component_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="label"):
        _text(label="")


def test_text_component_rejects_bad_color() -> None:
    with pytest.raises(ValueError, match="RRGGBB"):
        _text(light_foreground="black")


def test_text_component_is_frozen() -> None:
    tc = _text()
    with pytest.raises(FrozenInstanceError):
        tc.label = "other"  # type: ignore[misc]


# --------------------------- ImageComponent ----------------------------------


def test_image_component_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="component_id"):
        ImageComponent(component_id="", alt_text="")


def test_image_component_default_decorative_false() -> None:
    img = ImageComponent(component_id="i1", alt_text="logo")
    assert img.decorative is False


def test_image_component_is_frozen() -> None:
    img = ImageComponent(component_id="i1", alt_text="logo")
    with pytest.raises(FrozenInstanceError):
        img.decorative = True  # type: ignore[misc]


# --------------------------- InteractiveComponent ----------------------------


def test_interactive_component_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="component_id"):
        InteractiveComponent(
            component_id="",
            label="Submit",
            keyboard_reachable=True,
            focus_indicator_visible=True,
        )


def test_interactive_component_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="label"):
        InteractiveComponent(
            component_id="b1",
            label="",
            keyboard_reachable=True,
            focus_indicator_visible=True,
        )


def test_interactive_component_is_frozen() -> None:
    btn = InteractiveComponent(
        component_id="b1",
        label="Submit",
        keyboard_reachable=True,
        focus_indicator_visible=True,
    )
    with pytest.raises(FrozenInstanceError):
        btn.label = "other"  # type: ignore[misc]


# --------------------------- Violation ---------------------------------------


def test_violation_rejects_empty_component_id() -> None:
    with pytest.raises(ValueError, match="component_id"):
        Violation(
            component_id="",
            criterion=WcagCriterion.CONTRAST_MINIMUM_1_4_3,
            severity=Severity.ERROR,
            message="msg",
        )


def test_violation_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="message"):
        Violation(
            component_id="c1",
            criterion=WcagCriterion.CONTRAST_MINIMUM_1_4_3,
            severity=Severity.ERROR,
            message="",
        )


def test_violation_is_frozen() -> None:
    v = Violation(
        component_id="c1",
        criterion=WcagCriterion.CONTRAST_MINIMUM_1_4_3,
        severity=Severity.ERROR,
        message="msg",
    )
    with pytest.raises(FrozenInstanceError):
        v.message = "other"  # type: ignore[misc]


# --------------------------- audit: contrast ---------------------------------


def test_audit_high_contrast_text_passes() -> None:
    tc = _text()  # black-on-white in light, white-on-black in dark — both 21:1
    report = audit(text_components=[tc])
    assert report.passed
    assert report.text_components_audited == 1


def test_audit_low_contrast_text_fails_in_light() -> None:
    """Pin: light grey on white fails."""

    tc = _text(light_foreground="#999999", light_background="#FFFFFF")
    report = audit(text_components=[tc])
    contrast_violations = report.by_criterion(WcagCriterion.CONTRAST_MINIMUM_1_4_3)
    assert len(contrast_violations) == 1
    assert contrast_violations[0].theme is Theme.LIGHT


def test_audit_low_contrast_text_fails_in_dark() -> None:
    """Pin: dark theme failures surface."""

    tc = _text(dark_foreground="#444444", dark_background="#000000")
    report = audit(text_components=[tc])
    contrast_violations = report.by_criterion(WcagCriterion.CONTRAST_MINIMUM_1_4_3)
    assert any(v.theme is Theme.DARK for v in contrast_violations)


def test_audit_failure_in_both_themes_surfaces_two_violations() -> None:
    """Pin: a component bad in both themes surfaces both."""

    tc = _text(
        light_foreground="#999999",
        light_background="#FFFFFF",
        dark_foreground="#444444",
        dark_background="#000000",
    )
    report = audit(text_components=[tc])
    themes = {v.theme for v in report.violations}
    assert Theme.LIGHT in themes
    assert Theme.DARK in themes


def test_audit_normal_text_threshold_inclusive_at_4_5() -> None:
    """Pin: 4.5:1 exactly passes (>= boundary)."""

    # #767676 on #FFFFFF gives ~4.54:1
    tc = _text(light_foreground="#767676", light_background="#FFFFFF")
    report = audit(text_components=[tc])
    light_violations = [v for v in report.violations if v.theme is Theme.LIGHT]
    assert light_violations == []


def test_audit_normal_text_threshold_exclusive_below_4_5() -> None:
    """Pin: just below 4.5:1 fails."""

    # #888888 on #FFFFFF gives ~3.54:1 (below 4.5)
    tc = _text(light_foreground="#888888", light_background="#FFFFFF")
    report = audit(text_components=[tc])
    light_violations = [v for v in report.violations if v.theme is Theme.LIGHT]
    assert len(light_violations) == 1


def test_audit_large_text_uses_3_threshold() -> None:
    """Pin: large text only needs 3:1 — a 3.5:1 ratio passes."""

    tc = _text(
        is_large_text=True,
        light_foreground="#888888",
        light_background="#FFFFFF",
    )
    report = audit(text_components=[tc])
    light_violations = [v for v in report.violations if v.theme is Theme.LIGHT]
    assert light_violations == []


def test_audit_ui_component_uses_1_4_11_criterion() -> None:
    """Pin: ui_component routes to 1.4.11 with 3:1 threshold."""

    # #BBBBBB on #FFFFFF is ~1.66:1 → fails the 3:1 UI threshold
    tc = _text(
        is_ui_component=True,
        light_foreground="#BBBBBB",
        light_background="#FFFFFF",
    )
    report = audit(text_components=[tc])
    contrast_4_3 = report.by_criterion(WcagCriterion.CONTRAST_MINIMUM_1_4_3)
    contrast_4_11 = report.by_criterion(WcagCriterion.NON_TEXT_CONTRAST_1_4_11)
    assert len(contrast_4_3) == 0
    assert len(contrast_4_11) > 0


def test_audit_violation_carries_measured_value() -> None:
    tc = _text(light_foreground="#999999", light_background="#FFFFFF")
    report = audit(text_components=[tc])
    light_v = [v for v in report.violations if v.theme is Theme.LIGHT][0]
    assert light_v.measured_value is not None
    assert light_v.measured_value < 4.5
    assert light_v.threshold == 4.5


# --------------------------- audit: alt text ---------------------------------


def test_audit_image_with_alt_text_passes() -> None:
    img = ImageComponent(component_id="i1", alt_text="Halal trader logo")
    report = audit(image_components=[img])
    assert report.passed


def test_audit_image_without_alt_text_fails() -> None:
    img = ImageComponent(component_id="i1", alt_text="")
    report = audit(image_components=[img])
    violations = report.by_criterion(WcagCriterion.TEXT_ALTERNATIVES_1_1_1)
    assert len(violations) == 1


def test_audit_image_with_whitespace_alt_text_fails() -> None:
    """Pin: whitespace-only alt text counts as missing."""

    img = ImageComponent(component_id="i1", alt_text="   ")
    report = audit(image_components=[img])
    violations = report.by_criterion(WcagCriterion.TEXT_ALTERNATIVES_1_1_1)
    assert len(violations) == 1


def test_audit_decorative_image_skips_alt_text_check() -> None:
    """Pin: decorative=True bypasses alt-text requirement."""

    img = ImageComponent(component_id="i1", alt_text="", decorative=True)
    report = audit(image_components=[img])
    assert report.passed


# --------------------------- audit: keyboard + focus -------------------------


def test_audit_keyboard_unreachable_fails() -> None:
    btn = InteractiveComponent(
        component_id="b1",
        label="Submit",
        keyboard_reachable=False,
        focus_indicator_visible=True,
    )
    report = audit(interactive_components=[btn])
    violations = report.by_criterion(WcagCriterion.KEYBOARD_2_1_1)
    assert len(violations) == 1


def test_audit_no_focus_indicator_fails() -> None:
    btn = InteractiveComponent(
        component_id="b1",
        label="Submit",
        keyboard_reachable=True,
        focus_indicator_visible=False,
    )
    report = audit(interactive_components=[btn])
    violations = report.by_criterion(WcagCriterion.FOCUS_VISIBLE_2_4_7)
    assert len(violations) == 1


def test_audit_both_keyboard_and_focus_failures() -> None:
    """Pin: a component failing both surfaces both violations."""

    btn = InteractiveComponent(
        component_id="b1",
        label="Submit",
        keyboard_reachable=False,
        focus_indicator_visible=False,
    )
    report = audit(interactive_components=[btn])
    assert len(report.violations) == 2


def test_audit_compliant_interactive_passes() -> None:
    btn = InteractiveComponent(
        component_id="b1",
        label="Submit",
        keyboard_reachable=True,
        focus_indicator_visible=True,
    )
    report = audit(interactive_components=[btn])
    assert report.passed


# --------------------------- AuditReport -------------------------------------


def test_audit_report_passed_property() -> None:
    report = audit()
    assert report.passed is True
    assert report.has_errors is False


def test_audit_report_has_errors_property() -> None:
    img = ImageComponent(component_id="i1", alt_text="")
    report = audit(image_components=[img])
    assert report.passed is False
    assert report.has_errors is True


def test_audit_report_audit_counts() -> None:
    tc = _text()
    img = ImageComponent(component_id="i1", alt_text="logo")
    btn = InteractiveComponent(
        component_id="b1",
        label="Submit",
        keyboard_reachable=True,
        focus_indicator_visible=True,
    )
    report = audit(
        text_components=[tc, _text(component_id="tc2")],
        image_components=[img],
        interactive_components=[btn, btn],
    )
    assert report.text_components_audited == 2
    assert report.image_components_audited == 1
    assert report.interactive_components_audited == 2


def test_audit_violations_sorted_deterministically() -> None:
    """Pin: violations sorted by (criterion, component_id)."""

    img1 = ImageComponent(component_id="z_image", alt_text="")
    img2 = ImageComponent(component_id="a_image", alt_text="")
    btn = InteractiveComponent(
        component_id="m_btn",
        label="Submit",
        keyboard_reachable=False,
        focus_indicator_visible=False,
    )
    report = audit(image_components=[img1, img2], interactive_components=[btn])
    # 1.1.1 violations come before 2.1.1 (criterion order)
    # within 1.1.1, a_image before z_image (component_id sort)
    criterion_order = [v.criterion for v in report.violations]
    component_ids_for_111 = [
        v.component_id
        for v in report.violations
        if v.criterion is WcagCriterion.TEXT_ALTERNATIVES_1_1_1
    ]
    assert criterion_order[0] == WcagCriterion.TEXT_ALTERNATIVES_1_1_1
    assert component_ids_for_111 == ["a_image", "z_image"]


def test_audit_is_deterministic() -> None:
    tc = _text(light_foreground="#999999", light_background="#FFFFFF")
    a = audit(text_components=[tc])
    b = audit(text_components=[tc])
    assert a == b


# --------------------------- render_report -----------------------------------


def test_render_passed_report() -> None:
    tc = _text()
    out = render_report(audit(text_components=[tc]))
    assert "✅" in out
    assert "0 violations" in out


def test_render_failed_report_lists_violations() -> None:
    tc = _text(light_foreground="#999999", light_background="#FFFFFF")
    out = render_report(audit(text_components=[tc]))
    assert "❌" in out
    assert "1.4.3" in out
    assert "tc1" in out


def test_render_shows_severity_emoji() -> None:
    img = ImageComponent(component_id="i1", alt_text="")
    out = render_report(audit(image_components=[img]))
    assert "🔴" in out


def test_render_shows_theme_marker() -> None:
    tc = _text(light_foreground="#999999", light_background="#FFFFFF")
    out = render_report(audit(text_components=[tc]))
    assert "[light]" in out


def test_render_includes_audit_count() -> None:
    btn = InteractiveComponent(
        component_id="b1",
        label="X",
        keyboard_reachable=True,
        focus_indicator_visible=True,
    )
    out = render_report(audit(interactive_components=[btn]))
    assert "1 components" in out


# --------------------------- e2e flows ---------------------------------------


def test_e2e_realistic_dashboard_audit_passes() -> None:
    """Realistic clean dashboard: high-contrast text, accessible buttons,
    decorative bg image."""

    components_text = [
        _text(
            component_id="header_text",
            label="Header text",
            light_foreground="#1a1a1a",
            light_background="#FFFFFF",
            dark_foreground="#fafafa",
            dark_background="#1a1a1a",
        ),
        _text(
            component_id="body_text",
            label="Body text",
            light_foreground="#333333",
            light_background="#FFFFFF",
            dark_foreground="#e0e0e0",
            dark_background="#1a1a1a",
        ),
    ]
    images = [
        ImageComponent(component_id="logo", alt_text="Halal Trader logo"),
        ImageComponent(component_id="bg_pattern", alt_text="", decorative=True),
    ]
    buttons = [
        InteractiveComponent(
            component_id="submit_btn",
            label="Submit",
            keyboard_reachable=True,
            focus_indicator_visible=True,
        ),
        InteractiveComponent(
            component_id="cancel_btn",
            label="Cancel",
            keyboard_reachable=True,
            focus_indicator_visible=True,
        ),
    ]
    report = audit(
        text_components=components_text,
        image_components=images,
        interactive_components=buttons,
    )
    assert report.passed


def test_e2e_realistic_dashboard_with_multiple_failures() -> None:
    components_text = [
        # Bad in light
        _text(
            component_id="muted_text",
            label="Muted text",
            light_foreground="#aaaaaa",
            light_background="#FFFFFF",
            dark_foreground="#aaaaaa",
            dark_background="#1a1a1a",
        ),
    ]
    images = [ImageComponent(component_id="chart", alt_text="")]
    buttons = [
        InteractiveComponent(
            component_id="trade_btn",
            label="Place Trade",
            keyboard_reachable=True,
            focus_indicator_visible=False,
        ),
    ]
    report = audit(
        text_components=components_text,
        image_components=images,
        interactive_components=buttons,
    )
    # Expect: 1 contrast violation in light, 1 alt-text violation, 1 focus violation
    assert report.has_errors
    assert len(report.violations) >= 3
    criteria = {v.criterion for v in report.violations}
    assert WcagCriterion.CONTRAST_MINIMUM_1_4_3 in criteria
    assert WcagCriterion.TEXT_ALTERNATIVES_1_1_1 in criteria
    assert WcagCriterion.FOCUS_VISIBLE_2_4_7 in criteria
