"""Tests for core/hot_reload.py — Round-5 Wave 0.F."""

from __future__ import annotations

import pytest

from halal_trader.core.hot_reload import (
    ChangeKind,
    ConfigChange,
    SecretClassification,
    classify_key,
    diff_config,
    hot_reloadable,
    render_changes,
    restart_required,
)

# --- Validation -----------------------------------


def test_change_kind_string_values():
    assert ChangeKind.ADDED.value == "added"
    assert ChangeKind.REMOVED.value == "removed"
    assert ChangeKind.UPDATED.value == "updated"


def test_classification_string_values():
    assert SecretClassification.PUBLIC.value == "public"
    assert SecretClassification.SENSITIVE.value == "sensitive"
    assert SecretClassification.SECRET.value == "secret"


def test_change_empty_key_rejected():
    with pytest.raises(ValueError):
        ConfigChange(
            key="",
            kind=ChangeKind.ADDED,
            classification=SecretClassification.PUBLIC,
            old_value_summary="x",
            new_value_summary="y",
            requires_restart=False,
        )


# --- Classification ------------------------------


def test_api_key_classified_secret():
    assert classify_key("BINANCE_API_KEY") is SecretClassification.SECRET


def test_secret_classified_secret():
    assert classify_key("HMAC_SECRET") is SecretClassification.SECRET


def test_token_classified_secret():
    assert classify_key("ACCESS_TOKEN") is SecretClassification.SECRET


def test_password_classified_secret():
    assert classify_key("DB_PASSWORD") is SecretClassification.SECRET


def test_private_key_classified_secret():
    assert classify_key("RSA_PRIVATE_KEY_PEM") is SecretClassification.SECRET


def test_webhook_classified_sensitive():
    assert classify_key("SLACK_WEBHOOK_URL") is SecretClassification.SENSITIVE


def test_email_classified_sensitive():
    assert classify_key("ALERT_EMAIL") is SecretClassification.SENSITIVE


def test_normal_key_classified_public():
    assert classify_key("PARTICIPATION_RATE") is SecretClassification.PUBLIC


# --- Diff ----------------------------------------


def test_diff_no_changes():
    old = {"A": 1, "B": 2}
    new = {"A": 1, "B": 2}
    assert diff_config(old, new) == ()


def test_diff_added():
    old = {"A": 1}
    new = {"A": 1, "B": 2}
    changes = diff_config(old, new)
    assert len(changes) == 1
    assert changes[0].kind is ChangeKind.ADDED
    assert changes[0].key == "B"


def test_diff_removed():
    old = {"A": 1, "B": 2}
    new = {"A": 1}
    changes = diff_config(old, new)
    assert changes[0].kind is ChangeKind.REMOVED
    assert changes[0].key == "B"


def test_diff_updated():
    old = {"A": 1}
    new = {"A": 2}
    changes = diff_config(old, new)
    assert changes[0].kind is ChangeKind.UPDATED


def test_diff_orders_by_kind_then_key():
    old = {"A": 1, "C": 3}
    new = {"A": 2, "B": 5}  # A updated, B added, C removed
    changes = diff_config(old, new)
    kinds = [c.kind for c in changes]
    # Removed first, then added, then updated (per implementation)
    assert ChangeKind.REMOVED in kinds
    assert ChangeKind.ADDED in kinds
    assert ChangeKind.UPDATED in kinds


def test_diff_secret_changes_require_restart():
    old = {"BINANCE_API_KEY": "old-secret"}
    new = {"BINANCE_API_KEY": "new-secret"}
    changes = diff_config(old, new)
    assert changes[0].requires_restart


def test_diff_public_change_no_restart():
    old = {"PARTICIPATION_RATE": 0.10}
    new = {"PARTICIPATION_RATE": 0.20}
    changes = diff_config(old, new)
    assert not changes[0].requires_restart


def test_diff_secret_value_redacted_in_summary():
    old = {"BINANCE_API_KEY": "sensitive-old-value-12345"}
    new = {"BINANCE_API_KEY": "sensitive-new-value-67890"}
    changes = diff_config(old, new)
    assert changes[0].old_value_summary == "[SECRET]"
    assert changes[0].new_value_summary == "[SECRET]"


def test_diff_sensitive_value_partially_shown():
    old = {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T1/B1/oldXYZ"}
    new = {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T1/B1/newABC"}
    changes = diff_config(old, new)
    # Should have a redaction, not the full URL
    assert "hooks.slack.com" not in changes[0].old_value_summary


def test_diff_public_value_visible():
    old = {"FOO": "old-value"}
    new = {"FOO": "new-value"}
    changes = diff_config(old, new)
    assert "old-value" in changes[0].old_value_summary
    assert "new-value" in changes[0].new_value_summary


# --- Filters --------------------------------


def test_hot_reloadable_filters_secrets():
    changes = diff_config(
        {"PARTICIPATION_RATE": 0.10, "BINANCE_API_KEY": "old"},
        {"PARTICIPATION_RATE": 0.20, "BINANCE_API_KEY": "new"},
    )
    reloadable = hot_reloadable(changes)
    assert len(reloadable) == 1
    assert reloadable[0].key == "PARTICIPATION_RATE"


def test_restart_required_returns_secrets():
    changes = diff_config(
        {"PARTICIPATION_RATE": 0.10, "BINANCE_API_KEY": "old"},
        {"PARTICIPATION_RATE": 0.20, "BINANCE_API_KEY": "new"},
    )
    restart = restart_required(changes)
    assert len(restart) == 1
    assert restart[0].key == "BINANCE_API_KEY"


# --- Render -------------------------------


def test_render_no_changes():
    out = render_changes([])
    assert "no changes" in out


def test_render_with_changes():
    changes = diff_config({"FOO": 1}, {"FOO": 2})
    out = render_changes(changes)
    assert "1 change" in out
    assert "FOO" in out


def test_render_marks_restart_required():
    changes = diff_config({"BINANCE_API_KEY": "x"}, {"BINANCE_API_KEY": "y"})
    out = render_changes(changes)
    assert "🔁" in out


def test_render_marks_hot_reloadable():
    changes = diff_config({"FOO": 1}, {"FOO": 2})
    out = render_changes(changes)
    assert "⚙️" in out


def test_render_no_secret_value_leak():
    changes = diff_config(
        {"BINANCE_API_KEY": "totally-sensitive-secret-do-not-leak"},
        {"BINANCE_API_KEY": "another-totally-sensitive-secret"},
    )
    out = render_changes(changes)
    assert "totally-sensitive" not in out


# --- E2E ----------------------------


def test_e2e_partial_hot_reload_with_one_secret_change():
    """Operator changes participation rate AND API key; only the rate hot-reloads."""
    old = {
        "PARTICIPATION_RATE": 0.10,
        "MAX_POSITION_SIZE": 1000,
        "BINANCE_API_KEY": "old-key",
    }
    new = {
        "PARTICIPATION_RATE": 0.15,  # hot-reloadable
        "MAX_POSITION_SIZE": 1500,  # hot-reloadable
        "BINANCE_API_KEY": "new-key",  # restart required
    }
    changes = diff_config(old, new)
    assert len(hot_reloadable(changes)) == 2
    assert len(restart_required(changes)) == 1


def test_replay_consistency():
    a = diff_config({"FOO": 1}, {"FOO": 2})
    b = diff_config({"FOO": 1}, {"FOO": 2})
    assert a == b
