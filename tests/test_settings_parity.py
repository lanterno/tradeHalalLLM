"""Ensure every Settings field appears in .env.example.

Run this in CI so a new `Field(...)` cannot ship without a documented entry.
Secret-only fields (no default and clearly an API key) still need a stub line
so operators know they exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from halal_trader.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


def _env_keys() -> set[str]:
    text = ENV_EXAMPLE.read_text()
    return {m.group(1) for m in re.finditer(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE)}


def test_env_example_exists():
    assert ENV_EXAMPLE.exists(), f"missing {ENV_EXAMPLE}"


def test_every_settings_field_documented():
    documented = _env_keys()
    missing = []
    for name in Settings.model_fields:
        if name.upper() not in documented:
            missing.append(name)
    assert not missing, (
        f"Settings fields missing from .env.example: {sorted(missing)}. "
        f"Add a line `{missing[0].upper()}=<default>` with a # comment."
    )


def test_no_unknown_keys_in_env_example():
    """Catch the reverse drift: stale env keys without a Settings field."""
    settings_keys = {name.upper() for name in Settings.model_fields}
    documented = _env_keys()
    unknown = documented - settings_keys
    assert not unknown, (
        f".env.example has keys not in Settings: {sorted(unknown)}. "
        f"Either add the field to config.py or remove the env line."
    )


@pytest.mark.parametrize("field_name", sorted(Settings.model_fields.keys()))
def test_field_default_matches_or_is_explicit(field_name):
    """The example default should not contradict the Settings default.

    Skipped for secrets (empty default + API/secret in the name) since the
    example purposely leaves them blank.
    """
    field = Settings.model_fields[field_name]
    if any(s in field_name for s in ("api_key", "secret", "token", "client_id", "chat_id")):
        return

    text = ENV_EXAMPLE.read_text()
    pattern = rf"^{field_name.upper()}=(.*)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match, f"{field_name.upper()} not found"
    example_value = match.group(1).strip()

    default = field.default
    if default is None or default == "":
        return
    if isinstance(default, bool):
        assert example_value.lower() == str(default).lower(), (
            f"{field_name.upper()}: example={example_value!r} default={default!r}"
        )
    elif isinstance(default, (int, float)):
        assert str(default) == example_value or float(example_value) == float(default), (
            f"{field_name.upper()}: example={example_value!r} default={default!r}"
        )
