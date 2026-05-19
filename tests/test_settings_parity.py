"""Ensure every leaf Settings field appears in .env.example.

Run this in CI so a new ``Field(...)`` cannot ship without a documented entry.
Secret-only fields (no default and clearly an API key) still need a stub line
so operators know they exist.

The nested ``Settings`` model has each sub-section under its own
``BaseSettings`` with an ``env_prefix``. We walk the tree, derive each
leaf's effective env-var name (prefix + field) or its explicit
``validation_alias``, then compare against the keys in ``.env.example``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from halal_trader.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
# Stocks-only starter template — same field set as the canonical
# example, just reorganized for operators running ``just stocks`` only.
ENV_STOCKS_EXAMPLE = PROJECT_ROOT / ".env.stocks.example"


def _env_keys(path: Path = ENV_EXAMPLE) -> set[str]:
    text = path.read_text()
    return {m.group(1) for m in re.finditer(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE)}


def _model_prefix(model: type[BaseSettings]) -> str:
    return model.model_config.get("env_prefix", "") or ""


def _walk_leaves(
    model: type[BaseSettings],
) -> list[tuple[str, FieldInfo, type[BaseSettings]]]:
    """Yield (env_name, field_info, owner_model) for every scalar leaf.

    Nested ``BaseSettings`` sub-models load .env with their own prefix
    in isolation — prefixes do not compound across the parent chain.
    """
    own_prefix = _model_prefix(model)
    out: list[tuple[str, FieldInfo, type[BaseSettings]]] = []
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseSettings):
            out.extend(_walk_leaves(ann))
            continue
        alias = field.validation_alias
        if isinstance(alias, str):
            env_name = alias
        else:
            env_name = (own_prefix + name).upper()
        out.append((env_name, field, model))
    return out


_LEAVES = _walk_leaves(Settings)
_LEAF_NAMES = sorted({env_name for env_name, _, _ in _LEAVES})


def test_env_example_exists():
    assert ENV_EXAMPLE.exists(), f"missing {ENV_EXAMPLE}"


def test_every_settings_field_documented():
    documented = _env_keys()
    missing = [name for name in _LEAF_NAMES if name not in documented]
    assert not missing, (
        f"Settings fields missing from .env.example: {sorted(missing)}. "
        f"Add a line `{missing[0]}=<default>` with a # comment."
    )


def test_no_unknown_keys_in_env_example():
    """Catch the reverse drift: stale env keys without a Settings field."""
    documented = _env_keys()
    unknown = documented - set(_LEAF_NAMES)
    assert not unknown, (
        f".env.example has keys not in Settings: {sorted(unknown)}. "
        f"Either add the field to config.py or remove the env line."
    )


@pytest.mark.parametrize("env_name", _LEAF_NAMES)
def test_field_default_matches_or_is_explicit(env_name):
    """The example default should not contradict the Settings default.

    Skipped for secrets (empty default + API/secret in the name) since the
    example purposely leaves them blank.
    """
    field = next(f for n, f, _ in _LEAVES if n == env_name)
    if any(s in env_name.lower() for s in ("api_key", "secret", "token", "client_id", "chat_id")):
        return

    text = ENV_EXAMPLE.read_text()
    pattern = rf"^{env_name}=(.*)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match, f"{env_name} not found"
    example_value = match.group(1).strip()

    default = field.default
    if default is None or default == "":
        return
    if isinstance(default, bool):
        assert example_value.lower() == str(default).lower(), (
            f"{env_name}: example={example_value!r} default={default!r}"
        )
    elif isinstance(default, (int, float)):
        assert str(default) == example_value or float(example_value) == float(default), (
            f"{env_name}: example={example_value!r} default={default!r}"
        )


# ── .env.stocks.example parity (same field set, reorganized) ─────


def test_env_stocks_example_exists():
    assert ENV_STOCKS_EXAMPLE.exists(), f"missing {ENV_STOCKS_EXAMPLE}"


def test_stocks_example_documents_every_settings_field():
    """The stocks-only starter template should be a complete reference
    — every leaf in ``Settings`` must be documented, even the crypto-side
    fields (kept at the bottom under "NOT USED for stocks"). Otherwise
    operators copying ``.env.stocks.example → .env`` would silently lose
    a knob they later wanted to flip on.
    """
    documented = _env_keys(ENV_STOCKS_EXAMPLE)
    missing = [name for name in _LEAF_NAMES if name not in documented]
    assert not missing, (
        f"Settings fields missing from .env.stocks.example: {sorted(missing)}. "
        f"Add a line `{missing[0]}=<default>` (probably under the "
        f"NOT USED section if it's crypto-side)."
    )


def test_stocks_example_has_no_unknown_keys():
    documented = _env_keys(ENV_STOCKS_EXAMPLE)
    unknown = documented - set(_LEAF_NAMES)
    assert not unknown, (
        f".env.stocks.example has keys not in Settings: {sorted(unknown)}. "
        f"Either add the field to config.py or remove the env line."
    )


def test_stocks_example_matches_canonical_field_set():
    """Both example files document the same fields. Add a field to
    ``Settings`` → both files must list it; remove a field → both files
    must drop the line. Catches drift before the docs lie to operators.
    """
    canonical = _env_keys(ENV_EXAMPLE)
    stocks = _env_keys(ENV_STOCKS_EXAMPLE)
    only_in_canonical = canonical - stocks
    only_in_stocks = stocks - canonical
    assert not only_in_canonical, (
        f"In .env.example but not .env.stocks.example: {sorted(only_in_canonical)}"
    )
    assert not only_in_stocks, (
        f"In .env.stocks.example but not .env.example: {sorted(only_in_stocks)}"
    )
