"""Runtime config hot-reload sentinel — Round-5 Wave 0.F.

When the operator wants to tweak a setting (e.g. lower the LLM cost
cap, change the participation rate) the existing flow is to restart
the bot. For non-secret settings, this module ships the **diff +
event sentinel** that watches a settings dict for changes + emits
structured events the cycle layer can subscribe to.

This is the **pure-Python primitive**. The actual file-watcher (env
mtime polling) sits one layer up; here we exercise the diffing +
event-emission logic in isolation.

Pinned semantics:

- **Closed-set ChangeKind ladder** (ADDED / REMOVED / UPDATED).
- **Closed-set SecretClassification** — secrets cannot hot-reload;
  attempts to change a secret-classified key emit a SECRET_CHANGE
  event the operator must approve via restart.
- **`diff` is pure** — caller passes (old, new) → returns ordered
  tuple of changes. No global state, no clock.
- **Sensitive-key handling** is by name match.
- **No-secret-leak pin** on render output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class ChangeKind(str, Enum):
    """Closed-set change kinds."""

    ADDED = "added"
    REMOVED = "removed"
    UPDATED = "updated"


class SecretClassification(str, Enum):
    """Closed-set sensitivity classes."""

    PUBLIC = "public"
    SENSITIVE = "sensitive"
    SECRET = "secret"


# Regex patterns for keys that must be classified as SECRET
_SECRET_PATTERNS = (
    re.compile(r".*_API_KEY$", re.I),
    re.compile(r".*_SECRET.*", re.I),
    re.compile(r".*_TOKEN$", re.I),
    re.compile(r".*PASSWORD.*", re.I),
    re.compile(r".*PRIVATE_KEY.*", re.I),
    re.compile(r".*HMAC.*", re.I),
)

# Regex patterns for SENSITIVE keys (visible to ops but redacted in logs)
_SENSITIVE_PATTERNS = (
    re.compile(r".*WEBHOOK.*", re.I),
    re.compile(r".*EMAIL.*", re.I),
    re.compile(r".*PHONE.*", re.I),
)


def classify_key(key: str) -> SecretClassification:
    """Classify a setting key by its sensitivity."""
    for pat in _SECRET_PATTERNS:
        if pat.match(key):
            return SecretClassification.SECRET
    for pat in _SENSITIVE_PATTERNS:
        if pat.match(key):
            return SecretClassification.SENSITIVE
    return SecretClassification.PUBLIC


@dataclass(frozen=True)
class ConfigChange:
    """A single config-change event."""

    key: str
    kind: ChangeKind
    classification: SecretClassification
    old_value_summary: str  # redacted
    new_value_summary: str  # redacted
    requires_restart: bool

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("key must be non-empty")


def _summarise(value: object, classification: SecretClassification) -> str:
    """Redact-aware summary."""
    if classification is SecretClassification.SECRET:
        return "[SECRET]"
    s = str(value)
    if classification is SecretClassification.SENSITIVE:
        if len(s) > 8:
            return s[:4] + "…" + s[-2:]
        return "[SENSITIVE]"
    return s


def diff_config(
    old_config: Mapping[str, object],
    new_config: Mapping[str, object],
) -> tuple[ConfigChange, ...]:
    """Diff two config mappings and emit a sorted tuple of changes."""
    old_keys = set(old_config.keys())
    new_keys = set(new_config.keys())

    changes: list[ConfigChange] = []
    for key in sorted(old_keys - new_keys):
        cls = classify_key(key)
        changes.append(
            ConfigChange(
                key=key,
                kind=ChangeKind.REMOVED,
                classification=cls,
                old_value_summary=_summarise(old_config[key], cls),
                new_value_summary="(removed)",
                requires_restart=cls is SecretClassification.SECRET,
            )
        )
    for key in sorted(new_keys - old_keys):
        cls = classify_key(key)
        changes.append(
            ConfigChange(
                key=key,
                kind=ChangeKind.ADDED,
                classification=cls,
                old_value_summary="(new)",
                new_value_summary=_summarise(new_config[key], cls),
                requires_restart=cls is SecretClassification.SECRET,
            )
        )
    for key in sorted(old_keys & new_keys):
        if old_config[key] != new_config[key]:
            cls = classify_key(key)
            changes.append(
                ConfigChange(
                    key=key,
                    kind=ChangeKind.UPDATED,
                    classification=cls,
                    old_value_summary=_summarise(old_config[key], cls),
                    new_value_summary=_summarise(new_config[key], cls),
                    requires_restart=cls is SecretClassification.SECRET,
                )
            )
    return tuple(changes)


def hot_reloadable(changes: Iterable[ConfigChange]) -> tuple[ConfigChange, ...]:
    """Filter changes that can be hot-reloaded (no restart required)."""
    return tuple(c for c in changes if not c.requires_restart)


def restart_required(changes: Iterable[ConfigChange]) -> tuple[ConfigChange, ...]:
    """Filter changes that require a full restart (typically secrets)."""
    return tuple(c for c in changes if c.requires_restart)


def render_changes(changes: Iterable[ConfigChange]) -> str:
    changes_t = tuple(changes)
    if not changes_t:
        return "Config diff: no changes"
    head = f"Config diff: {len(changes_t)} change(s)"
    lines = [head]
    for c in changes_t:
        marker = "🔁" if c.requires_restart else "⚙️"
        lines.append(
            f"  {marker} {c.kind.value} [{c.classification.value}] {c.key}: "
            f"{c.old_value_summary} → {c.new_value_summary}"
        )
    return "\n".join(lines)
