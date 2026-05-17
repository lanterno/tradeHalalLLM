"""Standard *Policy dataclass primitives — Round-5 Wave 0.G.

The codebase has accumulated 30+ ``*Policy`` dataclasses
(``MaysirPolicy``, ``GhararPolicy``, ``ZakatPolicy``, ``StructuringPolicy``,
``ScreenPolicy``, ``RibaPolicy``, etc.) — each correctly built but each
re-implementing the same boilerplate (frozen, slots, validation, replay
equality). Two costs follow:

1. **Replay tooling has to know each Policy's shape.** When the replay
   harness re-loads cycle inputs, it imports the specific dataclass.
2. **Operators tweaking thresholds touch N modules.** A "stricter
   defaults across the board" pass means visiting 30 files.

This module ships the **policy registry** + a small structural base
helper so every new ``*Policy`` registers itself + accepts a uniform
"export to dict / import from dict" round-trip. It is *additive*: it
does not require existing ``*Policy`` dataclasses to subclass anything
(Python dataclasses don't compose subclassing well with frozen+slots
across boundaries). Instead, modules call :func:`register_policy` once
(typically at module import), and the registry exposes a single
``policies()`` view for tooling.

Pinned semantics:

- **The registry is import-time-built.** No mutation after first lookup
  is needed for normal operation, but ``register_policy`` is idempotent
  per ``policy_id`` (re-registering the same id with the same class is a
  no-op; re-registering with a *different* class raises).
- **``policies()`` is a frozen view.** Callers iterate, never mutate.
- **``snapshot_to_dict`` + ``snapshot_from_dict``** are the round-trip
  primitives. They use ``dataclasses.asdict`` on the way out (recursing
  into nested dataclasses + handling Enum values) and ``cls(**values)``
  on the way back, which works for any frozen dataclass whose fields
  are JSON-friendly.
- **No global side-effects.** The registry is module-private; importing
  this module does not register anyone — modules register themselves.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from typing import Any

_REGISTRY: dict[str, type] = {}


def register_policy(policy_id: str, cls: type) -> type:
    """Register a ``*Policy`` dataclass under ``policy_id``.

    Idempotent: re-registering with the same class is a no-op.
    Re-registering a different class under the same id raises ``ValueError``.

    Returns the class so it can be used as a decorator::

        @register_policy_decorator("maysir")
        @dataclass(frozen=True)
        class MaysirPolicy: ...
    """
    if not policy_id or not policy_id.strip():
        raise ValueError("policy_id must be non-empty")
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    existing = _REGISTRY.get(policy_id)
    if existing is None:
        _REGISTRY[policy_id] = cls
    elif existing is not cls:
        raise ValueError(
            f"policy_id {policy_id!r} already registered for "
            f"{existing.__name__}; cannot rebind to {cls.__name__}"
        )
    return cls


def register_policy_decorator(policy_id: str):
    """Decorator form of :func:`register_policy`."""

    def _wrap(cls: type) -> type:
        return register_policy(policy_id, cls)

    return _wrap


def policy_class(policy_id: str) -> type | None:
    """Return the registered class for ``policy_id`` (or ``None``)."""
    return _REGISTRY.get(policy_id)


def policies() -> Mapping[str, type]:
    """Return a read-only view of the registered policies."""
    # Return a fresh dict so callers can't mutate the underlying registry.
    return dict(_REGISTRY)


def known_policy_ids() -> tuple[str, ...]:
    """Return the registered policy ids in registration order."""
    return tuple(_REGISTRY.keys())


# --- Snapshot round-trip -----------------------------------------------------


def _coerce_for_snapshot(value: Any) -> Any:
    """Recursively convert dataclass / Enum / set instances into JSON-friendly types."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _coerce_for_snapshot(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, (frozenset, set)):
        return sorted(_coerce_for_snapshot(v) for v in value)
    if isinstance(value, tuple):
        return [_coerce_for_snapshot(v) for v in value]
    if isinstance(value, list):
        return [_coerce_for_snapshot(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_for_snapshot(v) for k, v in value.items()}
    return value


def snapshot_to_dict(policy: Any) -> dict[str, Any]:
    """Serialize a frozen ``*Policy`` dataclass instance to a JSON-friendly dict."""
    if not dataclasses.is_dataclass(policy) or isinstance(policy, type):
        raise TypeError("snapshot_to_dict expects a dataclass *instance*")
    return _coerce_for_snapshot(policy)


def _coerce_field_value(field: dataclasses.Field, raw: Any) -> Any:
    """Best-effort cast of a snapshot value back into the field's declared type."""
    if raw is None:
        return None
    ftype = field.type
    if isinstance(ftype, type) and issubclass(ftype, Enum):
        return ftype(raw)
    return raw


def snapshot_from_dict(policy_id: str, data: Mapping[str, Any]) -> Any:
    """Reconstruct a ``*Policy`` instance from a snapshot dict.

    The class must be registered under ``policy_id``. Field values are
    cast back through ``_coerce_field_value`` (Enums get re-typed; other
    fields go straight through, relying on the dataclass's own
    validation in ``__post_init__``).
    """
    cls = policy_class(policy_id)
    if cls is None:
        raise KeyError(f"no policy registered under {policy_id!r}")
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    extras = set(data.keys()) - set(field_map.keys())
    if extras:
        raise ValueError(f"unknown fields for {policy_id!r}: {sorted(extras)}")
    kwargs = {
        name: _coerce_field_value(f, data[name]) for name, f in field_map.items() if name in data
    }
    return cls(**kwargs)


# --- Utilities ---------------------------------------------------------------


def reset_registry_for_testing() -> None:
    """Clear the registry. **Tests only** — never call from production code."""
    _REGISTRY.clear()


def render_registry() -> str:
    """Render the registry as a deterministic multi-line string for the dashboard."""
    if not _REGISTRY:
        return "No policies registered."
    lines = [f"{len(_REGISTRY)} policies registered:"]
    for pid in sorted(_REGISTRY.keys()):
        cls = _REGISTRY[pid]
        lines.append(f"  • {pid}: {cls.__module__}.{cls.__name__}")
    return "\n".join(lines)


__all__ = [
    "register_policy",
    "register_policy_decorator",
    "policy_class",
    "policies",
    "known_policy_ids",
    "snapshot_to_dict",
    "snapshot_from_dict",
    "reset_registry_for_testing",
    "render_registry",
]
