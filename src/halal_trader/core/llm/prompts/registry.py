"""In-process registry mapping prompt names to immutable, hashed templates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVersion:
    """A registered prompt template snapshot.

    ``version_id`` is a 12-char SHA-256 prefix over the raw template bytes.
    The window is wide enough that collisions are not a practical concern
    for this codebase (we have <50 prompts, not 2^48), and short enough to
    fit comfortably in log lines and DB columns.
    """

    name: str
    version_id: str
    template: str

    @property
    def short(self) -> str:
        """A compact ``name@version_id`` form for log lines."""
        return f"{self.name}@{self.version_id}"


_REGISTRY: dict[str, PromptVersion] = {}


def _hash_template(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def register(name: str, template: str) -> PromptVersion:
    """Register (or re-fetch) a prompt template.

    Idempotent: re-registering the same name with identical text returns
    the existing entry. Re-registering with **different** text raises
    — that means two modules both think they own this name, and silently
    overwriting one would erase the audit trail.
    """
    version_id = _hash_template(template)
    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing.template == template:
            return existing
        raise ValueError(
            f"Prompt name {name!r} already registered with a different template "
            f"(existing version {existing.version_id}, new {version_id}). "
            f"Pick a distinct name (e.g. add a suffix) instead of overwriting."
        )
    pv = PromptVersion(name=name, version_id=version_id, template=template)
    _REGISTRY[name] = pv
    return pv


def get_version(name: str) -> PromptVersion:
    """Return the registered version, raising KeyError if unknown."""
    return _REGISTRY[name]


def list_versions() -> dict[str, PromptVersion]:
    """A copy of the registry — useful for ``halal-trader prompts list``."""
    return dict(_REGISTRY)


def _reset_for_tests() -> None:
    """Clear the registry — tests only.

    The registry is process-global by design; tests that re-register
    prompts use this to start fresh.
    """
    _REGISTRY.clear()
