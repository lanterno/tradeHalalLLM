"""Prompt registry — single source of truth for "which prompt template ran."

Each strategy module registers its prompt *templates* (the static parts —
not the per-cycle data) here at import time. The registry computes a
stable short SHA over the template text so every LlmDecision row can
record exactly which prompt version produced it. Editing a template
mid-week and forgetting to bump the version is impossible: the hash
changes the moment the bytes change.

The registry is intentionally additive over the existing prompt files
(``crypto/prompts.py``, ``trading/strategy.py``) so we can adopt
versioning incrementally without a big-bang refactor.
"""

from halal_trader.core.llm.prompts.registry import (
    PromptVersion,
    get_version,
    list_versions,
    register,
)

__all__ = ["PromptVersion", "get_version", "list_versions", "register"]
