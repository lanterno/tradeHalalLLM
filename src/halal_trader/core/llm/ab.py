"""A/B prompt routing harness.

Lets us run two prompt versions against the same market data so we can
measure (Sharpe, win rate, cost) per version using the existing
``LlmDecision.prompt_version`` column. The harness is a tiny pure
component: it picks a variant per call, returns the picked
``(system, user, version_id)`` triple, and records nothing — recording
is already handled by ``BaseStrategy._run_llm_analysis`` via the
``prompt_version`` field, so the dashboard's
``/api/research/prompt-versions`` endpoint groups results without any
extra plumbing.

We pick variants **deterministically** by hashing a per-call key (the
cycle id, normally), not at random — so a single cycle always lands on
the same variant across crypto and stock pipelines, which prevents
cross-contamination when two bots share an LLM. Override the key
generation in tests to replay specific orderings.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from halal_trader.core.llm.prompts import PromptVersion


@dataclass(frozen=True)
class PromptVariant:
    """One variant in the A/B harness — name + assembled prompt."""

    version: PromptVersion
    system: str
    user: str
    weight: float = 1.0  # relative weight; harness normalises across all variants


@dataclass(frozen=True)
class ABRouter:
    """Picks one variant from a fixed set, weighted, deterministic by key.

    The router is *stateless and immutable* — that's deliberate. State
    (which variant won this cycle, what its outcome was) lives in the
    LlmDecision row via ``prompt_version``. The router only chooses.
    """

    variants: Sequence[PromptVariant]

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError("ABRouter requires at least one variant")
        if any(v.weight <= 0 for v in self.variants):
            raise ValueError("variant weights must be positive")

    def choose(self, key: str) -> PromptVariant:
        """Deterministically pick a variant for ``key``.

        Hash ``key`` to a uniform [0, 1) point and step through the
        weight intervals. Same key always yields the same variant, so
        replays line up.
        """
        total = sum(v.weight for v in self.variants)
        h = hashlib.sha256(key.encode("utf-8")).digest()
        # Use the low 8 bytes as an unsigned int → uniform in [0, 1).
        u = int.from_bytes(h[:8], "big") / 2**64
        target = u * total
        cumulative = 0.0
        for v in self.variants:
            cumulative += v.weight
            if target < cumulative:
                return v
        return self.variants[-1]  # rounding safety net


def expected_split(router: ABRouter) -> dict[str, float]:
    """Return the expected per-variant share given the configured weights.

    Useful for surfacing "we're routing 70% to v1, 30% to v2" in the
    dashboard so operators can see the experiment design at a glance.
    """
    total = sum(v.weight for v in router.variants)
    return {v.version.short: v.weight / total for v in router.variants}
