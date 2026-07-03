"""Retrieval-augmented thesis grounding (Task B slice 2; spec `rag_grounding`).

When the sparse thesis writer fires (material shift — the engine's one LLM
touch), retrieve the most analogous PAST trade rationales and their realized
outcomes from the legacy pgvector store and hand them to the prompt, so the
narrative is grounded in the bot's own history instead of only the current
evidence. Read-only over the store; conviction and direction are untouched
(retrieval→conviction is Task C, deliberately data-gated).

Coupling follows the perception precedent: this module depends on a local
duck-typed :class:`RationaleStore`; the concrete legacy
``halal_trader.core.llm.rag_db.DBRationaleStore`` (deterministic
HashingEmbedder — no external API, INV-1-friendly) is constructed only in
the composition root.

Degradation contract: ANY retrieval failure returns an empty context — the
thesis still gets written, just ungrounded. Retrieval must never be the
reason a narrative (let alone a belief) is missing.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from halabot.belief.schema import BeliefState

logger = logging.getLogger(__name__)


class RationaleStore(Protocol):
    """Duck-typed slice of the legacy DBRationaleStore surface."""

    async def query(
        self,
        text: str,
        *,
        k: int = 5,
        min_similarity: float = 0.1,
        symbol: str | None = None,
    ) -> list[tuple[Any, float]]: ...
    async def aggregate(self, hits: Any) -> dict[str, Any]: ...


def _query_text(b: BeliefState) -> str:
    """Deterministic setup descriptor, vocabulary-matched to the corpus.

    The store's HashingEmbedder is token-overlap only (MD5 bag-of-words),
    and the stored texts are LLM trade reasoning — indicator/direction
    words. Emit the same register: direction words, regime words, and the
    top evidence sources with their signs.
    """
    direction = "bullish long" if b.direction.value == "long_bias" else (
        "bearish short" if b.direction.value == "short_bias" else "neutral flat"
    )
    regime_words = b.regime.value.replace("_", " ")
    ev_words = " ".join(
        f"{e.source} {'bullish' if e.direction > 0 else 'bearish'}"
        for e in sorted(b.evidence, key=lambda e: -abs(e.direction * e.weight))[:5]
    )
    return f"{b.asset} {direction} {regime_words} {ev_words}"[:600]


def _format_hits(hits: list[tuple[Any, float]], stats: dict[str, Any]) -> str:
    """Compact prompt block (each line < 100 chars). Empty hits → ""."""
    if not hits:
        return ""
    lines = [
        f"Similar past setups (own history, n={stats.get('n', len(hits))}, "
        f"weighted win-rate {stats.get('weighted_win_rate', 0.0):.0%}, "
        f"weighted P&L {stats.get('weighted_pnl_pct', 0.0):+.2%}):"
    ]
    for row, sim in hits[:5]:
        outcome = "WIN" if getattr(row, "outcome_win", False) else "LOSS"
        pnl = float(getattr(row, "outcome_pnl_pct", 0.0) or 0.0)
        symbol = getattr(row, "symbol", "?")
        text = str(getattr(row, "text", "") or "").strip().replace("\n", " ")[:70]
        lines.append(f"  · sim={sim:+.2f} | {outcome} {pnl:+.2%} ({symbol}): {text}")
    return "\n".join(lines)


class SetupRetriever:
    """Queries the rationale store for setups analogous to a belief."""

    def __init__(
        self,
        store: RationaleStore,
        *,
        k: int = 5,
        min_similarity: float = 0.15,
    ) -> None:
        self._store = store
        self._k = k
        self._min_similarity = min_similarity

    async def context_for(self, belief: BeliefState) -> str:
        """Prompt block of analogous past setup→outcomes, or "" on any failure."""
        try:
            hits = await self._store.query(
                _query_text(belief), k=self._k, min_similarity=self._min_similarity
            )
            if not hits:
                return ""
            stats = await self._store.aggregate(hits)
            block = _format_hits(hits, stats)
            if block:
                logger.info(
                    "retrieval grounded %s thesis: %d hits, weighted win-rate %.0f%%",
                    belief.asset,
                    stats.get("n", len(hits)),
                    100.0 * float(stats.get("weighted_win_rate", 0.0)),
                )
            return block
        except Exception as exc:  # noqa: BLE001 — retrieval must never block the thesis
            logger.warning("setup retrieval failed for %s: %r", belief.asset, exc)
            return ""
