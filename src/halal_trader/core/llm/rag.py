"""RAG over the bot's own LLM reasoning traces — embedding + retrieval primitives.

The strategy LLM emits a ``reasoning`` field per decision; the cycle
persists thousands of these per week. The cheap upgrade is *retrieval
over our own track record*: at decision time, find the K most-similar
past rationales and surface their outcomes to the LLM as context.
("Last 3 times you saw a similar setup, you lost 2 of 3 — be careful.")

This module owns the **embedding + scoring + prompt formatting**
primitives. The actual store is DB-backed (`core/llm/rag_db.py`); the
public dataclass and pure functions here are re-used both by the DB
store and by callers shaping prompts.

Two design choices:

* **Hashing-trick embedding** (no external model). Token-level lowercase
  hash with a small fixed-dim feature vector + cosine. Quality is lower
  than MiniLM but you get exact reproducibility, zero install footprint,
  and ~50µs/text. The interface is :class:`Embedder`, so swapping in
  ``sentence-transformers`` later is one file.
* **DB-backed storage** in `rag_rationales` (Postgres + pgvector-ready).
  Linear-scan cosine over JSON-encoded vectors today; swap the body of
  :meth:`DBRationaleStore.query` for an HNSW index without touching the
  public API.

Public API:
    embedder = HashingEmbedder(dim=512)
    store = DBRationaleStore(engine, embedder=embedder)
    await store.add(trade_id="t1", symbol="BTCUSDT", text="rsi 35 with bb lower",
                    outcome_pnl_pct=0.012)
    hits = await store.query("similar regime here", k=5)
    text = format_rag_for_prompt(hits)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


class Embedder(Protocol):
    """Anything that turns a string into a fixed-length embedding."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


# ── Hashing-trick embedder ───────────────────────────────────────


@dataclass
class HashingEmbedder:
    """Hashes lowercase tokens into a fixed-dim bag-of-words vector.

    L2-normalised so cosine == dot product downstream. Stable: same
    text → same vector across runs (uses MD5, not Python's `hash()`).

    For ~thousands of trade rationales, recall is "decent" — enough
    to surface obvious analogues. Replace with MiniLM when the
    embedding-quality lift justifies the dependency.
    """

    dim: int = 512

    def _tokens(self, text: str) -> Iterable[str]:
        return (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim
        v = [0.0] * self.dim
        for tok in self._tokens(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return v
        return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised vectors.

    For unit vectors this is just the dot product, but we don't assume
    that — pre-normalisation in :class:`HashingEmbedder` keeps it cheap.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


# ── Result row ───────────────────────────────────────────────────


@dataclass
class RationaleRow:
    """One stored rationale + outcome — DB store hydrates this shape."""

    trade_id: str
    symbol: str
    text: str
    vector: list[float]
    outcome_pnl_pct: float
    outcome_win: bool
    setup_type: str | None = None
    timestamp: str = ""
    tags: list[str] = field(default_factory=list)


# ── Prompt formatter ─────────────────────────────────────────────


def format_rag_for_prompt(hits: list[tuple[RationaleRow, float]], *, max_rows: int = 5) -> str:
    """Render top hits as a compact prompt block.

    Empty result returns "" so the prompt builder can elide the section
    cleanly. Output keeps each line < 100 chars to avoid bloating the
    LLM input window with retrieval noise.
    """
    if not hits:
        return ""
    lines = ["Most analogous past rationales (your own history):"]
    for row, sim in hits[:max_rows]:
        outcome = "WIN" if row.outcome_win else "LOSS"
        text = (row.text or "").strip().replace("\n", " ")[:80]
        lines.append(
            f"  · sim={sim:+.2f} | {outcome} {row.outcome_pnl_pct:+.2%} ({row.symbol}): {text}"
        )
    return "\n".join(lines)
