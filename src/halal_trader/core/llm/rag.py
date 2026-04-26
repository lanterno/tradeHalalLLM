"""RAG over the bot's own LLM reasoning traces.

The strategy LLM emits a ``reasoning`` field per decision; the cycle
persists thousands of these per week. Right now they're audit-only.
The cheap upgrade is *retrieval over our own track record*: at decision
time, find the K most-similar past rationales and surface their
outcomes to the LLM as context. ("Last 3 times you saw a similar
setup, you lost 2 of 3 — be careful.")

This module is the retrieval layer. Two design choices:

* **Hashing-trick embedding** (no external model). Token-level lowercase
  hash with a small fixed-dim feature vector + cosine. Quality is lower
  than MiniLM but you get exact reproducibility, zero install footprint,
  and ~50µs/text. The interface is :class:`Embedder`, so swapping in
  ``sentence-transformers`` later is one file.
* **JSON sidecar** for the store. Per-trade rationale + outcome rows live
  next to the regret/thesis sidecars under ``data/analytics/``. Move to
  pgvector / LanceDB when the row count justifies it.

Public API:
    embedder = HashingEmbedder(dim=512)
    store = RationaleStore(path=..., embedder=embedder)
    store.add(trade_id="t1", text="rsi 35 with bb lower", outcome_pnl_pct=0.012)
    hits = store.query("similar regime here", k=5)
    text = format_rag_for_prompt(hits)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

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


# ── Store ────────────────────────────────────────────────────────


@dataclass
class RationaleRow:
    """One stored rationale + outcome."""

    trade_id: str
    symbol: str
    text: str
    vector: list[float]
    outcome_pnl_pct: float
    outcome_win: bool
    setup_type: str | None = None
    timestamp: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class RationaleStore:
    """JSON-on-disk store with linear-scan cosine retrieval.

    O(N) per query is fine up to a few thousand rows. When the store
    grows past that point, swap the body of :meth:`query` for a real
    vector index — the public API stays the same.
    """

    path: Path
    embedder: Embedder = field(default_factory=lambda: HashingEmbedder())
    rows: list[RationaleRow] = field(default_factory=list)
    capacity: int = 10_000

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("rationale store unreadable, starting fresh: %s", exc)
            return
        for row in raw.get("rows", []):
            try:
                self.rows.append(RationaleRow(**row))
            except Exception:  # noqa: BLE001
                continue

    def _save(self) -> None:
        self.path.write_text(json.dumps({"rows": [asdict(r) for r in self.rows]}, indent=2))

    @property
    def size(self) -> int:
        return len(self.rows)

    def add(
        self,
        *,
        trade_id: str,
        symbol: str,
        text: str,
        outcome_pnl_pct: float,
        setup_type: str | None = None,
        timestamp: str = "",
        tags: Iterable[str] | None = None,
    ) -> RationaleRow:
        """Embed + persist one rationale. Idempotent on ``trade_id``."""
        for existing in self.rows:
            if existing.trade_id == trade_id:
                return existing
        row = RationaleRow(
            trade_id=trade_id,
            symbol=symbol,
            text=text,
            vector=self.embedder.embed(text),
            outcome_pnl_pct=outcome_pnl_pct,
            outcome_win=outcome_pnl_pct > 0,
            setup_type=setup_type,
            timestamp=timestamp,
            tags=list(tags or []),
        )
        self.rows.append(row)
        if len(self.rows) > self.capacity:
            self.rows = self.rows[-self.capacity :]
        self._save()
        return row

    def query(
        self,
        text: str,
        *,
        k: int = 5,
        min_similarity: float = 0.1,
        symbol: str | None = None,
    ) -> list[tuple[RationaleRow, float]]:
        """Top-K cosine matches, optionally filtered by symbol."""
        if not text or not self.rows:
            return []
        q = self.embedder.embed(text)
        scored: list[tuple[RationaleRow, float]] = []
        for row in self.rows:
            if symbol is not None and row.symbol != symbol:
                continue
            score = cosine(q, row.vector)
            if score >= min_similarity:
                scored.append((row, score))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    def aggregate(self, hits: Iterable[tuple[RationaleRow, float]]) -> dict[str, Any]:
        """Similarity-weighted stats over a query result."""
        hits = list(hits)
        if not hits:
            return {"n": 0, "weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0}
        total_w = 0.0
        wp = 0.0
        ww = 0.0
        for row, sim in hits:
            w = max(0.0, sim)
            total_w += w
            wp += w * row.outcome_pnl_pct
            ww += w * (1.0 if row.outcome_win else 0.0)
        if total_w == 0:
            return {"n": len(hits), "weighted_pnl_pct": 0.0, "weighted_win_rate": 0.0}
        return {
            "n": len(hits),
            "weighted_pnl_pct": wp / total_w,
            "weighted_win_rate": ww / total_w,
        }


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
