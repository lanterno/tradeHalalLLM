"""TF-IDF full-text search over trade rationales.

Round-4 wave 5.C: every closed trade carries the LLM's reasoning
on the `LlmDecision.parsed_action.reasoning` field (and on
`Trade.llm_reasoning` after the cycle persists). Operators want
to ask "show me every trade that mentioned MACD divergence" or
"every losing trade where the LLM cited high vol". This module
is the in-memory index that answers.

Why a hand-rolled TF-IDF rather than Postgres full-text search:

* The rationale corpus is small (≤ 100k decisions over a year);
  in-memory search returns sub-millisecond and lets the dashboard
  scrub queries live without a DB round-trip per keystroke.
* Postgres `tsvector` weights by language-aware stemming we don't
  control, and the dashboard's explainability requirement ("why
  did this rank first?") needs per-token contributions exposed
  — easier to surface when we own the scoring path.
* Pure-Python keeps the index testable without a database and
  matches the rest of the Round-4 isolated-module pattern.

Two layers:

* **Tokenisation** — case-folded, split on word boundaries, drop
  short tokens + a small stop-word list. Pin: deliberately
  conservative — we don't stem ("macd-ing" is not a real word
  here) and don't lowercase to a separate locale (English-only
  corpus).
* **TF-IDF** — classic term-frequency × inverse-document-frequency
  with sub-linear TF dampening (`1 + log(tf)` so a rationale
  that mentions "MACD" 10 times doesn't score 10× a rationale
  that mentions it once).

The search returns ranked `SearchHit`s with per-query-term
contributions so the dashboard can highlight *why* each result
matched.

Halal alignment: pure read-only search over data the cycle
already persisted. Never opens a position, never bypasses the
screener.

Pure-Python; no numpy / scipy / DB / async.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

# ── Tokeniser ────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


# Small stop-word list — only the most common contentless tokens.
# Pin: deliberately tiny. Operators search for terms like "the
# market" or "is overbought" and removing every stop word kills
# phrase recall.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Lower-case word-boundary tokenisation. Pin: drops tokens
    shorter than 2 chars and the stop-word list.

    Single-char tokens are dropped because a hit on "a" or "i"
    is noise; keeping them would make every rationale's IDF for
    those tokens approach zero anyway."""
    if not text:
        return []
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    return [t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS]


# ── Documents ────────────────────────────────────────────


@dataclass(frozen=True)
class RationaleDoc:
    """One indexable rationale.

    ``doc_id`` is the operator's identifier — typically a trade /
    decision ID. ``text`` is the full rationale; the indexer
    tokenises and stores term frequencies. ``metadata`` is
    free-form pass-through (pair, side, regime, return_pct) so
    callers can filter results without a DB round-trip.
    """

    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


# ── Search hit ───────────────────────────────────────────


@dataclass(frozen=True)
class TermContribution:
    """How much one query term contributed to a hit's score."""

    term: str
    tf: int
    idf: float
    contribution: float


@dataclass(frozen=True)
class SearchHit:
    """One ranked result.

    ``score`` is the operator-visible relevance number. ``terms``
    is the per-query-term breakdown — the dashboard renders this
    as a stacked bar so the operator sees *why* the result ranked
    first ("matched MACD 4× and divergence 1×")."""

    doc: RationaleDoc
    score: float
    terms: list[TermContribution] = field(default_factory=list)
    snippet: str = ""


# ── Index ────────────────────────────────────────────────


class RationaleIndex:
    """In-memory TF-IDF index.

    ``add(doc)`` ingests a rationale; ``search(query, *, limit)``
    returns the top-K matching `SearchHit`s. The index is
    immutable from the caller's perspective beyond `add` /
    `clear` — pin so concurrent searches don't see partially-
    indexed state.

    Removal isn't supported by design: the trade post-mortem
    surface is append-only (an LLM decision row, once written,
    is the audit record). If the index grows past memory limits,
    the caller rebuilds from a date-bounded subset of the
    `LlmDecision` table.
    """

    def __init__(self) -> None:
        self._docs: list[RationaleDoc] = []
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._df: dict[str, int] = {}  # how many docs contain term
        self._doc_id_to_idx: dict[str, int] = {}

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def add(self, doc: RationaleDoc) -> None:
        """Add one document to the index.

        Pin: duplicate `doc_id` raises rather than silently
        overwriting — the audit trail's append-only invariant
        means an "update" should be impossible. If a caller
        rebuilds the index from scratch, they `clear()` first."""
        if doc.doc_id in self._doc_id_to_idx:
            raise ValueError(f"duplicate doc_id {doc.doc_id!r}; clear() to rebuild")
        tokens = tokenize(doc.text)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        idx = len(self._docs)
        self._docs.append(doc)
        self._term_freqs.append(tf)
        self._doc_lengths.append(len(tokens))
        self._doc_id_to_idx[doc.doc_id] = idx
        # Update document-frequency index.
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1

    def clear(self) -> None:
        self._docs.clear()
        self._term_freqs.clear()
        self._doc_lengths.clear()
        self._df.clear()
        self._doc_id_to_idx.clear()

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing.

        Pin: `log((N + 1) / (df + 1)) + 1` keeps the IDF non-
        negative even for terms that appear in every document
        (e.g. "buy"). A zero IDF would make those contributions
        invisible; a positive floor keeps them ranked sensibly
        below truly rare terms."""
        if self.doc_count == 0:
            return 0.0
        df = self._df.get(term, 0)
        return math.log((self.doc_count + 1) / (df + 1)) + 1.0

    def _score(self, doc_idx: int, query_terms: list[str]) -> tuple[float, list[TermContribution]]:
        """Score one document against a tokenised query.

        Returns the total + the per-term contributions. Pin:
        sub-linear TF dampening (`1 + log(tf)`) so a 10× repeated
        term doesn't 10× the score."""
        tf_map = self._term_freqs[doc_idx]
        total = 0.0
        contribs: list[TermContribution] = []
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            tf_weight = 1.0 + math.log(tf)
            contribution = tf_weight * idf
            total += contribution
            contribs.append(TermContribution(term=term, tf=tf, idf=idf, contribution=contribution))
        return total, contribs

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        require_all: bool = False,
    ) -> list[SearchHit]:
        """Search for a query string. Pin: tokenises the query the
        same way as documents (case-folded, stop-words dropped)
        so a query "the MACD" matches a doc that says "MACD".

        ``require_all=True`` filters out docs that don't contain
        every query term — operator's "must include" search.
        Default `False` is OR-style ranking.

        Returns up to ``limit`` `SearchHit`s sorted by descending
        score. Stable on ties: earlier-added docs rank first.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive; got {limit}")
        query_terms = tokenize(query)
        if not query_terms:
            return []
        # De-duplicate query terms while preserving order.
        seen: set[str] = set()
        ordered_terms: list[str] = []
        for t in query_terms:
            if t not in seen:
                seen.add(t)
                ordered_terms.append(t)

        results: list[SearchHit] = []
        for idx, doc in enumerate(self._docs):
            tf_map = self._term_freqs[idx]
            if require_all and not all(t in tf_map for t in ordered_terms):
                continue
            score, contribs = self._score(idx, ordered_terms)
            if score == 0.0:
                continue
            snippet = _build_snippet(doc.text, ordered_terms)
            results.append(SearchHit(doc=doc, score=score, terms=contribs, snippet=snippet))
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:limit]

    def get_doc(self, doc_id: str) -> RationaleDoc | None:
        idx = self._doc_id_to_idx.get(doc_id)
        if idx is None:
            return None
        return self._docs[idx]


# ── Snippet generation ───────────────────────────────────


def _build_snippet(text: str, query_terms: list[str], *, max_chars: int = 200) -> str:
    """Build a snippet centred on the first query-term match.

    Pin: snippet is the operator's first glance at *why* a result
    matched — show the matching context, not the start of the doc."""
    if not text:
        return ""
    lower = text.lower()
    first_pos = -1
    for term in query_terms:
        pos = lower.find(term)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos
    if first_pos == -1:
        # No literal substring match (tokeniser dropped a hit)
        # — return a head snippet so operators still see content.
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    half = max_chars // 2
    start = max(0, first_pos - half)
    end = min(len(text), start + max_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


# ── Render helper ─────────────────────────────────────────


def render_results(hits: Iterable[SearchHit]) -> str:
    """CLI / Slack-ready text payload for a result list."""
    hit_list = list(hits)
    if not hit_list:
        return "=== Rationale search ===\n(no matches)"
    lines = ["=== Rationale search ==="]
    for i, hit in enumerate(hit_list, start=1):
        lines.append(f"  {i:>2}. {hit.doc.doc_id} score={hit.score:.3f}")
        if hit.snippet:
            lines.append(f"      …{hit.snippet}")
        if hit.terms:
            term_pairs = ", ".join(f"{t.term}×{t.tf}" for t in hit.terms)
            lines.append(f"      matched: {term_pairs}")
    return "\n".join(lines)


__all__ = [
    "RationaleDoc",
    "RationaleIndex",
    "SearchHit",
    "TermContribution",
    "render_results",
    "tokenize",
]
