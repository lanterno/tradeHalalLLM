"""Rule-based sentiment classifier for SEC filing text.

Round-4 wave 4.H: today's bot reads sentiment from headlines
(CryptoPanic, Reddit) and aggregates by symbol. Headlines move
fast but are noisy; filings are slow but ground-truth. A
**filing-text classifier** lets the bot turn the EDGAR feed
(`trading/scheduler.py` already pulls 8-K material events into
the catalyst stream) into a per-filing sentiment score the LLM
prompt can quote.

Scope: pure rule-based. The model is a small set of
**operator-readable lexicons** (positive / negative / hedge
modifiers) with a per-occurrence score and an aggregation
formula. Pin: rules over a trained model because (a) operators
need to read the rule and challenge a verdict — opaque ML on a
compliance-adjacent surface is the wrong default; (b) the
classifier never exceeds 100 lines of audit-able dataclass +
regex; (c) re-training requires labelled SEC text we don't
have.

The classifier handles three concerns:

* **Tokenisation** — case-folded word-boundary split,
  identical to `core/rationale_search.py`'s tokeniser so a
  user searching "MACD" hits the same tokens the sentiment
  layer counted.
* **Lexicon scoring** — per-token contributions with a hedge
  modifier that flips the sign on negation contexts ("not
  bullish" → bearish).
* **Aggregate scoring** — bounded `[-1, 1]` normalised score
  + `Sentiment` enum (`BULLISH / NEUTRAL / BEARISH`) +
  per-rule attribution so the dashboard can render "matched
  3× 'beat estimates' (bullish), 1× 'risk factors' (bearish)".

Halal alignment: the classifier is read-only signal generation;
it never opens a position or screens an asset. The lexicons
are publicly checkable English phrases; no operator-IP or
PII handling.

Pure-Python; no NumPy / DB / network. The lexicons are
module-level frozensets so a runtime mutation can't tilt the
classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

# ── Vocabulary ────────────────────────────────────────────


class Sentiment(str, Enum):
    """The three-state sentiment label.

    Pin: BULLISH / BEARISH / NEUTRAL — same vocabulary the
    existing `sentiment/` Reddit / CryptoPanic modules use, so
    a downstream consumer (the LLM prompt) sees one unified
    tag set across all sentiment sources."""

    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


# Score bands. Pin: ≥ 0.15 → BULLISH, ≤ -0.15 → BEARISH; the
# narrow ±0.15 zone keeps the classifier from declaring a
# definite tilt on a sparse signal.
_BULLISH_THRESHOLD = 0.15
_BEARISH_THRESHOLD = -0.15


# ── Lexicons ──────────────────────────────────────────────


# Positive phrases. Score weight 1.0 unless overridden in
# `_PHRASE_OVERRIDES`. Pin: phrases (multi-word) capture
# domain idioms ("beat estimates" is bullish; either word
# alone is ambiguous).
_POSITIVE_PHRASES: frozenset[str] = frozenset(
    {
        "beat estimates",
        "raised guidance",
        "record revenue",
        "record earnings",
        "record profit",
        "exceeded expectations",
        "above consensus",
        "upgraded outlook",
        "strong demand",
        "robust growth",
        "accelerated growth",
        "margin expansion",
        "operating leverage",
        "share buyback",
        "dividend increase",
        "favorable trend",
        "secular tailwind",
        "competitive moat",
        "expanding market",
        "cost discipline",
    }
)


# Negative phrases. Score weight 1.0 unless overridden.
_NEGATIVE_PHRASES: frozenset[str] = frozenset(
    {
        "missed estimates",
        "lowered guidance",
        "revenue declined",
        "operating loss",
        "below consensus",
        "downgraded outlook",
        "weak demand",
        "slowing growth",
        "margin compression",
        "going concern",
        "material weakness",
        "litigation risk",
        "regulatory action",
        "subpoena received",
        "cyber incident",
        "data breach",
        "supply chain disruption",
        "inventory buildup",
        "restructuring charge",
        "impairment charge",
        "writedown",
        "covenant breach",
        "credit downgrade",
        "delisted",
        "bankruptcy",
        "fraud",
    }
)


# Strong-signal phrases get a 2.0× weight. Pin: keep the list
# tight — every phrase here pulls the score noticeably, so
# adding one is consequential. New entries go through review.
_PHRASE_WEIGHTS: dict[str, float] = {
    "going concern": 2.0,
    "material weakness": 2.0,
    "fraud": 2.5,
    "bankruptcy": 2.5,
    "delisted": 2.0,
    "covenant breach": 2.0,
    "subpoena received": 1.5,
    "data breach": 1.5,
    "cyber incident": 1.5,
    "record revenue": 1.5,
    "record earnings": 1.5,
    "record profit": 1.5,
    "raised guidance": 1.5,
    "lowered guidance": 1.5,
    "share buyback": 1.5,
    "dividend increase": 1.5,
}


# Negation hedges. When a positive phrase appears within
# `_NEGATION_WINDOW` tokens after a hedge, its score flips sign.
# Pin: the window is small (4 tokens) — "not just bullish but
# also..." should NOT count as negation; only direct
# adjacency.
_NEGATION_HEDGES: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "none",
        "nothing",
        "never",
        "without",
        "unable",
        "fails",
        "failing",
        "lacks",
        "absent",
    }
)


_NEGATION_WINDOW = 4


# Phrase-discovery regex: a phrase is one or more whitespace-
# separated tokens; we tokenise the text, then sliding-window
# match phrase prefixes.


# ── Tokenisation ──────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def _tokenize(text: str) -> list[str]:
    """Lower-case word-boundary tokenisation. Pin: identical to
    `core/rationale_search.py`'s tokeniser so a "MACD"-style
    keyword search hits the same tokens the sentiment layer
    counted (no coverage drift)."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _phrase_to_tokens(phrase: str) -> tuple[str, ...]:
    """Convert a phrase to its lower-cased token tuple."""
    return tuple(_tokenize(phrase))


# Pre-compute phrase-token tuples for O(1) lookup.
_POSITIVE_PHRASE_TOKENS: dict[tuple[str, ...], str] = {
    _phrase_to_tokens(p): p for p in _POSITIVE_PHRASES
}
_NEGATIVE_PHRASE_TOKENS: dict[tuple[str, ...], str] = {
    _phrase_to_tokens(p): p for p in _NEGATIVE_PHRASES
}


# ── Output ────────────────────────────────────────────────


@dataclass(frozen=True)
class PhraseHit:
    """One occurrence of a known phrase in the text.

    ``contribution`` is the signed score this hit contributed
    to the aggregate (positive for bullish, negative for
    bearish; sign-flipped if a negation hedge fired).
    ``token_index`` is the position in the tokenised text — used
    by the dashboard's "show me the matching context" feature.
    """

    phrase: str
    polarity: Sentiment  # BULLISH or BEARISH; never NEUTRAL
    weight: float
    token_index: int
    negated: bool
    contribution: float


@dataclass(frozen=True)
class SentimentResult:
    """Aggregated sentiment with per-hit attribution."""

    label: Sentiment
    score: float  # bounded [-1, 1]
    hits: list[PhraseHit] = field(default_factory=list)
    raw_positive: float = 0.0
    raw_negative: float = 0.0
    token_count: int = 0
    summary: str = ""

    @property
    def is_neutral(self) -> bool:
        return self.label == Sentiment.NEUTRAL


# ── Scorer ────────────────────────────────────────────────


def _has_negation_in_window(tokens: list[str], end_index: int, window: int) -> bool:
    """Check whether any token in the `window` positions before
    `end_index` is a negation hedge.

    Pin: `end_index` is the START of the phrase; we look at the
    `window` tokens immediately preceding it. A hedge further
    back doesn't apply (the natural-language prosody of
    negation is local)."""
    start = max(0, end_index - window)
    for i in range(start, end_index):
        if tokens[i] in _NEGATION_HEDGES:
            return True
    return False


def _scan_phrases(
    tokens: list[str],
    phrase_lookup: dict[tuple[str, ...], str],
    polarity: Sentiment,
) -> list[PhraseHit]:
    """Sliding-window scan for any of the phrases in
    `phrase_lookup`. Each phrase's tokens are pre-computed.

    Pin: longest-match-first per starting position so "record
    revenue growth" scores once as `record revenue` (the longer
    phrase wins) rather than overlapping into separate
    matches."""
    hits: list[PhraseHit] = []
    n = len(tokens)
    i = 0
    while i < n:
        # Try the longest matching phrase first.
        best_match: tuple[tuple[str, ...], str] | None = None
        for phrase_tokens, original in phrase_lookup.items():
            length = len(phrase_tokens)
            if i + length > n:
                continue
            if tuple(tokens[i : i + length]) == phrase_tokens:
                if best_match is None or len(phrase_tokens) > len(best_match[0]):
                    best_match = (phrase_tokens, original)
        if best_match is None:
            i += 1
            continue
        phrase_tokens, original = best_match
        weight = _PHRASE_WEIGHTS.get(original, 1.0)
        negated = _has_negation_in_window(tokens, i, _NEGATION_WINDOW)
        # When negated, sign flips: a negated bullish becomes
        # bearish contribution.
        sign = -1.0 if negated else 1.0
        polarity_sign = 1.0 if polarity == Sentiment.BULLISH else -1.0
        contribution = sign * polarity_sign * weight
        hits.append(
            PhraseHit(
                phrase=original,
                polarity=polarity,
                weight=weight,
                token_index=i,
                negated=negated,
                contribution=contribution,
            )
        )
        i += len(phrase_tokens)
    return hits


def _normalise_score(raw_positive: float, raw_negative: float) -> float:
    """Map raw scores to bounded `[-1, 1]`.

    Pin: the formula is `(positive - negative) / (positive +
    negative + 1)` — the `+ 1` damping prevents a single hit
    from saturating the score, and the difference-over-sum
    keeps the bound symmetric. A document with strong positive
    AND strong negative hits lands closer to 0 than either
    extreme — operators should see "mixed" rather than the
    larger side."""
    total = raw_positive + raw_negative
    if total == 0:
        return 0.0
    return (raw_positive - raw_negative) / (total + 1.0)


def _label_from_score(score: float) -> Sentiment:
    """Pin the threshold bands."""
    if score >= _BULLISH_THRESHOLD:
        return Sentiment.BULLISH
    if score <= _BEARISH_THRESHOLD:
        return Sentiment.BEARISH
    return Sentiment.NEUTRAL


def _build_summary(
    label: Sentiment,
    score: float,
    hit_count: int,
    token_count: int,
) -> str:
    if hit_count == 0:
        return f"neutral · no matching phrases in {token_count} tokens"
    return f"{label.value} · score {score:+.2f} from {hit_count} matches in {token_count} tokens"


# ── Entry point ───────────────────────────────────────────


def classify(text: str) -> SentimentResult:
    """Score one text snippet.

    Empty / whitespace-only input → NEUTRAL with score 0.0
    and no hits. The aggregate handles this so callers can
    feed in untrusted input without guarding."""
    tokens = _tokenize(text)
    if not tokens:
        return SentimentResult(
            label=Sentiment.NEUTRAL,
            score=0.0,
            hits=[],
            raw_positive=0.0,
            raw_negative=0.0,
            token_count=0,
            summary="neutral · empty input",
        )

    pos_hits = _scan_phrases(tokens, _POSITIVE_PHRASE_TOKENS, Sentiment.BULLISH)
    neg_hits = _scan_phrases(tokens, _NEGATIVE_PHRASE_TOKENS, Sentiment.BEARISH)
    all_hits = sorted(pos_hits + neg_hits, key=lambda h: h.token_index)

    raw_positive = sum(max(0.0, h.contribution) for h in all_hits)
    raw_negative = sum(max(0.0, -h.contribution) for h in all_hits)
    score = _normalise_score(raw_positive, raw_negative)
    label = _label_from_score(score)

    return SentimentResult(
        label=label,
        score=score,
        hits=all_hits,
        raw_positive=raw_positive,
        raw_negative=raw_negative,
        token_count=len(tokens),
        summary=_build_summary(label, score, len(all_hits), len(tokens)),
    )


# ── Per-symbol aggregator ─────────────────────────────────


@dataclass(frozen=True)
class SymbolSentiment:
    """Aggregate of multiple filings' sentiment for one symbol.

    Pin: `score` averages with confidence-weighting (filings
    with more matches dominate). `dominant_label` picks the
    bucket that best represents the cohort.
    """

    symbol: str
    filings_analysed: int
    score: float
    dominant_label: Sentiment
    bullish_count: int
    neutral_count: int
    bearish_count: int


def aggregate_for_symbol(
    *,
    symbol: str,
    filings: Iterable[str],
) -> SymbolSentiment:
    """Aggregate sentiment across multiple filings for one
    symbol.

    Pin: a symbol with no filings returns `dominant_label =
    NEUTRAL` with score 0 — the operator's "no signal yet"
    state. Doesn't crash on an empty iterable."""
    results = [classify(f) for f in filings]
    if not results:
        return SymbolSentiment(
            symbol=symbol,
            filings_analysed=0,
            score=0.0,
            dominant_label=Sentiment.NEUTRAL,
            bullish_count=0,
            neutral_count=0,
            bearish_count=0,
        )

    # Confidence-weighted average: a filing with 5 hits weighs
    # more than one with 1.
    total_weight = 0.0
    weighted_sum = 0.0
    bullish = neutral = bearish = 0
    for r in results:
        weight = max(1.0, float(len(r.hits)))
        weighted_sum += r.score * weight
        total_weight += weight
        if r.label == Sentiment.BULLISH:
            bullish += 1
        elif r.label == Sentiment.BEARISH:
            bearish += 1
        else:
            neutral += 1

    avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    dominant = _label_from_score(avg_score)

    return SymbolSentiment(
        symbol=symbol,
        filings_analysed=len(results),
        score=avg_score,
        dominant_label=dominant,
        bullish_count=bullish,
        neutral_count=neutral,
        bearish_count=bearish,
    )


# ── Render helper ─────────────────────────────────────────


def render_result(result: SentimentResult) -> str:
    """One-line operator-readable summary suitable for log /
    Slack / a dashboard tile."""
    emoji = {
        Sentiment.BULLISH: "🟢",
        Sentiment.NEUTRAL: "🟡",
        Sentiment.BEARISH: "🔴",
    }[result.label]
    line = f"{emoji} {result.summary}"
    if result.hits:
        # Show top 3 contributions (largest |contribution| first).
        top = sorted(result.hits, key=lambda h: -abs(h.contribution))[:3]
        match_text = ", ".join(
            f"'{h.phrase}'{'×' if h.negated else ''}({h.contribution:+.1f})" for h in top
        )
        line += f"  matches: {match_text}"
    return line


__all__ = [
    "PhraseHit",
    "Sentiment",
    "SentimentResult",
    "SymbolSentiment",
    "aggregate_for_symbol",
    "classify",
    "render_result",
]
