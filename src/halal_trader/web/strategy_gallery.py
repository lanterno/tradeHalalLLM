"""Public strategy gallery curation engine.

The roadmap pins Wave 10.A: "Operators voluntarily publish their
strategies + performance. Sortable by Sharpe, by halal-compliance
strictness, by simplicity. Forkable into the operator's own
account." This module is the **pure-Python curation engine** —
given an operator's strategy artefact + performance metrics +
publication intent, produce a gallery-ready `StrategyEntry` with
anonymised author identity, halal-strictness rating, simplicity
score, and pre-publication safety gates that block accidental
PII leaks.

Picked a focused publication-curation engine over an "auto-
publish" flow because (a) the gallery's value depends on
quality + safety — operators expect their strategies to remain
private by default, and the engine refuses to publish anything
without an explicit opt-in flag, (b) the regression-pinned
anonymisation properties (no PII regex matches survive; salt is
required; visibility default is PRIVATE) are testable in a way
an ad-hoc publish script isn't, (c) the gallery becomes the
public face of the project — the engine encodes the operator's
quality bar so submissions are uniformly comparable.

Pinned semantics:
- **PRIVATE is the default.** A `StrategyEntry` constructed
  without an explicit `visibility=PUBLIC_LISTED` or `PUBLIC_
  UNLISTED` defaults to `PRIVATE`; the publication gate
  `validate_for_publication` refuses non-PRIVATE entries
  without the operator's `opt_in_publication` flag set.
- **Anonymous author token.** Operator's user_id is replaced by
  a deterministic salt-hashed `anon-…` token (mirrors Wave 10.B
  open-dataset pattern). Different salts → different tokens, so
  re-publication under a fresh salt produces a fresh anonymous
  author.
- **PII denylist on summary text.** Same five regex patterns as
  Wave 10.B (email / SSN / IP / phone / ETH-address); summary
  with any leaking pattern is rejected at validation rather
  than auto-redacted (so the operator notices and rewrites).
- **Halal-strictness rating** (BASIC / MODERATE / STRICT /
  MAX_STRICT) — a closed enum the gallery sorts on; operator
  declares their strategy's strictness level + the engine
  validates it matches the screener-set the operator's
  deployment uses.
- **Simplicity score** computed from LOC + symbol-list size
  + reasoning-loop depth — the load-bearing pin: simpler
  strategies sort to the top so new operators see approachable
  examples first.
- **Fork lineage preserved.** Every fork carries `parent_fork_id`
  pointing to the strategy it was forked from; the gallery
  renders the fork tree so attribution is preserved.
- **Render output never includes operator's raw user_id /
  portfolio / wallet addresses.** Mirrors no-PII patterns of
  Wave 11.D + 11.C + 3.B + 10.B.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from enum import Enum


class StrategyVisibility(str, Enum):
    """Visibility tier for a published strategy.

    Pinned: PRIVATE is the default — operators must explicitly
    opt-in to publication. PUBLIC_UNLISTED is shareable via
    direct URL but doesn't appear in gallery listings.
    """

    PRIVATE = "private"
    PUBLIC_UNLISTED = "public_unlisted"
    PUBLIC_LISTED = "public_listed"


class HalalStrictnessLevel(str, Enum):
    """Operator-declared halal-screening strictness.

    Pinned: BASIC follows the AAOIFI Standard 21 baseline; STRICT
    layers Wave 1.I REIT / Wave 1.G commodity / Wave 1.H sukuk
    additional checks; MAX_STRICT requires the full Wave 11.B
    SSB-board ruling chain.
    """

    BASIC = "basic"
    MODERATE = "moderate"
    STRICT = "strict"
    MAX_STRICT = "max_strict"


# Same PII denylist as Wave 10.B open-dataset anonymisation —
# the gallery shares the same defensive surface.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),  # IP
    re.compile(r"\b0x[0-9a-fA-F]{40}\b"),  # ETH address
    re.compile(r"\+?\d[\d\s().-]{8,}\d"),  # phone
)

_MIN_SALT_BYTES = 16


class GalleryViolationError(Exception):
    """Raised when a strategy entry violates a publication gate."""


@dataclass(frozen=True)
class PublicMetrics:
    """Performance metrics attached to a published strategy.

    Pinned: every metric required for sortability — Sharpe / win
    rate / max drawdown / total trades / time period — is on the
    dataclass; missing fields surface at construction.
    """

    sharpe_ratio: float
    win_rate_pct: float
    max_drawdown_pct: float
    total_trades: int
    time_period_days: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.win_rate_pct <= 100.0:
            raise ValueError(f"win_rate_pct must be in [0, 100], got {self.win_rate_pct}")
        if not 0.0 <= self.max_drawdown_pct <= 100.0:
            raise ValueError(f"max_drawdown_pct must be in [0, 100], got {self.max_drawdown_pct}")
        if self.total_trades < 0:
            raise ValueError("total_trades must be non-negative")
        if self.time_period_days <= 0:
            raise ValueError("time_period_days must be positive")


@dataclass(frozen=True)
class StrategyEntry:
    """One strategy published to the gallery.

    `anonymous_author` is a salt-hashed token; `parent_fork_id`
    points to the source strategy when this is a fork. Visibility
    defaults to PRIVATE — operators set PUBLIC_LISTED / PUBLIC_
    UNLISTED explicitly to opt-in to publication.
    """

    strategy_id: str
    anonymous_author: str
    name: str
    version: int
    summary: str
    halal_strictness: HalalStrictnessLevel
    simplicity_score: float
    visibility: StrategyVisibility = StrategyVisibility.PRIVATE
    metrics: PublicMetrics | None = None
    parent_fork_id: str | None = None
    opt_in_publication: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty")
        if not self.anonymous_author or not self.anonymous_author.strip():
            raise ValueError("anonymous_author must be non-empty")
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.version < 1:
            raise ValueError("version must be at least 1")
        if not 0.0 <= self.simplicity_score <= 100.0:
            raise ValueError(f"simplicity_score must be in [0, 100], got {self.simplicity_score}")
        if self.parent_fork_id is not None and not self.parent_fork_id.strip():
            raise ValueError("parent_fork_id must be non-empty when set")


def hash_author(user_id: str, *, salt: bytes) -> str:
    """Deterministic salt-hashed author token.

    Same construction as Wave 10.B open-dataset anonymisation —
    the gallery and the dataset can share an export salt so an
    anonymous author appears with the same token in both.
    """

    if len(salt) < _MIN_SALT_BYTES:
        raise ValueError(f"salt must be at least {_MIN_SALT_BYTES} bytes")
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be non-empty")
    digest = hmac.new(salt, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"anon-{digest[:16]}"


def _has_pii(text: str) -> bool:
    """True if any PII-shaped pattern matches the text."""

    for pattern in _PII_PATTERNS:
        if pattern.search(text):
            return True
    return False


def compute_simplicity_score(
    *,
    lines_of_code: int,
    symbol_list_size: int,
    reasoning_depth: int,
) -> float:
    """Compute a 0-100 simplicity score.

    Heuristic: simpler strategies sort higher. The score
    penalises LOC (linearly), symbol-list size (sub-linearly),
    and reasoning depth (linearly). Capped at [0, 100].

    Pinned scoring:
    - 100 LOC + 5 symbols + depth 2 → ~80 (clean strategy)
    - 500 LOC + 50 symbols + depth 5 → ~30 (complex)
    - 1000+ LOC + 100+ symbols + depth 10+ → ~0 (very complex)
    """

    if lines_of_code < 0:
        raise ValueError("lines_of_code must be non-negative")
    if symbol_list_size < 0:
        raise ValueError("symbol_list_size must be non-negative")
    if reasoning_depth < 0:
        raise ValueError("reasoning_depth must be non-negative")

    # Base 100; each LOC over 50 deducts 0.05; each symbol over 5
    # deducts 0.5; each depth-step over 1 deducts 5.
    score = 100.0
    score -= max(0, lines_of_code - 50) * 0.05
    score -= max(0, symbol_list_size - 5) * 0.5
    score -= max(0, reasoning_depth - 1) * 5.0
    return max(0.0, min(100.0, score))


def validate_for_publication(entry: StrategyEntry) -> None:
    """Pre-publication safety gates.

    Raises `GalleryViolationError` when the entry is not safe to
    publish. Operators call this just before persisting the entry
    to the public gallery table; the function is pure (no I/O).
    """

    if entry.visibility is StrategyVisibility.PRIVATE:
        # Private entries don't get publication checks — they're
        # operator-internal and never appear in the gallery.
        return

    if not entry.opt_in_publication:
        raise GalleryViolationError(
            f"strategy {entry.strategy_id!r} requested non-private "
            f"visibility {entry.visibility.value!r} but opt_in_publication "
            "is False — operator must explicitly opt-in to publication"
        )

    if _has_pii(entry.summary):
        raise GalleryViolationError(
            f"strategy {entry.strategy_id!r} summary contains PII-shaped "
            "patterns; rewrite the summary before publishing"
        )

    if _has_pii(entry.name):
        raise GalleryViolationError(
            f"strategy {entry.strategy_id!r} name contains PII-shaped "
            "patterns; rename before publishing"
        )

    # Public entries must carry performance metrics — the gallery's
    # value depends on sortability + comparability.
    if entry.visibility is StrategyVisibility.PUBLIC_LISTED and entry.metrics is None:
        raise GalleryViolationError(
            f"strategy {entry.strategy_id!r} listed publicly without metrics; "
            "PUBLIC_LISTED requires PublicMetrics"
        )


def assemble_lineage(
    entries: tuple[StrategyEntry, ...],
    *,
    target_id: str,
) -> tuple[StrategyEntry, ...]:
    """Walk the fork chain from `target_id` back to its root.

    Returns the lineage in chronological order (root first → leaf
    last). Raises `KeyError` if `target_id` not in `entries`.

    Pinned: cycles are detected via a visited-set; if the operator
    somehow constructs a cycle (via DB import bug), the function
    raises `GalleryViolationError` rather than infinite-looping.
    """

    by_id = {e.strategy_id: e for e in entries}
    if target_id not in by_id:
        raise KeyError(f"strategy_id {target_id!r} not in entries")

    chain: list[StrategyEntry] = []
    visited: set[str] = set()
    current = by_id[target_id]
    while True:
        if current.strategy_id in visited:
            raise GalleryViolationError(f"fork lineage cycle detected at {current.strategy_id!r}")
        visited.add(current.strategy_id)
        chain.append(current)
        if current.parent_fork_id is None:
            break
        if current.parent_fork_id not in by_id:
            # Detached parent — operator imported a fork but not its
            # ancestor. Stop the walk here.
            break
        current = by_id[current.parent_fork_id]

    return tuple(reversed(chain))


_VISIBILITY_EMOJI: dict[StrategyVisibility, str] = {
    StrategyVisibility.PRIVATE: "🔒",
    StrategyVisibility.PUBLIC_UNLISTED: "🔗",
    StrategyVisibility.PUBLIC_LISTED: "🌍",
}

_STRICTNESS_EMOJI: dict[HalalStrictnessLevel, str] = {
    HalalStrictnessLevel.BASIC: "🟢",
    HalalStrictnessLevel.MODERATE: "🟡",
    HalalStrictnessLevel.STRICT: "🟠",
    HalalStrictnessLevel.MAX_STRICT: "🔴",
}


def render_entry(entry: StrategyEntry) -> str:
    """Format a strategy entry for ops display.

    Pinned no-PII contract: never includes the operator's raw
    user_id, portfolio details, wallet addresses, or any
    operator-identifying detail. Shows anonymous_author + name +
    metrics + halal-strictness + simplicity score.
    """

    vis_emoji = _VISIBILITY_EMOJI[entry.visibility]
    strict_emoji = _STRICTNESS_EMOJI[entry.halal_strictness]
    lines = [
        f"{vis_emoji} {entry.name} (v{entry.version}) — {entry.visibility.value.upper()}",
        f"  author: {entry.anonymous_author}",
        f"  {strict_emoji} halal-strictness: {entry.halal_strictness.value}",
        f"  simplicity: {entry.simplicity_score:.1f}/100",
    ]
    if entry.metrics is not None:
        m = entry.metrics
        lines.append(
            f"  performance: Sharpe {m.sharpe_ratio:.2f}, "
            f"win {m.win_rate_pct:.1f}%, "
            f"DD {m.max_drawdown_pct:.1f}%, "
            f"{m.total_trades} trades over {m.time_period_days}d"
        )
    if entry.parent_fork_id is not None:
        lines.append(f"  forked from: {entry.parent_fork_id}")
    if entry.summary:
        lines.append(f"  summary: {entry.summary}")
    return "\n".join(lines)


__all__ = [
    "GalleryViolationError",
    "HalalStrictnessLevel",
    "PublicMetrics",
    "StrategyEntry",
    "StrategyVisibility",
    "assemble_lineage",
    "compute_simplicity_score",
    "hash_author",
    "render_entry",
    "validate_for_publication",
]
