"""Open-dataset anonymisation engine.

The roadmap pins Wave 10.B: "The aggregated, anonymised LLM
decision history becomes a public dataset. Academic research
community downloads it, cites the platform. Halal-finance
researchers get a goldmine of empirical data." This module is
the **pure-Python anonymisation engine** — given the operator's
raw `LlmDecision` rows, produce a public-research-safe dataset
where (a) per-user identifiers are stripped to deterministic
salt-hashed tokens, (b) free-form text is scrubbed of PII via a
denylist, (c) USD amounts are bucketed, (d) timestamps are
rounded to the hour, and (e) k-anonymity is enforced (rows whose
quasi-identifier combination doesn't appear in ≥ k other rows
are dropped).

Picked a focused anonymisation engine over an "export everything
the operator has" flow because (a) academic users want a clean,
de-identified, cite-able dataset — they don't need raw operator
data, (b) the regression-pinned anonymisation properties (no
PII regex matches survive; salt is required; k-anonymity floor)
are testable in a way an ad-hoc export script isn't, (c) the
explicit AnonymisationPolicy lets operators tune the
aggressiveness per their compliance posture (some operators may
want stricter k=10 for higher-risk publication).

Pinned semantics:
- **Salt required.** Anonymisation requires an operator-supplied
  salt; without it, the user_id hash is reversible by anyone
  with a list of plausible user_ids. The constructor refuses
  empty / short-salt input.
- **Deterministic hash within a single export.** The same
  user_id maps to the same anonymised token for one run, so
  researchers can correlate the same anonymised user's decisions
  across the dataset. Across runs with different salts, the
  mapping changes — so re-exports can't be cross-referenced.
- **PII denylist applied to free-form text.** Email-shaped,
  SSN-shaped, IP-shaped, phone-shaped patterns matched and
  replaced with `<redacted>` placeholder. The pin guards
  against operator rationale containing accidental PII.
- **k-anonymity floor (default k=5).** A row whose quasi-
  identifier tuple (anonymous_user / sector / regime) doesn't
  appear in ≥ k other rows is dropped. Operators tune k upward
  for stricter publications; k=1 disables the check (operator
  opt-out for non-public internal exports).
- **Timestamp rounded to hour.** Reduces timing-correlation
  attack surface — researchers see the hour but not the minute.
- **USD buckets.** `MICRO < $100`, `SMALL $100-1k`, `MEDIUM
  $1k-10k`, `LARGE $10k-100k`, `WHALE > $100k` — coarse-enough
  to prevent re-identification via amount fingerprints.
- **Render output excludes raw values.** The engine's pure
  output is the anonymised dataset; never the raw input.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class USDBucket(str, Enum):
    """Coarse bucket for USD amounts."""

    MICRO = "micro"  # < $100
    SMALL = "small"  # $100-$1k
    MEDIUM = "medium"  # $1k-$10k
    LARGE = "large"  # $10k-$100k
    WHALE = "whale"  # > $100k


def _bucket_usd(amount_usd: float) -> USDBucket:
    """Map a USD amount to its bucket."""

    if amount_usd < 100:
        return USDBucket.MICRO
    if amount_usd < 1_000:
        return USDBucket.SMALL
    if amount_usd < 10_000:
        return USDBucket.MEDIUM
    if amount_usd < 100_000:
        return USDBucket.LARGE
    return USDBucket.WHALE


# PII regex patterns — applied to free-form text fields like
# rationale. The patterns intentionally err on the side of false
# positives (better to redact a non-PII string than leak a real
# email).
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<redacted-email>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<redacted-ssn>"),
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "<redacted-ip>",
    ),
    (
        re.compile(r"\+?\d[\d\s().-]{8,}\d"),
        "<redacted-phone>",
    ),
    (
        re.compile(r"\b0x[0-9a-fA-F]{40}\b"),
        "<redacted-eth-address>",
    ),
)

_MIN_SALT_BYTES = 16


@dataclass(frozen=True)
class RawDecision:
    """One raw LLM-decision row from the operator's database."""

    decision_id: str
    user_id: str
    timestamp: datetime
    symbol: str
    sector: str
    regime: str
    action: str  # buy / sell / hold
    notional_usd: float
    rationale: str

    def __post_init__(self) -> None:
        if not self.decision_id or not self.decision_id.strip():
            raise ValueError("decision_id must be non-empty")
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.notional_usd < 0:
            raise ValueError("notional_usd must be non-negative")


@dataclass(frozen=True)
class AnonymisedDecision:
    """Public-dataset row.

    `anonymous_user` is a deterministic salt-hashed token derived
    from the raw user_id — researchers can correlate decisions by
    the same anonymous user within the export but cannot recover
    the operator's user_id without the salt.
    """

    anonymous_user: str
    timestamp_hour: datetime  # rounded to hour
    symbol: str
    sector: str
    regime: str
    action: str
    usd_bucket: USDBucket
    rationale_redacted: str


@dataclass(frozen=True)
class AnonymisationPolicy:
    """Operator-tunable policy."""

    k_anonymity_floor: int = 5
    enable_pii_redaction: bool = True
    bucket_usd_amounts: bool = True

    def __post_init__(self) -> None:
        if self.k_anonymity_floor < 1:
            raise ValueError("k_anonymity_floor must be at least 1")


DEFAULT_POLICY = AnonymisationPolicy()


@dataclass(frozen=True)
class AnonymisationResult:
    """Output of the anonymisation pass."""

    decisions: tuple[AnonymisedDecision, ...]
    raw_input_count: int
    dropped_for_k_anonymity: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _hash_user(user_id: str, *, salt: bytes) -> str:
    """Deterministic salt-hashed user token.

    HMAC-SHA256 of user_id keyed by salt; truncated to first 16
    hex chars (still 64 bits of entropy — collision-resistant
    within the dataset size of any plausible export).
    """

    digest = hmac.new(salt, user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"anon-{digest[:16]}"


def _redact_pii(text: str) -> str:
    """Apply the PII denylist regexes."""

    redacted = text
    for pattern, replacement in _PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _round_to_hour(dt: datetime) -> datetime:
    """Round a timezone-aware datetime down to the hour."""

    return dt.replace(minute=0, second=0, microsecond=0)


def anonymise_decision(
    raw: RawDecision,
    *,
    salt: bytes,
    policy: AnonymisationPolicy = DEFAULT_POLICY,
) -> AnonymisedDecision:
    """Anonymise a single decision row.

    Pure transform: takes a `RawDecision`, returns an
    `AnonymisedDecision` with the operator-identifying fields
    stripped, the rationale redacted, the timestamp rounded, and
    the USD amount bucketed.
    """

    if len(salt) < _MIN_SALT_BYTES:
        raise ValueError(f"salt must be at least {_MIN_SALT_BYTES} bytes")

    rationale = _redact_pii(raw.rationale) if policy.enable_pii_redaction else raw.rationale
    bucket = _bucket_usd(raw.notional_usd)

    return AnonymisedDecision(
        anonymous_user=_hash_user(raw.user_id, salt=salt),
        timestamp_hour=_round_to_hour(raw.timestamp),
        symbol=raw.symbol,
        sector=raw.sector,
        regime=raw.regime,
        action=raw.action,
        usd_bucket=bucket,
        rationale_redacted=rationale,
    )


def anonymise_dataset(
    raw_decisions: tuple[RawDecision, ...],
    *,
    salt: bytes,
    policy: AnonymisationPolicy = DEFAULT_POLICY,
) -> AnonymisationResult:
    """Anonymise + k-anonymity-filter a dataset.

    The k-anonymity filter drops rows whose quasi-identifier tuple
    (anonymous_user, sector, regime) doesn't appear in ≥ k rows
    total. Researchers cite this as the "k=5 anonymisation"
    standard — it prevents re-identification through unique
    combinations.
    """

    if len(salt) < _MIN_SALT_BYTES:
        raise ValueError(f"salt must be at least {_MIN_SALT_BYTES} bytes")

    warnings: list[str] = []
    anonymised = tuple(anonymise_decision(raw, salt=salt, policy=policy) for raw in raw_decisions)

    # k-anonymity filter
    qid_counts: Counter[tuple[str, str, str]] = Counter()
    for d in anonymised:
        qid_counts[(d.anonymous_user, d.sector, d.regime)] += 1

    if policy.k_anonymity_floor <= 1:
        # Operator opt-out — return everything
        return AnonymisationResult(
            decisions=anonymised,
            raw_input_count=len(raw_decisions),
            dropped_for_k_anonymity=0,
            warnings=tuple(warnings),
        )

    kept: list[AnonymisedDecision] = []
    dropped = 0
    for d in anonymised:
        if qid_counts[(d.anonymous_user, d.sector, d.regime)] >= policy.k_anonymity_floor:
            kept.append(d)
        else:
            dropped += 1

    if dropped > 0:
        warnings.append(
            f"dropped {dropped} rows below k={policy.k_anonymity_floor} anonymity threshold"
        )

    return AnonymisationResult(
        decisions=tuple(kept),
        raw_input_count=len(raw_decisions),
        dropped_for_k_anonymity=dropped,
        warnings=tuple(warnings),
    )


def render_summary(result: AnonymisationResult) -> str:
    """Operator-facing summary of an anonymisation run.

    Pinned no-raw-data contract: never includes raw user_ids,
    raw rationale text, or raw USD amounts. Shows counts only.
    """

    lines = [
        f"📊 dataset anonymisation: {len(result.decisions)} rows out of "
        f"{result.raw_input_count} raw input",
    ]
    if result.dropped_for_k_anonymity > 0:
        lines.append(f"  dropped: {result.dropped_for_k_anonymity} (k-anonymity)")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  · {w}")
    if result.decisions:
        # Bucket distribution (counts only)
        bucket_counts: Counter[USDBucket] = Counter(d.usd_bucket for d in result.decisions)
        lines.append("  USD bucket distribution:")
        for bucket in USDBucket:
            count = bucket_counts.get(bucket, 0)
            if count > 0:
                lines.append(f"    {bucket.value}: {count}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "AnonymisationPolicy",
    "AnonymisationResult",
    "AnonymisedDecision",
    "RawDecision",
    "USDBucket",
    "anonymise_dataset",
    "anonymise_decision",
    "render_summary",
]
