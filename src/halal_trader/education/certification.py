"""Halal trader certification exam grader — Round-5 Wave 20.B.

Tiered certification ladder:
- HT-1 (Halal Trader 1) — beginner exam
- HT-2 — intermediate exam (prerequisite: HT-1)
- HT-3 — advanced exam (prerequisite: HT-2)

Each exam is multiple-choice; the grader scores answers, enforces a
pass threshold per exam, and tracks attempts with a cooldown to deter
brute-forcing.

Pinned semantics:

- **Closed-set CertTier ladder** — HT_1 / HT_2 / HT_3.
- **Closed-set QuestionKind** — SINGLE_CHOICE / MULTI_CHOICE / TRUE_FALSE.
- **Prerequisite ladder is strict**: HT_2 requires HT_1 pass;
  HT_3 requires HT_2 pass.
- **Pass threshold** is per-exam (default 0.70 for HT-1, 0.75 for HT-2,
  0.80 for HT-3 — harder for higher tiers).
- **Attempt cooldown** — failed attempts trigger a 24-hour cooldown
  before another attempt is permitted; configurable.
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render — candidate IDs masked; correct-
  answer keys never echoed in render output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class CertTier(str, Enum):
    """Closed-set certification ladder."""

    HT_1 = "ht_1"
    HT_2 = "ht_2"
    HT_3 = "ht_3"


_TIER_ORDER: dict[CertTier, int] = {
    CertTier.HT_1: 1,
    CertTier.HT_2: 2,
    CertTier.HT_3: 3,
}


_PREREQ: dict[CertTier, CertTier | None] = {
    CertTier.HT_1: None,
    CertTier.HT_2: CertTier.HT_1,
    CertTier.HT_3: CertTier.HT_2,
}


_DEFAULT_PASS_THRESHOLD: dict[CertTier, float] = {
    CertTier.HT_1: 0.70,
    CertTier.HT_2: 0.75,
    CertTier.HT_3: 0.80,
}


class QuestionKind(str, Enum):
    """Closed-set question kind ladder."""

    SINGLE_CHOICE = "single_choice"
    MULTI_CHOICE = "multi_choice"
    TRUE_FALSE = "true_false"


@dataclass(frozen=True)
class Question:
    """One exam question.

    `correct_keys` is the set of option-keys considered correct. For
    SINGLE_CHOICE and TRUE_FALSE this has exactly one element; for
    MULTI_CHOICE it can have multiple.
    """

    question_id: str
    kind: QuestionKind
    prompt: str
    option_keys: tuple[str, ...]
    correct_keys: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not self.option_keys:
            raise ValueError("option_keys must be non-empty")
        if len(set(self.option_keys)) != len(self.option_keys):
            raise ValueError("option_keys must be unique")
        if not self.correct_keys:
            raise ValueError("correct_keys must be non-empty")
        # Correct keys must be a subset of option keys.
        opt_set = set(self.option_keys)
        for k in self.correct_keys:
            if k not in opt_set:
                raise ValueError(f"correct_key {k!r} not in option_keys")
        # Kind-specific cardinality pins.
        if self.kind in (QuestionKind.SINGLE_CHOICE, QuestionKind.TRUE_FALSE):
            if len(self.correct_keys) != 1:
                raise ValueError(f"{self.kind.value} requires exactly one correct_key")
        if self.kind is QuestionKind.TRUE_FALSE and len(self.option_keys) != 2:
            raise ValueError("TRUE_FALSE requires exactly two option_keys")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True)
class Exam:
    """A full exam — tier + questions."""

    exam_id: str
    tier: CertTier
    questions: tuple[Question, ...]
    pass_threshold: float | None = None
    """Override default threshold; None falls back to tier default."""

    def __post_init__(self) -> None:
        if not self.exam_id or not self.exam_id.strip():
            raise ValueError("exam_id must be non-empty")
        if not self.questions:
            raise ValueError("exam must have at least one question")
        ids = [q.question_id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question_id in exam")
        if self.pass_threshold is not None and not 0.0 < self.pass_threshold < 1.0:
            raise ValueError("pass_threshold must be in (0, 1)")

    def effective_threshold(self) -> float:
        if self.pass_threshold is not None:
            return self.pass_threshold
        return _DEFAULT_PASS_THRESHOLD[self.tier]

    def total_weight(self) -> float:
        return sum(q.weight for q in self.questions)


@dataclass(frozen=True)
class Answer:
    """Candidate's answer to one question."""

    question_id: str
    selected_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if len(set(self.selected_keys)) != len(self.selected_keys):
            raise ValueError("selected_keys must be unique")


@dataclass(frozen=True)
class AttemptRecord:
    """One sat exam record."""

    attempt_id: str
    candidate_id: str
    exam_id: str
    tier: CertTier
    started_at: datetime
    finished_at: datetime
    raw_score: float
    """Weighted points earned."""
    total_weight: float
    passed: bool

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.attempt_id.strip():
            raise ValueError("attempt_id must be non-empty")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if self.finished_at <= self.started_at:
            raise ValueError("finished_at must be after started_at")
        if self.raw_score < 0:
            raise ValueError("raw_score must be non-negative")
        if self.total_weight <= 0:
            raise ValueError("total_weight must be positive")
        if self.raw_score > self.total_weight + 1e-9:
            raise ValueError("raw_score cannot exceed total_weight")

    def percent_score(self) -> float:
        return self.raw_score / self.total_weight


def _question_correct(question: Question, answer: Answer) -> bool:
    """True iff the answer's selected_keys exactly match correct_keys.

    Pinned: MULTI_CHOICE is all-or-nothing — partial credit is not
    awarded; it muddies the pass/fail line and operators have asked
    for clarity over leniency.
    """
    return set(answer.selected_keys) == set(question.correct_keys)


def grade(
    exam: Exam,
    answers: Iterable[Answer],
    *,
    attempt_id: str,
    candidate_id: str,
    started_at: datetime,
    finished_at: datetime,
) -> AttemptRecord:
    """Grade an exam attempt and produce an `AttemptRecord`.

    Pinned: only questions with a matching answer can earn points;
    missing answers count as zero. Unknown answers (no matching
    question) are silently ignored — the operator's UI should
    prevent this, but the grader is defensive.
    """
    by_id = {a.question_id: a for a in answers}
    raw = 0.0
    for q in exam.questions:
        ans = by_id.get(q.question_id)
        if ans is None:
            continue
        if _question_correct(q, ans):
            raw += q.weight
    total = exam.total_weight()
    passed = (raw / total) >= exam.effective_threshold() - 1e-12
    return AttemptRecord(
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        exam_id=exam.exam_id,
        tier=exam.tier,
        started_at=started_at,
        finished_at=finished_at,
        raw_score=raw,
        total_weight=total,
        passed=passed,
    )


@dataclass(frozen=True)
class CandidateHistory:
    """A candidate's exam attempts across all tiers."""

    candidate_id: str
    attempts: tuple[AttemptRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        for a in self.attempts:
            if a.candidate_id != self.candidate_id:
                raise ValueError("attempt.candidate_id must match history.candidate_id")

    def attempts_for_tier(self, tier: CertTier) -> tuple[AttemptRecord, ...]:
        return tuple(a for a in self.attempts if a.tier is tier)

    def has_passed(self, tier: CertTier) -> bool:
        return any(a.passed for a in self.attempts_for_tier(tier))

    def highest_passed_tier(self) -> CertTier | None:
        for tier in (CertTier.HT_3, CertTier.HT_2, CertTier.HT_1):
            if self.has_passed(tier):
                return tier
        return None


def can_take(
    history: CandidateHistory,
    tier: CertTier,
    *,
    now: datetime,
    cooldown_hours: int = 24,
) -> tuple[bool, str]:
    """Can this candidate take `tier`'s exam right now?

    Returns (is_allowed, reason). Pinned reasons:
    - "missing prerequisite"
    - "already certified at this tier"
    - "cooldown active until <iso>"
    - "ok"
    """
    if cooldown_hours <= 0:
        raise ValueError("cooldown_hours must be positive")
    prereq = _PREREQ[tier]
    if prereq is not None and not history.has_passed(prereq):
        return False, f"missing prerequisite {prereq.value}"
    if history.has_passed(tier):
        return False, "already certified at this tier"
    failed = [a for a in history.attempts_for_tier(tier) if not a.passed]
    if failed:
        last = max(failed, key=lambda a: a.finished_at)
        cooldown_until = last.finished_at + timedelta(hours=cooldown_hours)
        if now < cooldown_until:
            return False, f"cooldown active until {cooldown_until.isoformat()}"
    return True, "ok"


@dataclass(frozen=True)
class Certificate:
    """Issued certificate."""

    certificate_id: str
    candidate_id: str
    tier: CertTier
    issued_on: datetime
    """Hash anchored on (candidate, tier, attempt_id) for tamper detection."""
    anchor_hash: str
    attempt_id: str


def _anchor(candidate_id: str, tier: CertTier, attempt_id: str) -> str:
    payload = json.dumps(
        {
            "candidate": candidate_id,
            "tier": tier.value,
            "attempt": attempt_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def issue_certificate(
    history: CandidateHistory,
    *,
    tier: CertTier,
    certificate_id: str,
    issued_on: datetime,
) -> Certificate:
    """Issue a certificate for the given tier; raises if the candidate
    hasn't passed."""
    passing = [a for a in history.attempts_for_tier(tier) if a.passed]
    if not passing:
        raise ValueError(f"candidate has no passing attempt at {tier.value}")
    best = max(passing, key=lambda a: a.percent_score())
    return Certificate(
        certificate_id=certificate_id,
        candidate_id=history.candidate_id,
        tier=tier,
        issued_on=issued_on,
        anchor_hash=_anchor(history.candidate_id, tier, best.attempt_id),
        attempt_id=best.attempt_id,
    )


def verify_certificate(cert: Certificate) -> bool:
    """True iff `cert.anchor_hash` matches the canonical derivation."""
    expected = _anchor(cert.candidate_id, cert.tier, cert.attempt_id)
    return cert.anchor_hash == expected


def _mask(candidate_id: str) -> str:
    if len(candidate_id) <= 4:
        return "***"
    return candidate_id[:2] + "…" + candidate_id[-2:]


def render_attempt(attempt: AttemptRecord) -> str:
    """Operator-readable attempt summary. Pin: does not echo answers
    or correct-answer keys."""
    result = "✅ PASS" if attempt.passed else "❌ FAIL"
    return (
        f"📝 {attempt.attempt_id} [{attempt.tier.value}] "
        f"{result} — {attempt.percent_score() * 100:.2f}% "
        f"({attempt.raw_score:.2f}/{attempt.total_weight:.2f})\n"
        f"  Candidate {_mask(attempt.candidate_id)} | "
        f"{attempt.started_at.isoformat()} → {attempt.finished_at.isoformat()}"
    )


def render_certificate(cert: Certificate) -> str:
    return (
        f"🎓 Certificate {cert.certificate_id} [{cert.tier.value}]\n"
        f"  Candidate: {_mask(cert.candidate_id)}\n"
        f"  Issued: {cert.issued_on.isoformat()}\n"
        f"  Anchor: {cert.anchor_hash[:16]}…"
    )
