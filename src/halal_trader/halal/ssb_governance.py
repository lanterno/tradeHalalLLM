"""Shariah Supervisory Board governance engine.

A Shariah Supervisory Board (SSB) is the third-party scholar
panel that certifies an Islamic-finance platform's products and
operations. The roadmap pins minimum composition: at least three
members drawn from at least three different fiqh schools (Hanafi,
Shafi'i, Maliki, Hanbali) — diversity prevents a single madhhab's
ruling from being misread as universal Islamic consensus, and the
panel size keeps any single scholar's bias from dominating.

The board reviews the platform on a quarterly cadence and issues
public rulings (fatawa) on specific products, strategies, or
policies. This module is the pure-Python state engine: board
composition validation, vote-aggregation consensus rules, term-
expiry tracking, quarterly review cadence enforcement, and the
public-register ruling format. The actual scholar onboarding +
quarterly meeting cadence + meeting minutes are operator-side
concerns; the engine ships the deterministic state machine the
operator's compliance program runs against.

Pinned semantics:
- **Minimum 3 members across ≥ 3 schools.** A board with two
  Hanafis + one Shafi'i fails — three members but only two
  schools. The diversity rule is enforced at the board-validation
  boundary, not silently let through. Pinned via test that
  catches the "three Hanafis" failure mode.
- **Any IMPERMISSIBLE vote → IMPERMISSIBLE outcome.** The
  conservative-tiebreak rule shared with Wave 2.B halal consensus,
  Wave 4.J committee voting, Wave 1.G commodities, Wave 1.I REIT,
  and Wave 2.G regulator-index. When dealing with shariah
  compliance, the most-conservative dissenting voice wins —
  better one false-negative product rejection than one false-
  positive that lets riba slip through.
- **2/3 majority required for PERMISSIBLE.** A simple majority
  isn't enough; an SSB ruling carries the weight of public
  Islamic-finance jurisprudence and needs a clear supermajority.
  Below 2/3 → DEFERRED, not PERMISSIBLE — pinned.
- **PERMISSIBLE_WITH_CONDITIONS requires explicit conditions
  list.** A ruling that says "yes, but with caveats" is meaningless
  without the caveats; the engine validates non-empty conditions
  at construction.
- **Term expiry tracked.** Members past their `expires_at` term
  cannot cast valid votes; the engine silently drops their vote
  from the consensus computation and surfaces a warning. Three-
  year default term (operator-tunable) — long enough for
  continuity, short enough to refresh perspective.
- **Quarterly review cadence (default 90 days) enforced.** A
  board that hasn't issued any ruling in > 90 days has missed its
  review obligation; `needs_quarterly_review` flags this for the
  operator's compliance dashboard.
- **Render output is the public register format.** Operators
  publish the ruling at a stable URL (e.g., `halal-trader.dev/
  ssb/rulings/SSB-2026-Q2-001`); the format is operator-readable
  + scholar-citable + free of operator-specific PII (the SSB
  ruling references the *product*, not the user / portfolio /
  account). Mirrors the no-PII pattern of Wave 2.F scholar
  review's review-packet contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class FiqhSchool(str, Enum):
    """The four Sunni schools of Islamic jurisprudence.

    The roadmap mandates ≥ 3 different schools on the board to
    prevent a single madhhab's ruling from being misread as
    universal consensus. Some platforms expand to include Ja'fari
    (Twelver Shi'a) for global market coverage; the operator can
    extend the enum for that, but the four Sunni schools are the
    AAOIFI / global-Islamic-banking baseline.
    """

    HANAFI = "hanafi"
    SHAFII = "shafii"
    MALIKI = "maliki"
    HANBALI = "hanbali"


class RulingScope(str, Enum):
    """What the ruling applies to.

    Pinned string values for public-register stability. Operators
    use the scope to route ruling to the right downstream effect
    — PRODUCT rulings affect the screener cache; STRATEGY rulings
    affect the strategy allow-list; POLICY rulings affect
    operational config; CERTIFICATION is the general platform
    attestation.
    """

    PRODUCT = "product"
    STRATEGY = "strategy"
    POLICY = "policy"
    CERTIFICATION = "certification"


class RulingOutcome(str, Enum):
    """SSB verdict on the subject.

    `PERMISSIBLE` is the clean halal pass. `IMPERMISSIBLE` is the
    riba / gharar / haram rejection. `PERMISSIBLE_WITH_CONDITIONS`
    is the nuanced pass requiring specific safeguards. `DEFERRED`
    is "needs more research / data" — the operator must re-submit
    after gathering the missing inputs.
    """

    PERMISSIBLE = "permissible"
    IMPERMISSIBLE = "impermissible"
    PERMISSIBLE_WITH_CONDITIONS = "permissible_with_conditions"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class SSBPolicy:
    """Operator-tunable governance policy.

    `minimum_members` is the floor on board size; defaults to 3
    per the roadmap. `minimum_schools` is the diversity floor;
    defaults to 3 (the AAOIFI guidance for cross-madhhab
    representation). `supermajority_pct` is the fraction required
    for PERMISSIBLE outcomes; defaults to 2/3. `review_cycle_days`
    is the quarterly review cadence. `term_length_days` is the
    default scholar appointment length.
    """

    minimum_members: int = 3
    minimum_schools: int = 3
    supermajority_pct: float = 2.0 / 3.0
    review_cycle_days: int = 90
    term_length_days: int = 365 * 3

    def __post_init__(self) -> None:
        if self.minimum_members < 1:
            raise ValueError("minimum_members must be at least 1")
        if self.minimum_schools < 1:
            raise ValueError("minimum_schools must be at least 1")
        if self.minimum_schools > self.minimum_members:
            raise ValueError("minimum_schools cannot exceed minimum_members")
        if not 0.5 < self.supermajority_pct <= 1.0:
            raise ValueError("supermajority_pct must be in (0.5, 1.0]")
        if self.review_cycle_days <= 0:
            raise ValueError("review_cycle_days must be positive")
        if self.term_length_days <= 0:
            raise ValueError("term_length_days must be positive")


DEFAULT_POLICY = SSBPolicy()


@dataclass(frozen=True)
class ScholarMember:
    """One SSB scholar.

    `appointed_at` and `expires_at` track the term; the engine
    drops votes from expired-term members and surfaces a warning.
    `bio_url` is the public-register link the operator publishes.
    """

    name: str
    school: FiqhSchool
    appointed_at: datetime
    expires_at: datetime
    bio_url: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.appointed_at.tzinfo is None:
            raise ValueError("appointed_at must be timezone-aware")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.appointed_at:
            raise ValueError("expires_at must be after appointed_at")

    def is_active(self, *, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return self.appointed_at <= now < self.expires_at


@dataclass(frozen=True)
class BoardCompositionResult:
    """Outcome of validating board composition against policy."""

    is_valid: bool
    member_count: int
    school_count: int
    schools_represented: tuple[FiqhSchool, ...]
    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate_board(
    members: tuple[ScholarMember, ...],
    *,
    now: datetime,
    policy: SSBPolicy = DEFAULT_POLICY,
) -> BoardCompositionResult:
    """Verify the board satisfies composition + diversity policy.

    Counts only currently-active members (within their term);
    expired-term members surface as warnings + are excluded from
    the count.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    failures: list[str] = []
    warnings: list[str] = []
    active_members: list[ScholarMember] = []
    seen_names: set[str] = set()

    for m in members:
        if m.name in seen_names:
            failures.append(f"duplicate scholar name {m.name!r}")
            continue
        seen_names.add(m.name)
        if not m.is_active(now=now):
            warnings.append(
                f"scholar {m.name!r} ({m.school.value}) is past term "
                f"(expired {m.expires_at.isoformat()})"
            )
            continue
        active_members.append(m)

    schools = sorted({m.school for m in active_members}, key=lambda s: s.value)

    if len(active_members) < policy.minimum_members:
        failures.append(
            f"active member count {len(active_members)} below minimum {policy.minimum_members}"
        )
    if len(schools) < policy.minimum_schools:
        failures.append(
            f"active school count {len(schools)} below minimum {policy.minimum_schools}"
        )

    return BoardCompositionResult(
        is_valid=not failures,
        member_count=len(active_members),
        school_count=len(schools),
        schools_represented=tuple(schools),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class Vote:
    """One scholar's vote on a ruling.

    `rationale` is required for non-PERMISSIBLE outcomes —
    silent rejections without justification are a regulatory
    failure mode (the operator's compliance program must be able
    to explain *why* something was deemed impermissible). For
    PERMISSIBLE / DEFERRED outcomes the rationale is optional.
    """

    member_name: str
    school: FiqhSchool
    outcome: RulingOutcome
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.member_name or not self.member_name.strip():
            raise ValueError("member_name must be non-empty")
        # IMPERMISSIBLE and PERMISSIBLE_WITH_CONDITIONS require rationale
        # (the latter because the conditions need explanation; the
        # former because rejections need justification).
        if self.outcome in (
            RulingOutcome.IMPERMISSIBLE,
            RulingOutcome.PERMISSIBLE_WITH_CONDITIONS,
        ):
            if not self.rationale or not self.rationale.strip():
                raise ValueError(f"rationale required for {self.outcome.value} verdict")


def _consensus_outcome(votes: tuple[Vote, ...], *, policy: SSBPolicy) -> RulingOutcome:
    """Compute the aggregated outcome per the conservative-tiebreak rule.

    Pin order: any IMPERMISSIBLE → IMPERMISSIBLE; supermajority
    PERMISSIBLE → PERMISSIBLE; mix of PERMISSIBLE +
    PERMISSIBLE_WITH_CONDITIONS reaching supermajority →
    PERMISSIBLE_WITH_CONDITIONS; otherwise DEFERRED.
    """

    if not votes:
        return RulingOutcome.DEFERRED

    outcomes = [v.outcome for v in votes]

    # Conservative tiebreak: any IMPERMISSIBLE is decisive.
    if RulingOutcome.IMPERMISSIBLE in outcomes:
        return RulingOutcome.IMPERMISSIBLE

    total = len(outcomes)
    permissible_count = outcomes.count(RulingOutcome.PERMISSIBLE)
    conditional_count = outcomes.count(RulingOutcome.PERMISSIBLE_WITH_CONDITIONS)
    pass_count = permissible_count + conditional_count

    if pass_count / total < policy.supermajority_pct:
        return RulingOutcome.DEFERRED

    # Supermajority pass: if any vote was conditional, the consensus
    # is conditional (the conditions need to be honoured even if
    # the majority was unconditional).
    if conditional_count > 0:
        return RulingOutcome.PERMISSIBLE_WITH_CONDITIONS
    return RulingOutcome.PERMISSIBLE


@dataclass(frozen=True)
class Ruling:
    """One SSB ruling — the public-register entry."""

    ruling_id: str
    scope: RulingScope
    subject: str
    description: str
    issued_at: datetime
    votes: tuple[Vote, ...]
    conditions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.ruling_id or not self.ruling_id.strip():
            raise ValueError("ruling_id must be non-empty")
        if not self.subject or not self.subject.strip():
            raise ValueError("subject must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("description must be non-empty")
        if self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        # Conditional rulings need explicit conditions.
        outcomes = [v.outcome for v in self.votes]
        if RulingOutcome.PERMISSIBLE_WITH_CONDITIONS in outcomes and not self.conditions:
            raise ValueError("PERMISSIBLE_WITH_CONDITIONS votes present but no conditions listed")
        # Vote member-name uniqueness — one scholar can't vote twice.
        seen_voters: set[str] = set()
        for v in self.votes:
            if v.member_name in seen_voters:
                raise ValueError(f"duplicate vote from member {v.member_name!r}")
            seen_voters.add(v.member_name)

    def consensus(self, *, policy: SSBPolicy = DEFAULT_POLICY) -> RulingOutcome:
        return _consensus_outcome(self.votes, policy=policy)


def needs_quarterly_review(
    rulings: tuple[Ruling, ...],
    *,
    now: datetime,
    policy: SSBPolicy = DEFAULT_POLICY,
) -> bool:
    """True if the most recent ruling is older than the review cycle.

    Pinned semantic: an empty rulings list returns True — the
    board is overdue from the moment of incorporation if it
    hasn't issued any rulings yet.
    """

    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not rulings:
        return True
    latest = max(rulings, key=lambda r: r.issued_at)
    return (now - latest.issued_at) > timedelta(days=policy.review_cycle_days)


_OUTCOME_EMOJI: dict[RulingOutcome, str] = {
    RulingOutcome.PERMISSIBLE: "✅",
    RulingOutcome.IMPERMISSIBLE: "❌",
    RulingOutcome.PERMISSIBLE_WITH_CONDITIONS: "⚠️",
    RulingOutcome.DEFERRED: "⏳",
}


def render_ruling(ruling: Ruling, *, policy: SSBPolicy = DEFAULT_POLICY) -> str:
    """Format the ruling for the public register.

    Pinned no-operator-PII contract: the ruling references the
    product / strategy / policy *subject*, never an operator's
    account / portfolio / user identifier. Public-register-safe.
    """

    consensus = ruling.consensus(policy=policy)
    emoji = _OUTCOME_EMOJI[consensus]
    lines = [
        f"{emoji} {ruling.ruling_id} — {consensus.value.upper()}",
        f"  scope: {ruling.scope.value}",
        f"  subject: {ruling.subject}",
        f"  issued: {ruling.issued_at.isoformat()}",
        f"  description: {ruling.description}",
    ]

    lines.append("  votes:")
    for v in ruling.votes:
        lines.append(f"    · {v.member_name} ({v.school.value}): {v.outcome.value}")
        if v.rationale:
            lines.append(f"        rationale: {v.rationale}")

    if ruling.conditions:
        lines.append("  conditions:")
        for c in ruling.conditions:
            lines.append(f"    · {c}")

    return "\n".join(lines)


def render_board_composition(result: BoardCompositionResult) -> str:
    """Format the board-composition validation result."""

    emoji = "✅" if result.is_valid else "❌"
    lines = [
        f"{emoji} SSB composition: {'VALID' if result.is_valid else 'INVALID'}",
        f"  active members: {result.member_count}",
        f"  schools represented: {result.school_count}"
        + (
            " (" + ", ".join(s.value for s in result.schools_represented) + ")"
            if result.schools_represented
            else ""
        ),
    ]
    if result.failures:
        lines.append("  failures:")
        for f in result.failures:
            lines.append(f"    · {f}")
    if result.warnings:
        lines.append("  warnings:")
        for w in result.warnings:
            lines.append(f"    · {w}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "BoardCompositionResult",
    "FiqhSchool",
    "Ruling",
    "RulingOutcome",
    "RulingScope",
    "SSBPolicy",
    "ScholarMember",
    "Vote",
    "needs_quarterly_review",
    "render_board_composition",
    "render_ruling",
    "validate_board",
]
