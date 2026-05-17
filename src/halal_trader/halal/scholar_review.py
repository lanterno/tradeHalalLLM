"""Scholar-facing review-packet renderer + verdict recorder.

Round-4 wave 2.F: when the bot encounters a symbol with
insufficient screening data (an IPO Zoya hasn't classified yet,
a token below CoinGecko's coverage threshold, an edge case
between scholar profiles), the existing
`halal/exception_queue.py` queues a `pending` `ExceptionEntry`.
This module is the **scholar-facing layer** on top: it renders
each pending entry into a review packet (markdown brief +
operator-supplied context) and records the scholar's verdict
back through a structured `ScholarVerdict` object that the
queue can apply.

Two responsibilities:

* **`render_review_packet(entry, context)`** — produces a
  markdown review brief the scholar reads. Includes the
  symbol, the kind of decision (`equity` / `crypto` /
  `sukuk` / `commodity`), the reasoning the bot
  recorded, and any operator-supplied context (recent price
  data, SEC filing snippets, sector classification).
  Pin: never includes operator-identifying data
  (account IDs, position sizes); the scholar reviews the
  symbol on its merits.

* **`record_verdict(verdict, queue)`** — applies a
  `ScholarVerdict` (approve / reject / defer with optional
  note) to the queue. Validates that the entry exists and
  is `pending` before applying — pin: a verdict on a
  non-pending entry surfaces as a clear error rather than
  silently overwriting an earlier decision.

Why a separate module from `exception_queue.py`:

* The queue is the storage / state machine; this module is
  *presentation + workflow*. Keeping them apart means a
  future SQL refactor of the queue doesn't ripple into the
  packet shape.
* The packet renderer is operator-extensible — operators
  can add custom context types (price snapshot, filing
  text, peer-comparable list) without changing the queue
  contract.

Halal alignment: the workflow is **scholar-driven**. The bot
never auto-approves an exception; the scholar must explicitly
record a verdict via the packet → email → response → record
chain. The audit trail captures who decided + when + the
optional operator note that prompted the review.

Pure-Python; no DB / network. Email pipeline (rendering the
packet to HTML + sending via SMTP / SendGrid) is the caller's
job — this module produces the markdown body the email
template wraps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

# ── Verdict vocabulary ───────────────────────────────────


class VerdictKind(str, Enum):
    """The four outcomes of a scholar review.

    Pin: matches the existing `ExceptionStatus` literal in
    `exception_queue.py` so the verdict can be applied directly
    to the queue without translation.

    * ``APPROVED`` — the symbol is permissible under this
      operator's profile; the screener cache may treat as halal.
    * ``REJECTED`` — the symbol is not permissible; cache as
      not_halal.
    * ``DEFERRED`` — the scholar wants more time / data; the
      entry stays pending.
    * ``WITHDRAWN`` — the operator no longer needs the review
      (e.g., the symbol was de-listed before the scholar
      responded).
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    WITHDRAWN = "withdrawn"


# ── Packet input ─────────────────────────────────────────


@dataclass(frozen=True)
class ReviewContext:
    """Operator-supplied context that helps the scholar decide.

    All fields optional — the packet renders only the sections
    the operator filled in. Pin: NO fields here are
    operator-identifying or position-level (no account ID, no
    notional). The scholar evaluates the symbol on its merits;
    operator context appears only as the *reason this question
    came up* (e.g., "newly listed in IPO 2026-04-15").
    """

    sector: str = ""
    market_cap_usd: float | None = None
    recent_revenue_breakdown: str = ""
    debt_to_marketcap_pct: float | None = None
    non_permissible_income_pct: float | None = None
    notes: str = ""
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewPacket:
    """Rendered review brief ready for the email pipeline.

    ``subject`` is the email subject line. ``markdown`` is the
    body — the email template wraps in HTML."""

    entry_id: str
    instrument: str
    kind: str
    subject: str
    markdown: str


def _format_pct(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def _format_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def render_review_packet(
    *,
    entry_id: str,
    instrument: str,
    kind: str,
    reasoning: str,
    context: ReviewContext | None = None,
    profile_name: str = "aaoifi_default",
) -> ReviewPacket:
    """Render a scholar-readable markdown brief.

    Pin: the brief never contains operator-identifying data.
    Profile name is included so the scholar knows which
    threshold set the operator's bot is currently using; it's
    metadata, not PII.
    """
    ctx = context or ReviewContext()
    subject = f"[Halal Review] {instrument} ({kind}) — pending verdict"

    lines = [
        f"# Scholar review request: {instrument}",
        "",
        f"**Kind:** {kind}",
        f"**Profile in use:** `{profile_name}`",
        f"**Entry ID:** `{entry_id}`",
        "",
        "## Why this review is needed",
        "",
        reasoning if reasoning else "_(no reasoning recorded)_",
    ]

    if ctx.sector or ctx.market_cap_usd is not None:
        lines.extend(
            [
                "",
                "## Symbol context",
                "",
            ]
        )
        if ctx.sector:
            lines.append(f"- **Sector:** {ctx.sector}")
        if ctx.market_cap_usd is not None:
            lines.append(f"- **Market cap:** {_format_usd(ctx.market_cap_usd)}")

    has_financials = (
        ctx.debt_to_marketcap_pct is not None
        or ctx.non_permissible_income_pct is not None
        or ctx.recent_revenue_breakdown
    )
    if has_financials:
        lines.extend(
            [
                "",
                "## Financial screen inputs",
                "",
            ]
        )
        if ctx.debt_to_marketcap_pct is not None:
            lines.append(f"- **Debt / market cap:** {_format_pct(ctx.debt_to_marketcap_pct)}")
        if ctx.non_permissible_income_pct is not None:
            lines.append(
                f"- **Non-permissible income / revenue:** "
                f"{_format_pct(ctx.non_permissible_income_pct)}"
            )
        if ctx.recent_revenue_breakdown:
            lines.append("")
            lines.append(f"> {ctx.recent_revenue_breakdown}")

    if ctx.notes:
        lines.extend(["", "## Notes", "", ctx.notes])

    if ctx.references:
        lines.extend(["", "## References", ""])
        for ref in ctx.references:
            lines.append(f"- {ref}")

    lines.extend(
        [
            "",
            "## How to respond",
            "",
            "Reply to this email with one of:",
            "",
            "- **APPROVED** — the symbol is permissible for this profile.",
            "- **REJECTED** — the symbol is not permissible.",
            "- **DEFERRED** — request more time / data; the entry stays pending.",
            "",
            "Add a one-line rationale on the next line of your reply. The "
            "operator's bot records your verdict + rationale in the audit "
            "trail; future reviewers see what you decided and why.",
        ]
    )

    return ReviewPacket(
        entry_id=entry_id,
        instrument=instrument,
        kind=kind,
        subject=subject,
        markdown="\n".join(lines),
    )


# ── Verdict + recorder ───────────────────────────────────


@dataclass(frozen=True)
class ScholarVerdict:
    """One scholar's response to a review packet.

    ``rationale`` is the scholar's one-line note (mandatory for
    ``REJECTED`` and ``APPROVED``; optional for ``DEFERRED`` /
    ``WITHDRAWN``). The operator's audit trail records this
    verbatim.

    ``decided_by`` is the scholar's email or identifier — pin:
    must be non-empty so the audit trail can attribute the
    verdict.
    """

    entry_id: str
    kind: VerdictKind
    decided_by: str
    rationale: str = ""
    decided_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("entry_id must be non-empty")
        if not self.decided_by:
            raise ValueError(
                "decided_by must be non-empty (audit trail needs the scholar's identifier)"
            )
        if self.kind in (VerdictKind.APPROVED, VerdictKind.REJECTED):
            if not self.rationale.strip():
                raise ValueError(
                    f"{self.kind.value} verdict requires a non-empty "
                    "rationale (scholar must justify the call)"
                )


@dataclass(frozen=True)
class RecordedVerdict:
    """The audit-trail row a verdict produces."""

    entry_id: str
    instrument: str
    kind: VerdictKind
    decided_by: str
    rationale: str
    decided_at: datetime
    previous_status: str  # what the entry's status was before


def _verdict_to_status(kind: VerdictKind) -> str:
    """Map verdict → exception-queue status string. Matches the
    `ExceptionStatus` literal used in `exception_queue.py`."""
    if kind == VerdictKind.APPROVED:
        return "approved"
    if kind == VerdictKind.REJECTED:
        return "rejected"
    if kind == VerdictKind.DEFERRED:
        return "deferred"
    if kind == VerdictKind.WITHDRAWN:
        return "withdrawn"
    raise ValueError(f"unknown verdict kind {kind!r}")


def apply_verdict(
    verdict: ScholarVerdict,
    *,
    pending_entry_status: str,
    pending_entry_instrument: str,
    now: datetime | None = None,
) -> RecordedVerdict:
    """Validate the verdict against the entry's current status
    and produce the audit-trail row.

    Pin: a verdict on a non-pending entry raises rather than
    silently overwriting. Operators may end up with a stale
    review email if a scholar replied late; the bot must
    surface "this entry was already decided" rather than
    flip-flopping the cache.

    The actual queue mutation (write the new status to
    Postgres) is the caller's job — this function returns the
    `RecordedVerdict` the caller persists alongside the queue
    row update.
    """
    if pending_entry_status != "pending":
        raise ValueError(
            f"cannot apply verdict to entry {verdict.entry_id!r}: "
            f"current status is {pending_entry_status!r}, not 'pending'"
        )
    decided_at = verdict.decided_at or (now or datetime.now(UTC))
    return RecordedVerdict(
        entry_id=verdict.entry_id,
        instrument=pending_entry_instrument,
        kind=verdict.kind,
        decided_by=verdict.decided_by,
        rationale=verdict.rationale,
        decided_at=decided_at,
        previous_status=pending_entry_status,
    )


# ── Render helpers ───────────────────────────────────────


def render_recorded_verdict(verdict: RecordedVerdict) -> str:
    """Operator-readable summary of a recorded verdict for log /
    Slack / audit-trail rendering."""
    emoji = {
        VerdictKind.APPROVED: "✅",
        VerdictKind.REJECTED: "❌",
        VerdictKind.DEFERRED: "⏳",
        VerdictKind.WITHDRAWN: "↩️",
    }[verdict.kind]
    line = (
        f"{emoji} {verdict.instrument} ({verdict.entry_id}) "
        f"{verdict.kind.value} by {verdict.decided_by} "
        f"at {verdict.decided_at:%Y-%m-%d %H:%M UTC}"
    )
    if verdict.rationale:
        line += f"\n  → {verdict.rationale}"
    return line


__all__ = [
    "RecordedVerdict",
    "ReviewContext",
    "ReviewPacket",
    "ScholarVerdict",
    "VerdictKind",
    "apply_verdict",
    "render_recorded_verdict",
    "render_review_packet",
]
