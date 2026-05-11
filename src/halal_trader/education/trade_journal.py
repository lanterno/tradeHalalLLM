"""Trade journal + coach scoring — Round-5 Wave 20.D.

Per-trade structured journal entries. Each entry captures the
operator's *process* (rationale, stop placement, sizing, emotional
state) — not the outcome. A rule-based coach surfaces process-quality
flags (no stop set, vague rationale, sizing breach, missing thesis
tags, emotional-trading risk).

This is a **journal record + coach** module; the LLM-driven richer
coach can plug in later but operators get value from the deterministic
rule set first.

Pinned semantics:

- **Closed-set EntryStatus** — DRAFT / FINALISED / SUPERSEDED.
- **Closed-set CoachFlagKind**: NO_STOP / VAGUE_RATIONALE /
  SIZING_BREACH / NO_THESIS_TAG / EMOTIONAL_RISK / NO_TARGET /
  R_R_TOO_LOW / ENTRY_AT_EXTREME.
- **Closed-set Severity** — NOTE / WARN / BLOCK.
- **Vague rationale** = < 30 chars OR contains "feel"/"gut"/"YOLO" without
  any specific market language.
- **Sizing breach** = position notional > `max_position_pct` of equity
  (default 5%).
- **R:R too low** = reward/risk < `min_r_r` (default 1.5).
- **Pure-Python deterministic.**
- **No-secret-leak pin** on render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


class EntryStatus(str, Enum):
    """Closed-set journal entry status."""

    DRAFT = "draft"
    FINALISED = "finalised"
    SUPERSEDED = "superseded"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class EmotionalState(str, Enum):
    """Closed-set self-reported state."""

    CALM = "calm"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    REVENGE = "revenge"
    FOMO = "fomo"
    CONFIDENT = "confident"


class CoachFlagKind(str, Enum):
    """Closed-set coach flag taxonomy."""

    NO_STOP = "no_stop"
    NO_TARGET = "no_target"
    VAGUE_RATIONALE = "vague_rationale"
    SIZING_BREACH = "sizing_breach"
    NO_THESIS_TAG = "no_thesis_tag"
    EMOTIONAL_RISK = "emotional_risk"
    R_R_TOO_LOW = "r_r_too_low"
    ENTRY_AT_EXTREME = "entry_at_extreme"


class Severity(str, Enum):
    """Closed-set severity ladder."""

    NOTE = "note"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class JournalEntry:
    """One trade journal entry."""

    entry_id: str
    trade_id: str
    author_id: str
    ticker: str
    side: Side
    entry_price: float
    quantity: float
    account_equity_at_entry: float
    rationale: str
    thesis_tags: tuple[str, ...] = ()
    stop_price: float | None = None
    target_price: float | None = None
    emotional_state: EmotionalState = EmotionalState.CALM
    rsi_at_entry: float | None = None
    """Optional RSI for the "entry at extreme" coach flag."""
    created_at: datetime | None = None
    status: EntryStatus = EntryStatus.DRAFT

    def __post_init__(self) -> None:
        if not self.entry_id or not self.entry_id.strip():
            raise ValueError("entry_id must be non-empty")
        if not self.trade_id or not self.trade_id.strip():
            raise ValueError("trade_id must be non-empty")
        if not self.author_id or not self.author_id.strip():
            raise ValueError("author_id must be non-empty")
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.account_equity_at_entry <= 0:
            raise ValueError("account_equity_at_entry must be positive")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when set")
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError("target_price must be positive when set")
        if self.rsi_at_entry is not None and not 0.0 <= self.rsi_at_entry <= 100.0:
            raise ValueError("rsi_at_entry must be in [0, 100]")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")
        if len(self.rationale) > 2000:
            raise ValueError("rationale must be ≤ 2000 chars")
        # Stop / target geometry by side.
        if self.side is Side.LONG:
            if self.stop_price is not None and self.stop_price >= self.entry_price:
                raise ValueError("LONG stop must be < entry_price")
            if self.target_price is not None and self.target_price <= self.entry_price:
                raise ValueError("LONG target must be > entry_price")
        else:
            if self.stop_price is not None and self.stop_price <= self.entry_price:
                raise ValueError("SHORT stop must be > entry_price")
            if self.target_price is not None and self.target_price >= self.entry_price:
                raise ValueError("SHORT target must be < entry_price")

    def position_notional(self) -> float:
        return self.entry_price * self.quantity

    def position_pct(self) -> float:
        return self.position_notional() / self.account_equity_at_entry

    def reward_to_risk(self) -> float | None:
        """Computes R:R when both stop and target are set."""
        if self.stop_price is None or self.target_price is None:
            return None
        if self.side is Side.LONG:
            risk = self.entry_price - self.stop_price
            reward = self.target_price - self.entry_price
        else:
            risk = self.stop_price - self.entry_price
            reward = self.entry_price - self.target_price
        if risk <= 0:
            return None
        return reward / risk


@dataclass(frozen=True)
class CoachFlag:
    """One issue flagged by the coach."""

    kind: CoachFlagKind
    severity: Severity
    message: str


@dataclass(frozen=True)
class CoachReport:
    """Output of `coach`."""

    entry_id: str
    flags: tuple[CoachFlag, ...]
    process_score: float
    """0–1. Higher is better; flat 1.0 - 0.10 × n_warn - 0.20 × n_block."""

    def has_block(self) -> bool:
        return any(f.severity is Severity.BLOCK for f in self.flags)

    def by_severity(self, sev: Severity) -> tuple[CoachFlag, ...]:
        return tuple(f for f in self.flags if f.severity is sev)


# Words that signal "vague rationale" — paired with no concrete market
# language.
_VAGUE_RE = re.compile(r"\b(feel|gut|YOLO|hunch|vibes?|seems good)\b", re.IGNORECASE)

# Concrete-market language whose presence indicates the operator at
# least gestured at a process. Operator-tunable.
_CONCRETE_RE = re.compile(
    r"\b(earnings|guidance|breakout|support|resistance|rsi|macd|"
    r"volume|sukuk|sector|catalyst|stop|target|halal|fundamental|"
    r"thesis|setup|signal|pivot|moving average|ma\b|ema\b)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CoachPolicy:
    """Tunable coach policy."""

    max_position_pct: float = 0.05
    """Sizing breach trigger."""
    min_rationale_chars: int = 30
    min_r_r: float = 1.5
    """R:R below this is flagged when both stop + target are set."""
    extreme_rsi_low: float = 25.0
    extreme_rsi_high: float = 75.0
    emotional_block_states: tuple[EmotionalState, ...] = (
        EmotionalState.REVENGE,
        EmotionalState.FOMO,
    )
    emotional_warn_states: tuple[EmotionalState, ...] = (
        EmotionalState.FRUSTRATED,
        EmotionalState.EXCITED,
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.max_position_pct <= 1.0:
            raise ValueError("max_position_pct must be in (0, 1]")
        if self.min_rationale_chars <= 0:
            raise ValueError("min_rationale_chars must be positive")
        if self.min_r_r <= 0:
            raise ValueError("min_r_r must be positive")
        if not 0.0 <= self.extreme_rsi_low < self.extreme_rsi_high <= 100.0:
            raise ValueError("0 ≤ extreme_rsi_low < extreme_rsi_high ≤ 100 required")


def coach(
    entry: JournalEntry,
    *,
    policy: CoachPolicy | None = None,
) -> CoachReport:
    """Run the rule-based coach. Returns a frozen `CoachReport`."""
    pol = policy if policy is not None else CoachPolicy()
    flags: list[CoachFlag] = []

    if entry.stop_price is None:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.NO_STOP,
                severity=Severity.BLOCK,
                message="no stop price set — every entry needs a stop",
            )
        )
    if entry.target_price is None:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.NO_TARGET,
                severity=Severity.WARN,
                message="no target price set — process is harder to audit",
            )
        )
    if len(entry.rationale) < pol.min_rationale_chars or (
        _VAGUE_RE.search(entry.rationale) and not _CONCRETE_RE.search(entry.rationale)
    ):
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.VAGUE_RATIONALE,
                severity=Severity.WARN,
                message=(
                    f"rationale is vague (< {pol.min_rationale_chars} chars or no "
                    "concrete market language)"
                ),
            )
        )
    if entry.position_pct() > pol.max_position_pct + 1e-12:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.SIZING_BREACH,
                severity=Severity.BLOCK,
                message=(
                    f"position {entry.position_pct() * 100:.2f}% > cap "
                    f"{pol.max_position_pct * 100:.2f}%"
                ),
            )
        )
    if not entry.thesis_tags:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.NO_THESIS_TAG,
                severity=Severity.NOTE,
                message="no thesis tags set — attribution will be coarse",
            )
        )
    if entry.emotional_state in pol.emotional_block_states:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.EMOTIONAL_RISK,
                severity=Severity.BLOCK,
                message=(
                    f"emotional state {entry.emotional_state.value} — "
                    "step away before placing this trade"
                ),
            )
        )
    elif entry.emotional_state in pol.emotional_warn_states:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.EMOTIONAL_RISK,
                severity=Severity.WARN,
                message=(f"emotional state {entry.emotional_state.value} — process risk elevated"),
            )
        )
    rr = entry.reward_to_risk()
    if rr is not None and rr < pol.min_r_r:
        flags.append(
            CoachFlag(
                kind=CoachFlagKind.R_R_TOO_LOW,
                severity=Severity.WARN,
                message=(
                    f"R:R {rr:.2f} < {pol.min_r_r:.2f} minimum — expected value is unfavourable"
                ),
            )
        )
    if entry.rsi_at_entry is not None:
        if entry.side is Side.LONG and entry.rsi_at_entry >= pol.extreme_rsi_high:
            flags.append(
                CoachFlag(
                    kind=CoachFlagKind.ENTRY_AT_EXTREME,
                    severity=Severity.WARN,
                    message=(
                        f"LONG with RSI={entry.rsi_at_entry:.1f} "
                        f"≥ {pol.extreme_rsi_high:.0f} — overbought entry"
                    ),
                )
            )
        elif entry.side is Side.SHORT and entry.rsi_at_entry <= pol.extreme_rsi_low:
            flags.append(
                CoachFlag(
                    kind=CoachFlagKind.ENTRY_AT_EXTREME,
                    severity=Severity.WARN,
                    message=(
                        f"SHORT with RSI={entry.rsi_at_entry:.1f} "
                        f"≤ {pol.extreme_rsi_low:.0f} — oversold entry"
                    ),
                )
            )
    n_warn = sum(1 for f in flags if f.severity is Severity.WARN)
    n_block = sum(1 for f in flags if f.severity is Severity.BLOCK)
    score = max(0.0, 1.0 - 0.10 * n_warn - 0.20 * n_block)
    return CoachReport(
        entry_id=entry.entry_id,
        flags=tuple(flags),
        process_score=score,
    )


def finalise(entry: JournalEntry, *, at: datetime) -> JournalEntry:
    """Promote a DRAFT entry to FINALISED. SUPERSEDED is terminal."""
    if entry.status is EntryStatus.FINALISED:
        return entry
    if entry.status is EntryStatus.SUPERSEDED:
        raise ValueError("cannot finalise a SUPERSEDED entry")
    return replace(entry, status=EntryStatus.FINALISED, created_at=at)


def supersede(entry: JournalEntry) -> JournalEntry:
    if entry.status is EntryStatus.SUPERSEDED:
        raise ValueError("entry already SUPERSEDED")
    return replace(entry, status=EntryStatus.SUPERSEDED)


def _mask(party_id: str) -> str:
    if len(party_id) <= 4:
        return "***"
    return party_id[:2] + "…" + party_id[-2:]


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.NOTE: "ℹ️",
    Severity.WARN: "⚠️",
    Severity.BLOCK: "🛑",
}


def render_report(report: CoachReport) -> str:
    if not report.flags:
        return f"✅ {report.entry_id}: clean entry (score={report.process_score:.2f})"
    head = f"🧭 {report.entry_id}: {len(report.flags)} flag(s), score={report.process_score:.2f}"
    lines = [head]
    for f in report.flags:
        lines.append(f"  {_SEVERITY_EMOJI[f.severity]} [{f.kind.value}] {f.message}")
    return "\n".join(lines)


def render_entry(entry: JournalEntry) -> str:
    stop_str = f"{entry.stop_price:.2f}" if entry.stop_price is not None else "—"
    target_str = f"{entry.target_price:.2f}" if entry.target_price is not None else "—"
    rr = entry.reward_to_risk()
    rr_str = f"R:R={rr:.2f}" if rr is not None else "R:R=—"
    return (
        f"📓 {entry.entry_id} [{entry.status.value}] "
        f"{entry.side.value} {entry.ticker} qty={entry.quantity:.2f} "
        f"entry={entry.entry_price:.2f} stop={stop_str} target={target_str} "
        f"{rr_str}\n  Author {_mask(entry.author_id)} "
        f"size={entry.position_pct() * 100:.2f}% "
        f"mood={entry.emotional_state.value}"
    )
