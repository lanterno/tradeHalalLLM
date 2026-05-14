"""AI co-pilot intent classifier + safe-action gate.

The roadmap envisions a conversational dashboard: the user types
"what's my best-performing strategy this quarter?", "explain why
you sold AAPL yesterday", "set up a stop on my BTC position at
-3%", "halt trading". The platform becomes a chat interface
backed by the existing engine.

This module is the **safety gate** that sits between the chat
layer (LLM router that interprets the message) and the
underlying actions. It does NOT:
- Execute trades from natural language (the strategy does that)
- Move funds anywhere outside operator-controlled flows
- Delete user accounts via chat

It DOES:
- Classify the intent (read-only query / state-mutating action /
  forbidden)
- Reject natural-language wire-funds / delete-everything /
  execute-real-money-trade requests categorically — even if a
  jailbroken LLM tries to authorise them, the classifier
  refuses
- Require explicit confirmation for state-mutating actions
  (KILL_SWITCH, SET_STOP_LOSS) before the dashboard executes
- Surface the matched dangerous phrase in the rejection reason
  so the user understands why their request was blocked

Design choice: keyword-based classifier rather than LLM-based
intent extraction. Picked rule-based because (a) the safety gate
needs to be deterministic — an LLM mis-classifying "transfer all
funds to attacker-wallet" as a benign query is the worst-case
failure mode; (b) operators can read the classifier rule and
debug a misfire at the source rather than re-prompting; (c) the
existing LLM call path can still extract additional structure
(amounts, symbols) from the message *after* the gate has
classified it as safe.

Pinned semantics:
- **Closed dangerous-phrase set.** `_DANGEROUS_PHRASES` is a
  module-level frozenset; runtime mutation can't add a phrase
  via config. Operators extend via code review with a regression
  test.
- **DANGEROUS overrides everything.** A message matching any
  dangerous phrase returns DANGEROUS regardless of any other
  matched intent. The pin is regression-tested with a "transfer
  all funds AND show my balance" message that would otherwise
  match QUERY — the dangerous phrase wins.
- **State-mutating actions require confirmation.** The classifier
  flags `requires_confirmation=True` for KILL_SWITCH and
  SET_STOP_LOSS; the dashboard layer must surface a confirmation
  prompt before executing. The flag is data; the route enforces.
- **KYC gate for sensitive operations.** WITHDRAW / DEPOSIT
  intents (if the operator ever wires them up) require KYC
  VERIFIED; the gate sets `requires_kyc_verified=True` for those
  categories so the route layer composes with Wave 11.C.
- **Empty message returns UNKNOWN.** A blank prompt doesn't
  match any intent — operator's LLM layer should re-prompt the
  user. The engine never returns a hallucinated default intent.
- **Render output never echoes the user's full message.** The
  receipt summarises intent + matched-phrase + whether it was
  blocked, never the verbatim user input — guards against a
  poisoned prompt being rendered into the operator audit log
  and triggering downstream LLM re-execution. Mirrors the no-PII
  / no-secret-leak patterns of Wave 11.D + 11.C + 3.B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntentCategory(Enum):
    """The classified intent of a user message.

    `UNKNOWN` is the safe-default when no rule matches — the
    dashboard layer should re-prompt rather than guess.
    `DANGEROUS` is the categorical-block bucket: any matched
    dangerous phrase routes here regardless of other matches.
    """

    UNKNOWN = "unknown"
    QUERY = "query"  # read-only general question
    PORTFOLIO_QUERY = "portfolio_query"  # read-only portfolio data
    EXPLAIN = "explain"  # why-did-you-do-X
    STATUS = "status"  # operational health check
    SET_STOP_LOSS = "set_stop_loss"  # state-mutating but bounded
    KILL_SWITCH = "kill_switch"  # halt / resume — operationally critical
    DANGEROUS = "dangerous"  # blocked phrase categorically
    OUT_OF_SCOPE = "out_of_scope"  # not something the bot does


class ConfirmationRequirement(str, Enum):
    """Whether the action requires explicit user confirmation.

    `NEVER` for read-only intents; `ALWAYS` for state-mutating;
    `REJECT` is forbidden — the action never executes via the
    co-pilot regardless of confirmation.
    """

    NEVER = "never"
    ALWAYS = "always"
    REJECT = "reject"


# Phrases that route to DANGEROUS regardless of other matches.
# The set is module-level frozen so a runtime config drift can't
# add a phrase that bypasses the safety gate.
_DANGEROUS_PHRASES: frozenset[str] = frozenset(
    {
        # Account / data mutations
        "delete account",
        "delete my account",
        "close my account",
        "delete everything",
        "wipe all data",
        "drop tables",
        "drop database",
        # Fund movement (the bot never moves funds via chat)
        "wire funds to",
        "transfer funds to",
        "send all funds",
        "send funds to",
        "withdraw to address",
        "withdraw all",
        "send my balance",
        # Direct trade execution (strategy does that, not chat)
        "execute order",
        "place market order",
        "buy now real money",
        "sell now real money",
        "real money trade",
        # Privilege escalation
        "give me admin",
        "make me admin",
        "disable kyc",
        "skip kyc",
        "bypass screen",
        "bypass halal",
        "ignore screen",
        "override screening",
        # Credential / secret access
        "show api key",
        "reveal api key",
        "show password",
        "show secret",
        "dump secrets",
    }
)


# Phrases that route to OUT_OF_SCOPE — not malicious, just things
# this bot doesn't do. Surfacing them clearly helps the user
# understand the boundary.
_OUT_OF_SCOPE_PHRASES: frozenset[str] = frozenset(
    {
        "buy crypto with credit card",
        "send a wire",
        "open a bank account",
        "give tax advice",
        "give legal advice",
        "give medical advice",
        "predict the lottery",
        "predict price exactly",
    }
)


# Maps phrase keywords → intent. The first match wins (after
# DANGEROUS / OUT_OF_SCOPE are checked).
_INTENT_PHRASES: dict[IntentCategory, frozenset[str]] = {
    IntentCategory.STATUS: frozenset(
        {
            "is the bot running",
            "is the bot healthy",
            "system status",
            "halt status",
            "are you up",
            "are you running",
        }
    ),
    IntentCategory.PORTFOLIO_QUERY: frozenset(
        {
            "show my positions",
            "list my trades",
            "my open trades",
            "current positions",
            "show trades",
            "what trades",
            "my pnl",
            "show pnl",
            "best performing",
            "worst performing",
            "show performance",
        }
    ),
    IntentCategory.EXPLAIN: frozenset(
        {
            "explain why",
            "why did you",
            "why was",
            "rationale for",
            "reason for",
        }
    ),
    IntentCategory.SET_STOP_LOSS: frozenset(
        {
            "set stop loss",
            "set a stop",
            "stop loss at",
            "stop at",
            "set sl",
            "set trailing stop",
        }
    ),
    IntentCategory.KILL_SWITCH: frozenset(
        {
            "halt trading",
            "stop trading",
            "pause trading",
            "engage halt",
            "resume trading",
            "disengage halt",
            "kill switch",
        }
    ),
    IntentCategory.QUERY: frozenset(
        {
            "what is",
            "how does",
            "tell me about",
            "describe",
            "definition of",
        }
    ),
}


# Sensitive operations that require KYC VERIFIED before dispatch.
_KYC_REQUIRED_INTENTS: frozenset[IntentCategory] = frozenset(
    {
        IntentCategory.KILL_SWITCH,
        IntentCategory.SET_STOP_LOSS,
    }
)


_MIN_NON_BLANK_CHARS = 2


@dataclass(frozen=True)
class CopilotPolicy:
    """Operator-tunable copilot policy.

    `extra_dangerous_phrases` lets the operator add deployment-
    specific blocks (e.g., "transfer to wallet" for an exchange
    operator) without modifying the module's frozen set; the
    engine merges them at classify-time.
    """

    require_confirmation_for_kill_switch: bool = True
    require_confirmation_for_stop_loss: bool = True
    require_kyc_for_sensitive: bool = True
    extra_dangerous_phrases: frozenset[str] = field(default_factory=frozenset)
    max_message_length: int = 1000

    def __post_init__(self) -> None:
        if self.max_message_length <= 0:
            raise ValueError("max_message_length must be positive")
        for phrase in self.extra_dangerous_phrases:
            if not phrase or not phrase.strip():
                raise ValueError("extra_dangerous_phrases entries must be non-empty")


DEFAULT_POLICY = CopilotPolicy()


@dataclass(frozen=True)
class IntentClassification:
    """The classifier's verdict.

    `matched_phrase` carries the specific phrase that triggered
    the classification — the dashboard renders it in the rejection
    receipt so the user understands which keyword fired.
    """

    category: IntentCategory
    confidence: float
    matched_phrase: str
    requires_confirmation: ConfirmationRequirement
    requires_kyc_verified: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_blocked(self) -> bool:
        """True if the action should not execute via the co-pilot."""

        return (
            self.category is IntentCategory.DANGEROUS
            or self.requires_confirmation is ConfirmationRequirement.REJECT
        )


def _normalise(text: str) -> str:
    """Lower-case + strip whitespace + collapse internal whitespace."""

    return " ".join(text.lower().split())


def _matches_any(text: str, phrases: frozenset[str]) -> str | None:
    """Return the first matching phrase, or None."""

    for phrase in phrases:
        if phrase in text:
            return phrase
    return None


def classify_intent(
    message: str,
    *,
    policy: CopilotPolicy = DEFAULT_POLICY,
) -> IntentClassification:
    """Classify the user's natural-language message.

    Returns an `IntentClassification` with category + confidence
    + matched_phrase + confirmation requirement + KYC gate flag.
    The function is pure: deterministic for a given input.
    """

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if len(message) > policy.max_message_length:
        raise ValueError(f"message exceeds {policy.max_message_length} chars")

    normalised = _normalise(message)
    if len(normalised) < _MIN_NON_BLANK_CHARS:
        return IntentClassification(
            category=IntentCategory.UNKNOWN,
            confidence=0.0,
            matched_phrase="",
            requires_confirmation=ConfirmationRequirement.NEVER,
            requires_kyc_verified=False,
        )

    # 1. Dangerous-phrase check FIRST — overrides everything.
    all_dangerous = _DANGEROUS_PHRASES | policy.extra_dangerous_phrases
    matched = _matches_any(normalised, all_dangerous)
    if matched is not None:
        return IntentClassification(
            category=IntentCategory.DANGEROUS,
            confidence=1.0,
            matched_phrase=matched,
            requires_confirmation=ConfirmationRequirement.REJECT,
            requires_kyc_verified=False,
        )

    # 2. Out-of-scope check.
    matched = _matches_any(normalised, _OUT_OF_SCOPE_PHRASES)
    if matched is not None:
        return IntentClassification(
            category=IntentCategory.OUT_OF_SCOPE,
            confidence=1.0,
            matched_phrase=matched,
            requires_confirmation=ConfirmationRequirement.REJECT,
            requires_kyc_verified=False,
        )

    # 3. Category-specific phrase matches in priority order.
    # Order matters: more-specific intents (SET_STOP_LOSS,
    # KILL_SWITCH, EXPLAIN) checked before generic QUERY.
    for category in (
        IntentCategory.SET_STOP_LOSS,
        IntentCategory.KILL_SWITCH,
        IntentCategory.EXPLAIN,
        IntentCategory.STATUS,
        IntentCategory.PORTFOLIO_QUERY,
        IntentCategory.QUERY,
    ):
        matched = _matches_any(normalised, _INTENT_PHRASES[category])
        if matched is not None:
            confirmation = _confirmation_for(category, policy=policy)
            kyc_required = category in _KYC_REQUIRED_INTENTS and policy.require_kyc_for_sensitive
            return IntentClassification(
                category=category,
                confidence=0.9,  # rule-based confidence
                matched_phrase=matched,
                requires_confirmation=confirmation,
                requires_kyc_verified=kyc_required,
            )

    return IntentClassification(
        category=IntentCategory.UNKNOWN,
        confidence=0.0,
        matched_phrase="",
        requires_confirmation=ConfirmationRequirement.NEVER,
        requires_kyc_verified=False,
    )


def _confirmation_for(
    category: IntentCategory, *, policy: CopilotPolicy
) -> ConfirmationRequirement:
    if category is IntentCategory.KILL_SWITCH and policy.require_confirmation_for_kill_switch:
        return ConfirmationRequirement.ALWAYS
    if category is IntentCategory.SET_STOP_LOSS and policy.require_confirmation_for_stop_loss:
        return ConfirmationRequirement.ALWAYS
    return ConfirmationRequirement.NEVER


_CATEGORY_EMOJI: dict[IntentCategory, str] = {
    IntentCategory.UNKNOWN: "❓",
    IntentCategory.QUERY: "💬",
    IntentCategory.PORTFOLIO_QUERY: "📊",
    IntentCategory.EXPLAIN: "🔍",
    IntentCategory.STATUS: "🩺",
    IntentCategory.SET_STOP_LOSS: "🛑",
    IntentCategory.KILL_SWITCH: "⛔",
    IntentCategory.DANGEROUS: "🚫",
    IntentCategory.OUT_OF_SCOPE: "↩️",
}


def render_classification(
    classification: IntentClassification,
) -> str:
    """Render the classification for ops display.

    Pinned no-message-echo contract: the rendered receipt never
    contains the verbatim user input — only the matched phrase
    + classification verdict. Mirrors the no-secret-leak pattern
    of Wave 11.D / 11.C / 3.B / 8.D.
    """

    emoji = _CATEGORY_EMOJI[classification.category]
    lines = [
        f"{emoji} intent: {classification.category.value} "
        f"(confidence {classification.confidence:.2f})",
    ]
    if classification.matched_phrase:
        lines.append(f"  matched phrase: {classification.matched_phrase!r}")
    lines.append(f"  confirmation: {classification.requires_confirmation.value}")
    if classification.requires_kyc_verified:
        lines.append("  kyc: VERIFIED required")
    if classification.is_blocked:
        lines.append("  status: BLOCKED")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_POLICY",
    "ConfirmationRequirement",
    "CopilotPolicy",
    "IntentCategory",
    "IntentClassification",
    "classify_intent",
    "render_classification",
]
