"""Typed event payload schemas (REARCHITECTURE Appendix A).

Payloads travel as plain dicts on :class:`Event`; these TypedDicts document and
type the shape each ``EventType`` carries. Ingress events (observations,
compliance) are validated at the bus boundary: a payload missing required keys
is logged with its type and dropped, never dispatched (fail-closed — INV-4).
Internal events (belief/policy/order) are produced by trusted engine code and
are not key-validated here.
"""

from __future__ import annotations

from halabot.platform.events import Event, EventType

# Minimal required-key contracts for ingress event types. Unlisted types pass
# (they're produced by trusted internal code). Kept intentionally small —
# essential keys only, so the gate catches genuinely malformed feeds without
# being brittle to optional fields.
REQUIRED_KEYS: dict[EventType, set[str]] = {
    EventType.OBSERVATION_BAR: {"o", "h", "low", "c"},
    EventType.OBSERVATION_NEWS: {"headline", "url"},
    EventType.OBSERVATION_PRICE: {"price"},
    # NOTE: deliberately absent from the missing-asset check below —
    # asset=None is legal for market-wide macro events (Appendix A).
    EventType.OBSERVATION_MACRO: {"kind", "scheduled_for", "expected_impact"},
    EventType.COMPLIANCE_VERDICT: {"status"},
}


def validate_payload(event: Event) -> str | None:
    """Return an error string if the event's payload is malformed, else None."""
    required = REQUIRED_KEYS.get(event.type)
    if required is None:
        return None
    missing = required - set(event.payload)
    if missing:
        return f"missing keys {sorted(missing)}"
    if event.asset is None and event.type in {
        EventType.OBSERVATION_BAR,
        EventType.OBSERVATION_NEWS,
        EventType.OBSERVATION_PRICE,
        EventType.COMPLIANCE_VERDICT,
    }:
        return "missing asset"
    return None
