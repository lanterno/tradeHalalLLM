"""Execution layer (REARCHITECTURE L6) — venue-agnostic orders, fills, reconcile.

DORMANT by design: nothing here is instantiated by ``app.build_engine``. The
engine runs read-only/shadow (Phase 3) until the Phase-3 A/B *significantly*
beats the live cycle (``analysis.significance.promotion_gate``) AND an operator
arms live mode (``ENGINE_LIVE`` + the un-loosenable SAFEGUARD floors, INV-9).
Phase-4 wiring (a later step) is the only place these components are connected to
a real venue. Until then they exist with full fake-venue unit coverage, never
touching a broker.
"""
