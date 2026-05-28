"""Observation payload schemas — what perception emits (REARCHITECTURE Appendix A).

Perception reports *facts*; it does not interpret. Note ``NewsObservation``
carries only a cheap lexicon polarity (or None) — the LLM scoring of a headline
is COGNITION's job, not perception's, so "we saw news" (always works) is
decoupled from "we understood it" (LLM, may be down — INV-1).
"""

from __future__ import annotations

from typing import TypedDict


class PriceObservation(TypedDict):
    asset: str
    price: float
    bid: float | None
    ask: float | None


class BarObservation(TypedDict):
    asset: str
    tf: str  # "1Min" | "1Hour" | "1Day"
    o: float
    h: float
    low: float  # 'l' is ambiguous; spell it out
    c: float
    v: float
    bar_ts: str  # ISO-8601


class NewsObservation(TypedDict):
    asset: str
    headline: str
    summary: str
    url: str
    published_at: str
    source: str
    lexicon_polarity: float | None  # cheap pre-LLM score; None if the lexicon abstained


class MacroObservation(TypedDict):
    kind: str  # "CPI" | "FOMC" | "NFP" | "GDP" | "earnings"
    asset: str | None  # None = market-wide
    scheduled_for: str
    expected_impact: float  # 0..1
    actual: float | None
    consensus: float | None


class SentimentObservation(TypedDict):
    asset: str
    mention_velocity: float
    novelty: float
    net_polarity: float
    window_min: int


class OnchainObservation(TypedDict):
    asset: str
    signal: str  # "whale_inflow" | "whale_outflow" | "basis"
    magnitude: float
    detail: str
