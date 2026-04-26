"""Two-source halal-compliance corroboration.

A single screening provider is a single point of failure: an outage or
silent data drift can lead us to either trade a non-compliant symbol or
sit out a fully-compliant one. The strict-mode policy below requires
*both* sources to agree the symbol is halal before we'll trade it.

Wires onto the existing :class:`ComplianceScreener` /
:class:`CryptoComplianceScreener` Protocols so individual provider
implementations don't need to change. A real second source (Wahed,
IdealRatings, etc.) lands in a follow-up; this module is the contract +
test seam so callers can adopt it now.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Sequence

from halal_trader.domain.ports import ComplianceScreener, CryptoComplianceScreener

logger = logging.getLogger(__name__)


class CorroborationPolicy(str, Enum):
    """How to combine two sources' opinions on a symbol.

    * ``UNANIMOUS`` — both must say halal (the conservative default;
      matches the spirit of the audit FK invariant).
    * ``MAJORITY_PRIMARY`` — primary wins; secondary used only when
      primary returns no opinion. Useful while a new second source is
      being shadow-validated.
    """

    UNANIMOUS = "unanimous"
    MAJORITY_PRIMARY = "majority_primary"


class CorroboratingScreener:
    """Wraps two stock screeners with a configurable agreement policy.

    Implements :class:`ComplianceScreener` so it can be swapped in
    transparently anywhere the existing single-source screener lives.
    """

    def __init__(
        self,
        primary: ComplianceScreener,
        secondary: ComplianceScreener,
        *,
        policy: CorroborationPolicy = CorroborationPolicy.UNANIMOUS,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._policy = policy

    async def ensure_cache(self, symbols: list[str] | None = None) -> None:
        # Refresh both caches concurrently — they're independent network calls.
        await asyncio.gather(
            self._primary.ensure_cache(symbols),
            self._secondary.ensure_cache(symbols),
            return_exceptions=False,
        )

    async def is_halal(self, symbol: str) -> bool:
        primary_ok, secondary_ok = await asyncio.gather(
            self._primary.is_halal(symbol),
            self._secondary.is_halal(symbol),
            return_exceptions=False,
        )
        return _decide(primary_ok, secondary_ok, self._policy)

    async def get_halal_symbols(self) -> list[str]:
        primary, secondary = await asyncio.gather(
            self._primary.get_halal_symbols(),
            self._secondary.get_halal_symbols(),
            return_exceptions=False,
        )
        return _combine_lists(primary, secondary, self._policy)

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        primary, secondary = await asyncio.gather(
            self._primary.filter_halal(symbols),
            self._secondary.filter_halal(symbols),
            return_exceptions=False,
        )
        return _combine_lists(primary, secondary, self._policy)


class CorroboratingCryptoScreener:
    """Crypto twin of :class:`CorroboratingScreener`."""

    def __init__(
        self,
        primary: CryptoComplianceScreener,
        secondary: CryptoComplianceScreener,
        *,
        policy: CorroborationPolicy = CorroborationPolicy.UNANIMOUS,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._policy = policy

    async def refresh_screening(self, symbols: list[str] | None = None) -> None:
        await asyncio.gather(
            self._primary.refresh_screening(symbols),
            self._secondary.refresh_screening(symbols),
            return_exceptions=False,
        )

    async def is_halal(self, symbol: str) -> bool:
        primary_ok, secondary_ok = await asyncio.gather(
            self._primary.is_halal(symbol),
            self._secondary.is_halal(symbol),
            return_exceptions=False,
        )
        return _decide(primary_ok, secondary_ok, self._policy)

    async def get_halal_pairs(self) -> list[str]:
        primary, secondary = await asyncio.gather(
            self._primary.get_halal_pairs(),
            self._secondary.get_halal_pairs(),
            return_exceptions=False,
        )
        return _combine_lists(primary, secondary, self._policy)

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        primary, secondary = await asyncio.gather(
            self._primary.filter_halal(symbols),
            self._secondary.filter_halal(symbols),
            return_exceptions=False,
        )
        return _combine_lists(primary, secondary, self._policy)


def _decide(primary: bool, secondary: bool, policy: CorroborationPolicy) -> bool:
    if policy is CorroborationPolicy.UNANIMOUS:
        return bool(primary and secondary)
    # MAJORITY_PRIMARY: primary's vote wins. Secondary recorded for audit
    # only — caller should be logging both opinions out-of-band so we can
    # evaluate the secondary before promoting it to UNANIMOUS.
    if not primary and secondary:
        logger.warning(
            "MAJORITY_PRIMARY: primary said not_halal, secondary said halal — "
            "deferring to primary. Promote secondary only after audit."
        )
    return primary


def _combine_lists(
    primary: Sequence[str], secondary: Sequence[str], policy: CorroborationPolicy
) -> list[str]:
    if policy is CorroborationPolicy.UNANIMOUS:
        return sorted(set(primary).intersection(secondary))
    # MAJORITY_PRIMARY: primary set is authoritative.
    return sorted(set(primary))
