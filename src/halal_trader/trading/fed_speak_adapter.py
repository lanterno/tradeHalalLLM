"""Adapter: turn the Fed-speak signal into per-symbol ``Catalyst`` rows.

The Fed-speak module (``trading/fed_speak.py``) is universe-wide —
its hawkish/dovish drift applies to every U.S. equity symbol the bot
trades. This adapter implements the ``CatalystSource`` protocol so
the existing ``StockCatalystFeed`` can fan one signal out across all
requested symbols in one place.
"""

from __future__ import annotations

from collections.abc import Sequence

from halal_trader.trading.catalysts import Catalyst
from halal_trader.trading.fed_speak import (
    FedSpeakFetcher,
    fed_speak_to_catalysts,
)


class FedSpeakCatalystSource:
    """:class:`CatalystSource` over the Fed RSS feed."""

    def __init__(self, fetcher: FedSpeakFetcher | None = None) -> None:
        self._fetcher = fetcher or FedSpeakFetcher()

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols:
            return []
        signal = await self._fetcher.fetch()
        return fed_speak_to_catalysts(signal, symbols)

    async def aclose(self) -> None:
        await self._fetcher.aclose()
