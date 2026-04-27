"""Adapter: turn ``OptionsIVSnapshot`` rows into ``Catalyst`` rows.

Lets the existing ``StockCatalystFeed`` consume the options-IV
source via the same protocol (``async fetch(symbols) -> list[Catalyst]``)
without the IV module knowing what a Catalyst is. Same pattern we
use for FRED + EDGAR.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from halal_trader.trading.catalysts import Catalyst
from halal_trader.trading.options_iv import (
    OptionsIVSnapshot,
    YahooOptionsIV,
)


class OptionsIVCatalystSource:
    """:class:`CatalystSource` over the Yahoo options surface.

    Each ticker becomes one Catalyst tagged ``options_iv`` (or
    ``options_iv_elevated`` / ``options_iv_skew`` when the IV / skew
    is in actionable territory) so the prompt can flag it directly
    and the risk policy can opt in to size shrinkage.
    """

    def __init__(self, fetcher: YahooOptionsIV | None = None) -> None:
        self._fetcher = fetcher or YahooOptionsIV()

    async def fetch(self, symbols: Sequence[str]) -> list[Catalyst]:
        if not symbols:
            return []
        snapshots = await self._fetcher.fetch(symbols)
        if not snapshots:
            return []
        out: list[Catalyst] = []
        now = datetime.now(UTC)
        for snap in snapshots.values():
            kind = _kind_for(snap)
            out.append(
                Catalyst(
                    symbol=snap.symbol,
                    kind=kind,
                    title=(
                        f"Options: ATM IV {snap.atm_iv:.0%}, "
                        f"P-C skew {snap.put_call_skew:+.2%}, "
                        f"P/C vol {snap.put_call_volume_ratio:.2f}x"
                    ),
                    timestamp=now,
                    source="yahoo-options",
                    extra={
                        "spot": snap.spot,
                        "atm_iv": snap.atm_iv,
                        "put_call_skew": snap.put_call_skew,
                        "label": snap.label,
                    },
                )
            )
        return out

    async def aclose(self) -> None:
        await self._fetcher.aclose()


def _kind_for(snap: OptionsIVSnapshot) -> str:
    if snap.label == "elevated_iv":
        return "options_iv_elevated"
    if snap.label == "downside_premium":
        return "options_iv_skew"
    return "options_iv"
