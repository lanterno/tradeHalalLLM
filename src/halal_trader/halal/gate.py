"""Order-time halal gate — last-line compliance check before placing a trade.

The strategy already filters its candidate universe through the halal
screener, but a screener cache that stales between the cycle's
filter-pass and the executor's place-order can let a freshly-flipped
non-compliant symbol through. The order-time gate closes that window:

* Re-check the symbol against the live cache (or re-screen if stale)
* Record the screening decision in ``halal_screenings`` so the trade
  is provably linked to a per-trade compliance verdict
* Refuse the order if the answer is anything but ``halal``

Usage::

    decision_id = await halal_gate(
        repo,
        screener=screener,
        symbol="AAPL",
        asset_class="stock",
    )
    if decision_id is None:
        return  # rejected, do not place order
    # ... place order ...
    await repo.record_trade(..., halal_screening_id=decision_id)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def halal_gate(
    repo: Any,
    *,
    screener: Any,
    symbol: str,
    asset_class: str,
    source: str = "cache",
) -> int | None:
    """Verify ``symbol`` is halal RIGHT NOW; record + return the screening id.

    Returns the new HalalScreening row id on success, or ``None`` to
    indicate the order must be rejected. The repository write happens
    on every call (success or failure) so the audit trail is complete
    even when we *block* the trade.

    The screener is expected to expose ``is_halal(symbol) -> bool`` and
    optionally ``refresh_if_stale`` for the mid-cycle freshness pass.
    Both stock and crypto screeners satisfy this surface.
    """
    if asset_class not in ("stock", "crypto"):
        raise ValueError(f"asset_class must be 'stock' or 'crypto'; got {asset_class!r}")

    refresh = getattr(screener, "refresh_if_stale", None)
    if refresh is not None:
        try:
            await refresh()
        except Exception as e:
            logger.debug("halal screener refresh_if_stale failed: %s", e)

    try:
        is_halal = bool(await screener.is_halal(symbol))
    except Exception as e:
        logger.warning("halal screener.is_halal raised for %s: %s", symbol, e)
        # Defensive — treat unknown as not_halal so a screener outage
        # never lets a non-compliant trade through.
        is_halal = False

    decision = "halal" if is_halal else "not_halal"
    sid = await repo.record_halal_screening(
        symbol=symbol,
        asset_class=asset_class,
        source=source,
        decision=decision,
        cache_hit=True,
    )

    if not is_halal:
        logger.warning(
            "halal gate REJECTED %s order on %s — recorded screening id %d",
            asset_class,
            symbol,
            sid,
        )
        return None

    return sid
