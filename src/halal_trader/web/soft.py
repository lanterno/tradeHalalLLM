"""How a page survives one missing piece: a read that fails degrades to a
fallback (logged at debug) instead of failing the whole response."""

from __future__ import annotations

import logging
from collections.abc import Awaitable

logger = logging.getLogger(__name__)


async def soft[T](what: str, read: Awaitable[T], fallback: T) -> T:
    """``await read``, or ``fallback`` if it raises (a table not created yet,
    a schema mid-migration): one missing source must not 500 the page."""
    try:
        return await read
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        logger.debug("%s degraded: %r", what, exc)
        return fallback
