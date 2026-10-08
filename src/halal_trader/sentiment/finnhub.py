"""Finnhub's company-news endpoint: the one request every news reader makes.

The day-trader's prompt (stocks_news.py), the reactor (stocks_events.py) and
the shadow's perception (halabot/perception/sources/finnhub_news.py) each
poll it with their own pacing and failure handling; the request, the key's
transport and the timestamp format are here.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
TIMEOUT_S = 10.0


async def company_news(
    client: httpx.AsyncClient,
    api_key: str,
    symbol: str,
    *,
    days: int = 1,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """The symbol's articles of the last ``days`` days, as Finnhub returns them.

    Raises on a transport or HTTP error; the caller decides what a failure
    means. The key goes in a header, never ``?token=``: httpx errors carry
    the request URL into every log line.
    """
    today = today or datetime.now(UTC).date()
    response = await client.get(
        COMPANY_NEWS_URL,
        params={
            "symbol": symbol,
            "from": (today - timedelta(days=days)).isoformat(),
            "to": today.isoformat(),
        },
        headers={"X-Finnhub-Token": api_key},
    )
    response.raise_for_status()
    data = response.json()
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def published_at(value: object) -> datetime | None:
    """An article's ``datetime`` (UNIX seconds, or already a datetime) in UTC, or None."""
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except OverflowError, OSError, ValueError:
        return None
