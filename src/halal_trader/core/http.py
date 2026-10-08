"""How every outbound HTTP client retries, paces itself and keeps secrets out of logs.

One policy, each client choosing its numbers: a transport error or a
retryable status (429, 5xx) is tried again after ``backoff * 2**attempt``
seconds, or what the server's ``Retry-After`` asks when that is longer,
never more than ``max_wait_s`` at once. Anything else is the caller's to
judge: :func:`request` returns the last response whatever its status, and
raises the last transport error once the tries are spent.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class Pacer:
    """Keeps successive requests at least ``min_interval_s`` apart (a rate limit)."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last = float("-inf")

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        delay = self._last + self.min_interval_s - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        self._last = loop.time()


def retry_after(response: httpx.Response) -> float | None:
    """Seconds a Retry-After header asks for (delta-seconds or an HTTP date), if any."""
    value = (response.headers.get("Retry-After") or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


def redact(text: str, *secrets: str) -> str:
    """``text`` with every non-empty secret replaced, for logs and alerts.

    An API that takes its key in the query string (FRED) puts it in every
    URL, and so in every httpx error's message."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


async def request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    label: str,
    retries: int = 0,
    backoff_s: float = 0.5,
    max_wait_s: float = 120.0,
    retry_status: frozenset[int] = RETRY_STATUS,
    pacer: Pacer | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Send one request, trying again up to ``retries`` times (see the module).

    ``label`` names the service in the retry warnings; the URL is left out of
    them, since some carry a key.
    """
    for attempt in range(retries + 1):
        if pacer is not None:
            await pacer.wait()
        final = attempt == retries
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            if final:
                raise
            delay = backoff_s * 2**attempt
            logger.warning(
                "%s: no answer (%s); retrying in %.1f s", label, type(exc).__name__, delay
            )
        else:
            if final or response.status_code not in retry_status:
                return response
            delay = max(retry_after(response) or 0.0, backoff_s * 2**attempt)
            logger.warning("%s: HTTP %d; retrying in %.1f s", label, response.status_code, delay)
        await asyncio.sleep(min(delay, max_wait_s))
    raise AssertionError("unreachable")  # pragma: no cover
