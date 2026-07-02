"""Shared credit/quota-exhaustion detection for LLM call errors.

One source of truth for the substrings that mean "the account is out of
money/credits" — a non-transient failure where retries burn API calls
without any chance of success. Matched case-insensitively on the error
message because the SDK exception class isn't always preserved through
the wrapper layers (FallbackLLM, asyncio.wait_for, broad excepts).

Markers cover the OpenAI-compat shape (``insufficient_quota`` /
"exceeded your current quota") and OpenRouter's 402 "Insufficient
credits".
"""

from __future__ import annotations

QUOTA_ERROR_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "exceeded your current quota",
    "insufficient credits",
)


def is_quota_error(error: Exception | str) -> bool:
    """True when *error* indicates credit/quota exhaustion (non-transient)."""
    msg = str(error).lower()
    return any(marker in msg for marker in QUOTA_ERROR_MARKERS)
