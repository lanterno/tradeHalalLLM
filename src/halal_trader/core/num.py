"""Reading numbers out of payloads and rows that may not hold one."""

from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float | None:
    """``value`` as a float, or None when it is missing (None, "") or not a number."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None
