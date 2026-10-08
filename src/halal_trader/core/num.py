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


def rounded(value: Any, digits: int) -> float | None:
    """``value`` as a float rounded to ``digits``, or None (see :func:`to_float`)."""
    number = to_float(value)
    return None if number is None else round(number, digits)


def money(value: Any) -> float | None:
    """Dollars as the API reports them: to the cent."""
    return rounded(value, 2)


def ratio(value: Any) -> float | None:
    """Weights, returns and other fractions as the API reports them: 5 decimals."""
    return rounded(value, 5)
