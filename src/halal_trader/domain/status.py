"""Trade lifecycle status enum.

Sentinel string statuses (`pending`, `submitted`, `filled`, …) are used
throughout the order execution + monitor + reconciler paths. This enum
gives them a single home so typos surface at lint time and downstream
code can match exhaustively.

The DB column stays ``str`` to avoid a migration; the enum compares
equal to its string value, so existing comparisons continue to work
both ways:

    >>> TradeStatus.FILLED == "filled"
    True
    >>> "filled" == TradeStatus.FILLED
    True
"""

from __future__ import annotations

from enum import StrEnum


class TradeStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELED = "canceled"
    CLOSED = "closed"
    ERROR = "error"

    @classmethod
    def is_terminal(cls, status: "TradeStatus | str") -> bool:
        """A status that means the order is fully resolved and won't change."""
        return status in {cls.FILLED, cls.REJECTED, cls.CANCELED, cls.CLOSED, cls.ERROR}

    @classmethod
    def is_open(cls, status: "TradeStatus | str") -> bool:
        """A status that may still produce fills or trade events."""
        return status in {cls.PENDING, cls.SUBMITTED, cls.PARTIALLY_FILLED}
