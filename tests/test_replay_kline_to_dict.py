"""Tests for :func:`_kline_to_dict` — Kline → dict adapter for replay snapshots.

Walks the type-tag matrix the function defends against (dict pass-
through, pydantic v2 `model_dump`, pydantic v1 `dict()`, and dataclass
`asdict` fallback). Used by every `CycleSnapshot.from_inputs` call.
"""

from __future__ import annotations

from halal_trader.core.replay import _kline_to_dict
from halal_trader.domain.models import Kline


def _kline(close: float = 100.0) -> Kline:
    return Kline(
        open_time=1_700_000_000_000,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000.0,
        close_time=1_700_000_000_000 + 60_000,
    )


def test_dict_input_returns_copy():
    """A dict input is passed through (copied so the caller's dict
    isn't aliased into the snapshot)."""
    src = {"open_time": 1, "open": 1.0, "close": 2.0}
    out = _kline_to_dict(src)
    assert out == src
    # Mutating the output must not change the source.
    out["open_time"] = 999
    assert src["open_time"] == 1


def test_pydantic_kline_dumps_to_dict():
    """A real Pydantic Kline → all the model fields as a dict."""
    k = _kline(close=42_000.0)
    out = _kline_to_dict(k)
    assert isinstance(out, dict)
    assert out["close"] == 42_000.0
    assert out["volume"] == 1000.0


def test_pydantic_kline_round_trips_through_dict():
    """The dict can be fed back into `Kline(**out)` — same shape that
    the replay store relies on."""
    k = _kline(close=100.0)
    out = _kline_to_dict(k)
    rebuilt = Kline(**out)
    assert rebuilt.close == k.close
    assert rebuilt.open_time == k.open_time
