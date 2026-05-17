"""Smoke tests for cycle init invariants.

The original microstructure / news helpers (``_build_microstructure_text`` /
``_build_news_text``) were extracted into ``BuildMicrostructureStage`` /
``BuildNewsStage`` (Wave B). Behaviour now lives in
``tests/test_cycle_stages.py``; the test added here is the only piece of
this file that wasn't covered by the stage-level tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from halal_trader.crypto.cycle import CryptoCycleService


def test_last_indicators_cache_initially_none():
    """The scheduler's adaptive-cadence selector reads this snapshot;
    it must default to ``None`` until the first cycle populates it."""
    cycle = CryptoCycleService(
        broker=MagicMock(),
        screener=MagicMock(),
        strategy=MagicMock(),
        executor=MagicMock(),
        portfolio=MagicMock(),
    )
    assert cycle.last_indicators_cache is None
