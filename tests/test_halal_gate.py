"""Order-time halal gate tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa

from halal_trader.db.repository import Repository
from halal_trader.halal.gate import halal_gate


def _screener(*, halal: bool, refresh_raises: Exception | None = None):
    s = MagicMock()
    s.is_halal = AsyncMock(return_value=halal)
    s.refresh_if_stale = AsyncMock(side_effect=refresh_raises)
    return s


async def test_halal_symbol_returns_screening_id(engine):
    repo = Repository(engine)
    sid = await halal_gate(repo, screener=_screener(halal=True), symbol="AAPL", asset_class="stock")
    assert sid is not None and sid > 0

    loaded = await repo.get_halal_screening(sid)
    assert loaded["decision"] == "halal"


async def test_non_halal_symbol_returns_none_but_records_audit(engine):
    """Even when blocking a trade, we persist the screening row for audit."""
    repo = Repository(engine)
    sid = await halal_gate(
        repo, screener=_screener(halal=False), symbol="GMBL", asset_class="stock"
    )
    assert sid is None
    async with engine.begin() as conn:
        row = await conn.execute(
            sa.text("SELECT decision FROM halal_screenings WHERE symbol = 'GMBL'")
        )
        assert row.scalar_one() == "not_halal"


async def test_screener_exception_treated_as_non_halal(engine):
    """A screener outage must never let a non-compliant trade through."""
    repo = Repository(engine)
    screener = _screener(halal=True)
    screener.is_halal = AsyncMock(side_effect=RuntimeError("network gone"))
    sid = await halal_gate(repo, screener=screener, symbol="AAPL", asset_class="stock")
    assert sid is None


async def test_refresh_failure_does_not_abort_gate(engine):
    """A flaky cache refresh should still allow the is_halal check to run."""
    repo = Repository(engine)
    screener = _screener(halal=True, refresh_raises=RuntimeError("zoya 503"))
    sid = await halal_gate(repo, screener=screener, symbol="AAPL", asset_class="stock")
    assert sid is not None


async def test_invalid_asset_class_raises(engine):
    repo = Repository(engine)
    with pytest.raises(ValueError):
        await halal_gate(repo, screener=_screener(halal=True), symbol="X", asset_class="options")


async def test_screener_without_refresh_method_still_works(engine):
    """Older screeners may not expose refresh_if_stale — gate should tolerate."""
    repo = Repository(engine)
    screener = MagicMock(spec=["is_halal"])
    screener.is_halal = AsyncMock(return_value=True)
    sid = await halal_gate(repo, screener=screener, symbol="AAPL", asset_class="stock")
    assert sid is not None
