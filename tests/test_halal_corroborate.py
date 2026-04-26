"""Two-source corroborating-screener policy tests."""

from __future__ import annotations

from halal_trader.halal.corroborate import (
    CorroboratingCryptoScreener,
    CorroboratingScreener,
    CorroborationPolicy,
)


class _StubScreener:
    """Minimal screener for tests; honours the ComplianceScreener Protocol."""

    def __init__(self, halal: dict[str, bool]) -> None:
        self._halal = halal
        self.refresh_calls = 0

    async def ensure_cache(self, symbols: list[str] | None = None) -> None:
        self.refresh_calls += 1

    async def refresh_screening(self, symbols: list[str] | None = None) -> None:
        self.refresh_calls += 1

    async def is_halal(self, symbol: str) -> bool:
        return self._halal.get(symbol, False)

    async def get_halal_symbols(self) -> list[str]:
        return [s for s, ok in self._halal.items() if ok]

    async def get_halal_pairs(self) -> list[str]:
        return [s for s, ok in self._halal.items() if ok]

    async def filter_halal(self, symbols: list[str]) -> list[str]:
        return [s for s in symbols if self._halal.get(s)]


# ── Stocks ─────────────────────────────────────────────────────


async def test_unanimous_requires_both_halal():
    primary = _StubScreener({"AAPL": True, "MSFT": True, "TSLA": False})
    secondary = _StubScreener({"AAPL": True, "MSFT": False, "TSLA": False})
    wrap = CorroboratingScreener(primary, secondary, policy=CorroborationPolicy.UNANIMOUS)

    assert await wrap.is_halal("AAPL") is True
    # MSFT: primary says halal, secondary disagrees → not halal under unanimous.
    assert await wrap.is_halal("MSFT") is False
    assert await wrap.is_halal("TSLA") is False


async def test_majority_primary_defers_to_primary():
    primary = _StubScreener({"AAPL": True, "MSFT": False})
    secondary = _StubScreener({"AAPL": False, "MSFT": True})
    wrap = CorroboratingScreener(primary, secondary, policy=CorroborationPolicy.MAJORITY_PRIMARY)
    assert await wrap.is_halal("AAPL") is True
    assert await wrap.is_halal("MSFT") is False


async def test_get_halal_symbols_unanimous_intersection():
    primary = _StubScreener({"AAPL": True, "MSFT": True, "GOOG": True})
    secondary = _StubScreener({"AAPL": True, "MSFT": True, "TSLA": True})
    wrap = CorroboratingScreener(primary, secondary)
    assert await wrap.get_halal_symbols() == ["AAPL", "MSFT"]


async def test_filter_halal_unanimous_intersection():
    primary = _StubScreener({"AAPL": True, "GOOG": False})
    secondary = _StubScreener({"AAPL": True, "GOOG": True})
    wrap = CorroboratingScreener(primary, secondary)
    assert await wrap.filter_halal(["AAPL", "GOOG"]) == ["AAPL"]


async def test_ensure_cache_refreshes_both_sources():
    primary = _StubScreener({})
    secondary = _StubScreener({})
    wrap = CorroboratingScreener(primary, secondary)
    await wrap.ensure_cache(["AAPL"])
    assert primary.refresh_calls == 1
    assert secondary.refresh_calls == 1


# ── Crypto ─────────────────────────────────────────────────────


async def test_crypto_unanimous_requires_both():
    primary = _StubScreener({"BTC": True, "ETH": True})
    secondary = _StubScreener({"BTC": True, "ETH": False})
    wrap = CorroboratingCryptoScreener(primary, secondary, policy=CorroborationPolicy.UNANIMOUS)
    assert await wrap.is_halal("BTC") is True
    assert await wrap.is_halal("ETH") is False


async def test_crypto_get_halal_pairs_intersection():
    primary = _StubScreener({"BTC": True, "ETH": True, "DOGE": True})
    secondary = _StubScreener({"BTC": True, "ETH": True, "PEPE": True})
    wrap = CorroboratingCryptoScreener(primary, secondary)
    assert await wrap.get_halal_pairs() == ["BTC", "ETH"]


async def test_majority_primary_logs_when_secondary_disagrees(caplog):
    primary = _StubScreener({"AAPL": False})
    secondary = _StubScreener({"AAPL": True})
    wrap = CorroboratingScreener(primary, secondary, policy=CorroborationPolicy.MAJORITY_PRIMARY)
    with caplog.at_level("WARNING"):
        await wrap.is_halal("AAPL")
    assert any("MAJORITY_PRIMARY" in rec.message for rec in caplog.records)


def test_policy_enum_values_pinned():
    """Wire format pin so config files can reference these strings."""
    assert CorroborationPolicy.UNANIMOUS.value == "unanimous"
    assert CorroborationPolicy.MAJORITY_PRIMARY.value == "majority_primary"


def test_corroborating_screener_satisfies_protocol():
    """Doc test — confirms the wrappers can be substituted at type level."""
    from halal_trader.domain.ports import ComplianceScreener, CryptoComplianceScreener

    primary = _StubScreener({})
    secondary = _StubScreener({})
    stock_wrap: ComplianceScreener = CorroboratingScreener(primary, secondary)
    crypto_wrap: CryptoComplianceScreener = CorroboratingCryptoScreener(primary, secondary)
    assert stock_wrap is not None and crypto_wrap is not None
