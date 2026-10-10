"""A re-screen of a few names sizes the index veto by the date's other screened names."""

from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from halal_trader.compliance.aaoifi import ScreenResult
from halal_trader.compliance.index_veto import IndexView, Peer, apply_veto
from halal_trader.compliance.runner import run_screen
from tests.test_compliance_screen import FakeSec

HELD = {f"H{i}": 10e9 + i * 1e9 for i in range(40)}  # the range starts near 17.8B
SPUS = IndexView("SPUS", date(2025, 11, 25), frozenset(HELD), frozenset({"held by name"}))


def test_peers_size_the_range_a_few_names_cannot() -> None:
    few = [ScreenResult("BIG", "halal", [], {"market_cap": 300e9})]
    assert apply_veto(few, {}, [SPUS])[0].verdict == "halal"  # 1 priced: no range
    peers = {s: Peer(c, "") for s, c in HELD.items()}
    (big,) = apply_veto(few, {}, [SPUS], peers)
    assert big.verdict == "not_halal"
    assert "market cap >= 17.8B" in big.reasons[-1]


def test_peers_are_never_vetoed_and_their_names_count_as_holdings() -> None:
    peers = {s: Peer(c, "") for s, c in list(HELD.items())[:29]}
    peers["NAMED"] = Peer(45e9, "Held By Name Inc.")  # the 30th held name, by its SEC name
    peers["OUT"] = Peer(500e9, "")  # excluded, but not part of this run
    few = [ScreenResult("SMALL", "halal", [], {"market_cap": 1e9})]
    assert apply_veto(few, {}, [SPUS], peers) == few
    big = [ScreenResult("BIG", "halal", [], {"market_cap": 300e9})]
    assert apply_veto(big, {}, [SPUS], peers)[0].verdict == "not_halal"


def test_this_runs_own_figures_win_over_a_peer_row_of_the_same_name() -> None:
    peers = {s: Peer(c, "") for s, c in HELD.items()} | {"BIG": Peer(1e9, "")}
    (big,) = apply_veto(
        [ScreenResult("BIG", "halal", [], {"market_cap": 300e9})], {}, [SPUS], peers
    )
    assert big.verdict == "not_halal"


async def test_a_run_of_one_name_with_peers_vetoes_as_the_full_run_would(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO daily_bars (symbol, day, adjustment, open, high, low, close, "
                "volume, fetched_at) VALUES ('SOFT', '2026-09-30', 'raw', 100, 100, 100, 100, "
                "1, now())"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO etf_holdings (etf, filed, period_end, ticker, name, cusip, "
                "weight_pct) VALUES ('SPUS', '2026-09-01', '2026-07-31', :t, :t, '', 1.0)"
            ),
            [{"t": f"H{i}"} for i in range(40)],
        )
    as_of = date(2026, 10, 1)
    # SOFT's market cap is 100k (1,000 shares at $100); the held names' run 50k to 89k.
    peers = {f"H{i}": Peer(50_000.0 + i * 1_000.0, "") for i in range(40)}

    (alone,) = await run_screen(FakeSec(), engine, ["SOFT"], as_of)  # type: ignore[arg-type]
    assert alone.verdict == "halal"
    (sized,) = await run_screen(FakeSec(), engine, ["SOFT"], as_of, peers=peers)  # type: ignore[arg-type]
    assert sized.verdict == "not_halal"
    assert "excluded by SPUS's Shariah index" in sized.reasons[-1]
    async with engine.connect() as conn:
        stored = (
            await conn.execute(text("SELECT symbol, verdict FROM halal_screen_results"))
        ).all()
    assert [tuple(r) for r in stored] == [("SOFT", "not_halal")]  # peers are not written
