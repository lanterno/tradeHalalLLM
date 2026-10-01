"""UniverseComplianceSource — fresh halal verdicts, lapse on departure, empty = transient."""

from __future__ import annotations

from datetime import UTC, datetime

from halabot.perception.sources.universe_compliance import UniverseComplianceSource
from halabot.platform.clock import FakeClock
from halabot.platform.events import Event, EventType

CLOCK = FakeClock(datetime(2026, 10, 1, 14, 0, tzinfo=UTC))


class Universe:
    """A mutable universe the test can change between polls."""

    def __init__(self, *symbols: str) -> None:
        self.symbols = list(symbols)

    async def __call__(self) -> list[str]:
        return list(self.symbols)


async def _poll(src: UniverseComplianceSource) -> dict[str, Event]:
    out: list[Event] = []

    async def emit(e: Event) -> None:
        out.append(e)

    await src.poll_once(emit)
    assert all(e.type == EventType.COMPLIANCE_VERDICT for e in out)
    return {e.asset: e for e in out}


async def test_every_member_is_stamped_halal_on_each_poll() -> None:
    src = UniverseComplianceSource(Universe("AAPL", "NVDA"), CLOCK)
    for _ in range(2):  # re-emitted every cadence: that is what keeps screened_at fresh
        events = await _poll(src)
        assert {a: e.payload["status"] for a, e in events.items()} == {
            "AAPL": "halal",
            "NVDA": "halal",
        }
        assert all(e.payload["transient_error"] is False for e in events.values())


async def test_a_name_that_leaves_the_universe_gets_a_real_doubtful_verdict() -> None:
    universe = Universe("AAPL", "NVDA")
    src = UniverseComplianceSource(universe, CLOCK)
    await _poll(src)
    universe.symbols = ["AAPL"]

    events = await _poll(src)

    assert events["AAPL"].payload["status"] == "halal"
    lapsed = events["NVDA"].payload
    assert lapsed["status"] == "doubtful"  # real, non-transient -> lapse exit on a holding
    assert lapsed["transient_error"] is False
    assert lapsed["detail"] == "left the halal universe"


async def test_departure_is_reported_once_not_forever() -> None:
    universe = Universe("AAPL", "NVDA")
    src = UniverseComplianceSource(universe, CLOCK)
    await _poll(src)
    universe.symbols = ["AAPL"]
    await _poll(src)

    events = await _poll(src)

    assert set(events) == {"AAPL"}


async def test_empty_universe_is_transient_not_a_mass_lapse() -> None:
    universe = Universe("AAPL", "NVDA")
    src = UniverseComplianceSource(universe, CLOCK)
    await _poll(src)
    universe.symbols = []  # cache empty / unreachable

    assert await _poll(src) == {}

    universe.symbols = ["AAPL", "NVDA"]  # cache back: nobody "departed" in between
    events = await _poll(src)
    assert {e.payload["status"] for e in events.values()} == {"halal"}


async def test_excluded_benchmark_is_never_stamped_halal() -> None:
    src = UniverseComplianceSource(Universe("AAPL", "SPY"), CLOCK, exclude=frozenset({"SPY"}))

    events = await _poll(src)

    assert set(events) == {"AAPL"}
