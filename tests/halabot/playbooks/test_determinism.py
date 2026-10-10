"""Determinism: the same records for any worker count, input order or batch size."""

from __future__ import annotations

import random
import sys
from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.playbooks.loader import Window, WindowUnlock
from halabot.playbooks.records import MemorySink, outcomes_sha256
from halabot.playbooks.sim import run, simulate_many, start_time, worker_of
from halabot.playbooks.types import SimConfig
from tests.halabot.playbooks._seed import seed_calendar, seed_market
from tests.halabot.playbooks._support import Toy, ToyFactory
from tests.halabot.playbooks._synth import WEEK, random_market


def test_the_partition_is_stable() -> None:
    assert worker_of("AAPL", 6) == worker_of("AAPL", 6)
    assert {worker_of(f"S{i}", 6) for i in range(200)} == set(range(6))
    assert all(worker_of(f"S{i}", 1) == 0 for i in range(50))


def test_one_and_six_workers_give_identical_records() -> None:
    market = random_market(11, 40, sessions=3)

    def go(workers: int, stories):  # type: ignore[no-untyped-def]
        return simulate_many(
            stories,
            lambda s: Toy(s, sessions=3, target=0.005),
            market.paths,
            market.spy,
            market.ctx,
            SimConfig(),
            workers=workers,
            keep_transitions=True,
        )

    one = go(1, market.stories)
    shuffled = list(market.stories)
    random.Random(3).shuffle(shuffled)
    six = go(6, shuffled)
    assert sum(o.trade is not None for o in one) > 5
    assert outcomes_sha256(one) == outcomes_sha256(six)
    assert [o.story_id for o in one] == [o.story_id for o in six]


async def test_run_is_identical_for_one_and_six_worker_processes(engine: AsyncEngine) -> None:
    market = random_market(12, 16, sessions=3)
    await seed_calendar(engine, date(2016, 1, 4), date(2016, 3, 31))
    await seed_market(engine, market)

    async def go(workers: int, batch: int, parallel=None):  # type: ignore[no-untyped-def]
        sink = MemorySink(run_id="00000000-0000-0000-0000-000000000001")
        summary = await run(
            engine,
            market.stories,
            ToyFactory(sessions=3, target=0.005),
            context=market.ctx,
            window=Window.GATE,
            window_end=date(2016, 3, 31),
            cfg=SimConfig(),
            unlock=WindowUnlock(),
            sink=sink,
            workers=workers,
            batch_paths=batch,
            parallel=parallel,
        )
        return sink, summary

    one, s1 = await go(1, 250)
    six, s6 = await go(6, 3)  # six partitions, many small batches (in-process on macOS)
    spawned, s3 = await go(3, 4, "spawn")  # a spawn pool: the factory and context pickle
    assert outcomes_sha256(spawned.outcomes) == outcomes_sha256(one.outcomes)
    assert s3.trades == s1.trades
    if sys.platform != "darwin":  # a fork pool exists only off macOS
        forked, _ = await go(6, 3, "fork")
        assert outcomes_sha256(forked.outcomes) == outcomes_sha256(one.outcomes)
    started = [st for st in market.stories if start_time(st) is not None]
    assert len(started) < len(market.stories)  # some never become NSN: news carriers only
    assert s1.outcomes == len(one.outcomes) == s1.started == len(started)
    assert s1.trades > 0 and s1.trades == s6.trades
    assert outcomes_sha256(one.outcomes) == outcomes_sha256(six.outcomes)
    direct = simulate_many(
        market.stories,
        lambda s: Toy(s, sessions=3, target=0.005),
        market.paths,
        market.spy,
        market.ctx,
        SimConfig(),
        run_id="00000000-0000-0000-0000-000000000001",
    )
    assert outcomes_sha256(direct) == outcomes_sha256(one.outcomes)  # the DB round trip is exact
    assert WEEK[0] == date(2016, 3, 7)
