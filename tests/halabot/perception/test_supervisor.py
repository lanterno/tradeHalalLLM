"""SourceSupervisor — restart-on-error + clean shutdown (INV-1/INV-2)."""

from __future__ import annotations

import asyncio

import pytest

from halabot.perception.base import Emit, SourceSupervisor
from halabot.platform.events import Event


async def _noop_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)  # yield control without real delay


async def _emit(_e: Event) -> None:
    pass


class _CrashThenBlock:
    """Crashes its first ``crashes`` runs, then blocks forever (healthy)."""

    name = "crashy"

    def __init__(self, crashes: int):
        self._left = crashes
        self.runs = 0
        self.healthy = asyncio.Event()

    async def run(self, emit: Emit) -> None:
        self.runs += 1
        if self._left > 0:
            self._left -= 1
            raise RuntimeError("boom")
        self.healthy.set()
        await asyncio.Event().wait()  # block until cancelled


@pytest.mark.asyncio
async def test_supervisor_restarts_until_healthy():
    src = _CrashThenBlock(crashes=2)
    sup = SourceSupervisor(restart_backoff_s=0.0, sleep=_noop_sleep)
    sup.start([src], _emit)
    await asyncio.wait_for(src.healthy.wait(), timeout=2.0)
    assert src.runs == 3  # crashed twice, healthy on the third
    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_stop_cancels_cleanly():
    src = _CrashThenBlock(crashes=0)
    sup = SourceSupervisor(restart_backoff_s=0.0, sleep=_noop_sleep)
    sup.start([src], _emit)
    await asyncio.wait_for(src.healthy.wait(), timeout=2.0)
    await sup.stop()  # must return without hanging or raising


@pytest.mark.asyncio
async def test_one_source_crash_does_not_affect_another():
    crashy = _CrashThenBlock(crashes=1)
    healthy = _CrashThenBlock(crashes=0)
    healthy.name = "healthy"
    sup = SourceSupervisor(restart_backoff_s=0.0, sleep=_noop_sleep)
    sup.start([crashy, healthy], _emit)
    await asyncio.wait_for(healthy.healthy.wait(), timeout=2.0)
    await asyncio.wait_for(crashy.healthy.wait(), timeout=2.0)  # recovered independently
    await sup.stop()
