"""Belief write sink — inline (synchronous) or single-worker ts-coalescing.

The :class:`CognitionRouter` produces evidence and hands belief *writes* to a
:class:`BeliefSink`. Two implementations:

* :class:`InlineBeliefSink` applies each write immediately on the calling task —
  deterministic and synchronous, used by tests and ``--once``.
* :class:`CoalescingBeliefWorker` queues writes and drains them on a **single**
  background task. Within a drained burst it merges *consecutive same-asset*
  evidence jobs and applies them as one ``apply_evidence`` with ``now`` = the
  batch's latest ts, so decay/merge see monotonic time and a strict-``ts`` replay
  reproduces the same belief version (Appendix F, INV-5).

Why a single worker rather than one task per asset (the Appendix-F ideal): the
Phase-3 shadow recomputes the whole portfolio on every ``belief.updated``, so two
asset workers publishing concurrently would race the shadow book. One serial
drain keeps global write order — the per-asset *coalescing* benefit without the
race. Per-asset parallel workers become safe once the policy is event-driven
(Phase 4); this is the seam for that.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from halabot.belief.schema import ComplianceVerdict, EvidenceItem
from halabot.belief.updater import BeliefUpdater

logger = logging.getLogger(__name__)


class BeliefSink(Protocol):
    async def evidence(
        self,
        asset: str,
        now: datetime,
        items: list[EvidenceItem],
        *,
        is_replay: bool = False,
        correlation_id: UUID | None = None,
    ) -> None: ...
    async def compliance(
        self,
        asset: str,
        verdict: ComplianceVerdict,
        now: datetime,
        *,
        correlation_id: UUID | None = None,
    ) -> None: ...


class InlineBeliefSink:
    """Applies writes immediately (synchronous, deterministic)."""

    def __init__(self, updater: BeliefUpdater) -> None:
        self._u = updater

    async def evidence(
        self,
        asset: str,
        now: datetime,
        items: list[EvidenceItem],
        *,
        is_replay: bool = False,
        correlation_id: UUID | None = None,
    ) -> None:
        await self._u.apply_evidence(
            asset, items, now, is_replay=is_replay, correlation_id=correlation_id
        )

    async def compliance(
        self,
        asset: str,
        verdict: ComplianceVerdict,
        now: datetime,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        await self._u.set_compliance(asset, verdict, now, correlation_id=correlation_id)


@dataclass
class _Ev:
    asset: str
    now: datetime
    items: list[EvidenceItem]
    is_replay: bool = False
    correlation_id: UUID | None = None


@dataclass
class _Co:
    asset: str
    verdict: ComplianceVerdict
    now: datetime
    correlation_id: UUID | None = None


class CoalescingBeliefWorker:
    """Single serial drain with per-asset ts-coalescing (see module docstring)."""

    def __init__(self, updater: BeliefUpdater) -> None:
        self._u = updater
        self._q: asyncio.Queue[_Ev | _Co] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        # Serializes batch application so a concurrent drain() (e.g. --once flush
        # or a test) never races the background _run on the same belief version.
        self._lock = asyncio.Lock()
        # _run waits on this rather than blocking on queue.get(), so it never
        # dequeues outside the lock (ordering safety).
        self._signal = asyncio.Event()

    async def evidence(
        self,
        asset: str,
        now: datetime,
        items: list[EvidenceItem],
        *,
        is_replay: bool = False,
        correlation_id: UUID | None = None,
    ) -> None:
        await self._q.put(_Ev(asset, now, items, is_replay, correlation_id))
        self._signal.set()

    async def compliance(
        self,
        asset: str,
        verdict: ComplianceVerdict,
        now: datetime,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        await self._q.put(_Co(asset, verdict, now, correlation_id))
        self._signal.set()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="belief-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        await self.drain()

    async def drain(self) -> None:
        """Process everything currently queued (for ``--once`` and shutdown).

        Shares the same locked drain-and-apply as :meth:`_run`, so a concurrent
        drain() can never reorder writes — nothing is dequeued outside the lock
        (fixes the get-before-lock ordering hazard)."""
        while not self._q.empty():
            await self._drain_locked()

    async def _run(self) -> None:
        while True:
            try:
                await self._signal.wait()  # woken by a put (no item dequeued here)
                await self._drain_locked()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — one bad batch never kills the worker (INV-1)
                logger.error("belief worker batch failed: %r", exc)

    async def _drain_locked(self) -> None:
        """Atomically (under the lock) clear the wake signal, dequeue everything
        currently queued, and apply it in FIFO order. Dequeue happens ONLY here,
        so _run and drain() never interleave dequeues → no reordering."""
        async with self._lock:
            self._signal.clear()  # a put during/after apply re-sets it → next pass
            jobs = self._drain_batch()
            if jobs:
                await self._apply(jobs)

    def _drain_batch(self) -> list[_Ev | _Co]:
        """Pop everything currently queued (non-blocking). Caller holds the lock."""
        jobs: list[_Ev | _Co] = []
        while True:
            try:
                jobs.append(self._q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return jobs

    async def _apply(self, jobs: list[_Ev | _Co]) -> None:
        i = 0
        while i < len(jobs):
            job = jobs[i]
            if isinstance(job, _Ev):
                asset = job.asset
                run: list[_Ev] = []
                while i < len(jobs) and isinstance(jobs[i], _Ev) and jobs[i].asset == asset:
                    run.append(jobs[i])  # type: ignore[arg-type]
                    i += 1
                items = [it for j in run for it in j.items]
                latest = max(run, key=lambda j: j.now)
                is_replay = all(j.is_replay for j in run)
                # Per-asset isolation: a transient store/bus error on ONE asset's
                # coalesced sub-batch must not discard the OTHER assets' writes in
                # this drain (the durable log keeps the observation; the next live
                # event for this asset re-derives it).
                try:
                    # Coalesced batch carries the latest event's correlation_id (the
                    # dominant chain — coalescing is best-effort, INV-5).
                    await self._u.apply_evidence(
                        asset,
                        items,
                        latest.now,
                        is_replay=is_replay,
                        correlation_id=latest.correlation_id,
                    )
                except Exception as exc:  # noqa: BLE001 — isolate this asset, keep the rest
                    logger.error("belief write failed for %s (dropped): %r", asset, exc)
            else:
                try:
                    await self._u.set_compliance(
                        job.asset, job.verdict, job.now, correlation_id=job.correlation_id
                    )
                except Exception as exc:  # noqa: BLE001 — isolate this asset, keep the rest
                    logger.error("compliance write failed for %s (dropped): %r", job.asset, exc)
                i += 1
