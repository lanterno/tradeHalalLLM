"""BeliefSink — inline (synchronous) + single-worker ts-coalescing (Appendix F)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from halabot.belief.schema import ComplianceVerdict, EvidenceItem
from halabot.cognition.worker import CoalescingBeliefWorker, InlineBeliefSink

T0 = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


class FakeUpdater:
    """Records apply_evidence / set_compliance calls in order."""

    def __init__(self) -> None:
        self.evidence_calls: list[tuple[str, datetime, int, bool]] = []
        self.compliance_calls: list[tuple[str, str, datetime]] = []

    async def apply_evidence(self, asset, items, now, *, is_replay=False, correlation_id=None):
        self.evidence_calls.append((asset, now, len(items), is_replay))

    async def set_compliance(self, asset, verdict, now, *, correlation_id=None):
        self.compliance_calls.append((asset, verdict.status, now))


def _ev(direction=1.0, *, ts=T0):
    return EvidenceItem(source="x", direction=direction, weight=1.0, ts=ts)


# ── InlineBeliefSink ──
@pytest.mark.asyncio
async def test_inline_sink_applies_immediately():
    u = FakeUpdater()
    sink = InlineBeliefSink(u)  # type: ignore[arg-type]
    await sink.evidence("NVDA", T0, [_ev()])
    await sink.compliance("NVDA", ComplianceVerdict("NVDA", "halal"), T0)
    assert u.evidence_calls == [("NVDA", T0, 1, False)]
    assert u.compliance_calls == [("NVDA", "halal", T0)]


# ── CoalescingBeliefWorker ──
@pytest.mark.asyncio
async def test_worker_coalesces_consecutive_same_asset_evidence():
    u = FakeUpdater()
    w = CoalescingBeliefWorker(u)  # type: ignore[arg-type]
    # Three NVDA evidence jobs queued before any drain → one coalesced apply.
    await w.evidence("NVDA", T0, [_ev(ts=T0)])
    await w.evidence("NVDA", T0 + timedelta(minutes=1), [_ev(ts=T0 + timedelta(minutes=1))])
    await w.evidence("NVDA", T0 + timedelta(minutes=2), [_ev(ts=T0 + timedelta(minutes=2))])
    await w.drain()
    assert len(u.evidence_calls) == 1
    asset, now, n_items, _ = u.evidence_calls[0]
    assert asset == "NVDA"
    assert n_items == 3                              # all items merged
    assert now == T0 + timedelta(minutes=2)          # now = latest ts (monotonic)


@pytest.mark.asyncio
async def test_worker_does_not_coalesce_across_assets():
    u = FakeUpdater()
    w = CoalescingBeliefWorker(u)  # type: ignore[arg-type]
    await w.evidence("NVDA", T0, [_ev()])
    await w.evidence("AAPL", T0, [_ev()])
    await w.evidence("NVDA", T0, [_ev()])  # not consecutive with the first NVDA
    await w.drain()
    assets = [c[0] for c in u.evidence_calls]
    assert assets == ["NVDA", "AAPL", "NVDA"]  # order preserved, no cross-asset merge


@pytest.mark.asyncio
async def test_worker_flushes_evidence_before_compliance():
    u = FakeUpdater()
    w = CoalescingBeliefWorker(u)  # type: ignore[arg-type]
    await w.evidence("NVDA", T0, [_ev()])
    await w.compliance("NVDA", ComplianceVerdict("NVDA", "not_halal"), T0)
    await w.evidence("NVDA", T0, [_ev()])
    await w.drain()
    # Evidence (1) then compliance then evidence (2) — causal order preserved.
    assert len(u.evidence_calls) == 2
    assert u.compliance_calls == [("NVDA", "not_halal", T0)]


@pytest.mark.asyncio
async def test_worker_isolates_a_failing_asset_in_a_batch():
    # Regression: one asset's apply error must NOT discard other assets' writes
    # in the same drained batch.
    class _Flaky(FakeUpdater):
        async def apply_evidence(self, asset, items, now, *, is_replay=False, correlation_id=None):
            if asset == "BAD":
                raise RuntimeError("store down for BAD")
            await super().apply_evidence(asset, items, now, is_replay=is_replay)

    u = _Flaky()
    w = CoalescingBeliefWorker(u)  # type: ignore[arg-type]
    await w.evidence("NVDA", T0, [_ev()])
    await w.evidence("BAD", T0, [_ev()])  # raises
    await w.evidence("AAPL", T0, [_ev()])
    await w.drain()
    applied = {c[0] for c in u.evidence_calls}
    assert applied == {"NVDA", "AAPL"}  # BAD dropped, the rest survived


@pytest.mark.asyncio
async def test_worker_run_task_processes_in_background():
    u = FakeUpdater()
    w = CoalescingBeliefWorker(u)  # type: ignore[arg-type]
    w.start()
    try:
        await w.evidence("NVDA", T0, [_ev()])
        for _ in range(50):
            await asyncio.sleep(0)
            if u.evidence_calls:
                break
        assert len(u.evidence_calls) == 1
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_worker_stop_drains_pending():
    u = FakeUpdater()
    w = CoalescingBeliefWorker(u)  # type: ignore[arg-type]
    w.start()
    await w.evidence("NVDA", T0, [_ev()])
    await w.stop()  # must flush the queued write before tearing down
    assert len(u.evidence_calls) == 1
