"""The belief updater — the heart of the engine (REARCHITECTURE L3, ★).

``apply_evidence`` is incremental and **deterministic-first**: decay → merge →
recompute regime/direction/levels/conviction with NO LLM, then refresh the
thesis with the LLM **only** on a material shift and only when the LLM is
healthy (INV-1). It depends on injected protocols for its collaborators
(regime classifier, level engine, calibrator, thesis writer, price/position
sources, LLM gate), so cognition/conviction wire in during their own phases and
the updater is testable in isolation with fakes.

Review fixes baked in: snapshot ``prev`` BEFORE mutation and compare raw-to-raw
in ``material_shift`` (R-11); drift/anomaly flags flow into ``conviction_raw``
(R-12); replay suppresses invalidation side-effects (fix R, bootstrap).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from halabot.belief.evidence import Calendar, decay, has_flag, merge, weighted_sum
from halabot.belief.levels import update_levels  # noqa: F401  (re-exported for default engine)
from halabot.belief.schema import (
    BeliefState,
    ComplianceVerdict,
    Direction,
    EvidenceItem,
    Levels,
    Regime,
    band_index,
    regime_support,
)
from halabot.belief.store import BeliefStore
from halabot.conviction.raw import Calibrator, conviction_raw
from halabot.platform.bus import EventBus
from halabot.platform.clock import Clock
from halabot.platform.events import EventType, new_event

logger = logging.getLogger(__name__)


# ── injected collaborator protocols ──
class RegimeClassifier(Protocol):
    def classify(self, evidence: list[EvidenceItem]) -> tuple[Regime, float]: ...


class LevelEngine(Protocol):
    async def levels_for(self, asset: str, prev: Levels) -> Levels: ...


class ThesisWriter(Protocol):
    async def write(self, belief: BeliefState) -> str: ...


class PriceSource(Protocol):
    def last_price(self, asset: str) -> float | None: ...


class PositionSource(Protocol):
    def has_position(self, asset: str) -> bool: ...


class LLMGate(Protocol):
    def available(self) -> bool: ...
    def breaker_open(self) -> bool: ...


@dataclass(frozen=True)
class UpdaterConfig:
    long_threshold: float = 0.05
    evidence_decay_halflife_min: float = 240.0
    catalyst_impact_threshold: float = 0.7
    max_thesis_age: timedelta = timedelta(hours=4)
    llm_thesis_enabled: bool = True


def material_shift(
    prev: BeliefState,
    *,
    new_raw: float,
    new_regime: Regime,
    now: datetime,
    has_open_position: bool,
    catalyst_impact_threshold: float,
    max_thesis_age: timedelta,
) -> bool:
    """Should we spend an LLM call to refresh the thesis? (REARCHITECTURE B.3)

    ``prev`` MUST be the previous *persisted* belief (snapshotted before this
    update mutated anything), so the comparisons are real deltas, not a value
    against itself (R-11). The conviction comparison is RAW-vs-RAW
    (``prev.conviction_raw`` vs ``new_raw``) — never calibrated-vs-raw (R-11).
    """
    if prev.regime != new_regime:
        return True  # regime flip
    if band_index(prev.conviction_raw) != band_index(new_raw):
        return True  # crossed a raw-conviction band edge
    if any(
        c.is_imminent(now) and c.expected_impact >= catalyst_impact_threshold
        for c in prev.catalysts_pending
    ):
        return True  # high-impact catalyst landing
    if has_open_position and prev.last_thesis_refresh is None:
        return True  # never wrote a thesis for a position we hold
    if (
        has_open_position
        and prev.last_thesis_refresh is not None
        and (now - prev.last_thesis_refresh) > max_thesis_age
    ):
        return True  # stale narrative on a live position
    return False


@dataclass
class BeliefUpdater:
    """Applies evidence to a belief and persists a new version."""

    store: BeliefStore
    bus: EventBus
    clock: Clock
    calendar: Calendar
    regime: RegimeClassifier
    levels: LevelEngine
    calibrator: Calibrator
    thesis_writer: ThesisWriter
    prices: PriceSource
    positions: PositionSource
    llm: LLMGate
    config: UpdaterConfig = field(default_factory=UpdaterConfig)

    async def set_compliance(
        self, asset: str, verdict: ComplianceVerdict, now: datetime
    ) -> BeliefState:
        """Stamp a halal verdict onto the asset's belief (INV-7 ingestion).

        Transient-safe (INV-2): a ``transient_error`` verdict never overwrites a
        real prior verdict — a screening outage must not flip a belief. Persists
        a new version and publishes ``belief.updated`` so downstream (policy)
        re-evaluates tradeability.

        Lapsed compliance (INV-7, fix R-05): when a *real* (non-transient)
        ``not_halal``/``doubtful`` verdict lands on a position we currently hold,
        emit ``belief.invalidated(compliance_lapsed)`` BEFORE the update so the
        policy drives target→0 and the position is force-exited regardless of
        conviction or P&L. A transient error never triggers this (INV-2).
        """
        b = await self.store.get(asset) or BeliefState.neutral(asset)
        if verdict.transient_error and b.halal is not None and not b.halal.transient_error:
            return b  # keep the good prior verdict (INV-2)
        b.halal = verdict
        b.last_updated = now
        version = await self.store.put(b)
        b.version = version

        lapsed = (
            not verdict.transient_error
            and verdict.status in ("not_halal", "doubtful")
            and self.positions.has_position(asset)
        )
        if lapsed:
            logger.warning(
                "compliance lapsed on HELD %s: status=%s — forcing exit (INV-7)",
                asset,
                verdict.status,
            )
            await self.bus.publish(
                new_event(
                    self.clock,
                    EventType.BELIEF_INVALIDATED,
                    source="belief.compliance",
                    asset=asset,
                    payload={
                        "version": version,
                        "reason": "compliance_lapsed",
                        "status": verdict.status,
                        "detail": verdict.detail,
                    },
                )
            )
        await self.bus.publish(
            new_event(
                self.clock,
                EventType.BELIEF_UPDATED,
                source="belief.compliance",
                asset=asset,
                payload=_summary(b),
            )
        )
        return b

    async def apply_evidence(
        self,
        asset: str,
        items: list[EvidenceItem],
        now: datetime,
        *,
        is_replay: bool = False,
    ) -> BeliefState:
        """Fold ``items`` into ``asset``'s belief, persist, and publish.

        Passing ``items=[]`` is a decay-only update — the heartbeat tick uses
        this so conviction fades on the passage of time even with no new data
        (fix R-08). ``is_replay`` suppresses invalidation side-effects so
        bootstrap replay warms beliefs without firing exits against historical
        prices (fix R, bootstrap).
        """
        b = await self.store.get(asset) or BeliefState.neutral(asset)
        prev = deepcopy(b)  # ★ snapshot BEFORE mutation (R-11)

        # 1. decay (trading-time) + merge (event_id dedup)
        b.evidence = decay(
            b.evidence,
            now,
            halflife_min=self.config.evidence_decay_halflife_min,
            calendar=self.calendar,
        )
        b.evidence = merge(b.evidence, items)

        # 2. deterministic fields — one `signed` source for direction + conviction
        signed = weighted_sum(b.evidence)
        b.direction = (
            Direction.LONG_BIAS if signed > self.config.long_threshold else Direction.NEUTRAL
        )
        b.regime, b.regime_confidence = self.regime.classify(b.evidence)
        b.levels = await self.levels.levels_for(asset, prev.levels)

        # 3. raw conviction (LLM-free) with drift/anomaly flags wired (R-12).
        #    The regime factor is categorical LONG support (regime_support),
        #    not the classifier's confidence — so a trend outranks a range for a
        #    long bet rather than the reverse (live-data fix, 2026-05-28).
        raw = conviction_raw(
            b.evidence,
            regime_support(b.regime),
            drift_flag=has_flag(b.evidence, "drift"),
            anomaly_flag=has_flag(b.evidence, "anomaly"),
        )
        b.conviction_raw = raw
        b.conviction = await self.calibrator.calibrate(asset, raw, features=_feature_vec(b))

        # 4. thesis refresh — material shift AND a healthy LLM (triple-guarded)
        if (
            self.config.llm_thesis_enabled
            and self.llm.available()
            and not self.llm.breaker_open()
            and material_shift(
                prev,
                new_raw=raw,
                new_regime=b.regime,
                now=now,
                has_open_position=self.positions.has_position(asset),
                catalyst_impact_threshold=self.config.catalyst_impact_threshold,
                max_thesis_age=self.config.max_thesis_age,
            )
        ):
            try:
                b.thesis = await self.thesis_writer.write(b)
                b.last_thesis_refresh = now
            except Exception as exc:  # noqa: BLE001 — LLM failure must not break the update (INV-1)
                logger.warning("thesis refresh failed for %s: %r", asset, exc)

        # 5. invalidation — live price only; never during replay (fix R, bootstrap)
        invalidated = False
        if not is_replay and b.levels.invalidation is not None:
            px = self.prices.last_price(asset)
            if px is not None and px < b.levels.invalidation:
                invalidated = True

        b.last_updated = now
        version = await self.store.put(b)
        b.version = version

        if invalidated:
            await self.bus.publish(
                new_event(
                    self.clock,
                    EventType.BELIEF_INVALIDATED,
                    source="belief.updater",
                    asset=asset,
                    payload={
                        "version": version,
                        "reason": "price_break",
                        "invalidation_level": b.levels.invalidation,
                        "last_price": self.prices.last_price(asset),
                    },
                )
            )
        await self.bus.publish(
            new_event(
                self.clock,
                EventType.BELIEF_UPDATED,
                source="belief.updater",
                asset=asset,
                payload=_summary(b),
            )
        )
        return b


def _feature_vec(b: BeliefState) -> dict[str, Any]:
    """Scoring-time features for the calibrator (training uses the entry
    snapshot, not these — no leakage)."""
    return {
        "regime": str(b.regime),
        "regime_confidence": b.regime_confidence,
        "raw": b.conviction_raw,
        "n_evidence": len(b.evidence),
        "drift": has_flag(b.evidence, "drift"),
        "anomaly": has_flag(b.evidence, "anomaly"),
    }


def _summary(b: BeliefState) -> dict[str, Any]:
    return {
        "version": b.version,
        "regime": str(b.regime),
        "regime_confidence": round(b.regime_confidence, 4),
        "direction": str(b.direction),
        "conviction": round(b.conviction, 4),
        "conviction_raw": round(b.conviction_raw, 4),
        "invalidation": b.levels.invalidation,
        "n_evidence": len(b.evidence),
    }
