"""Composition root — assemble the read-only understanding engine.

Wires the full Phase 0–3 stack on Postgres: durable bus + event log, belief
store + deterministic updater, cognition (interpreters + router), and the
log-only shadow policy. Runs as a **separate entrypoint**, isolated from the
legacy stock bot — if the engine misbehaves the live bot is unaffected, and the
engine never executes (Phase 3 is shadow-only).

The LLM thesis writer is OFF by default here (the shadow runs fully
deterministic + free); flip ``llm_thesis_enabled`` + inject a real writer when
desired. Positions default to "none" (no broker wired in shadow); prices come
from the bar buffer's last close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.belief.evidence import ContinuousCalendar, RegularHoursCalendar
from halabot.belief.schema import BeliefState
from halabot.belief.store import PgBeliefStore
from halabot.belief.updater import BeliefUpdater, UpdaterConfig
from halabot.cognition.bars import BarBuffer, BufferPriceSource
from halabot.cognition.base import Interpreter
from halabot.cognition.interpreters import (
    AnomalyInterpreter,
    DriftInterpreter,
    ForecasterInterpreter,
    IndicatorInterpreter,
    MultiFrameInterpreter,
    NewsLexiconInterpreter,
    NewsLlmInterpreter,
    RelativeStrengthInterpreter,
    RsiInterpreter,
    SupportResistanceInterpreter,
    TrendAlignmentInterpreter,
    VolumeConfirmationInterpreter,
)
from halabot.cognition.level_engine import BarLevelEngine
from halabot.cognition.regime import EvidenceRegimeClassifier
from halabot.cognition.router import CognitionRouter
from halabot.cognition.thesis import LlmGate, LlmThesisWriter
from halabot.cognition.worker import BeliefSink, CoalescingBeliefWorker, InlineBeliefSink
from halabot.conviction.calibrator import FittedCalibrator
from halabot.learning.retrain import CalibratorRetrainer
from halabot.learning.shadow_outcomes import ShadowOutcomeTracker
from halabot.learning.telemetry import ConvictionScoreWriter, TargetWeightWriter
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import Clock, SystemClock
from halabot.platform.config import HalabotSettings, get_settings
from halabot.platform.db import bootstrap_schema, make_engine
from halabot.platform.event_log import PgEventLog
from halabot.policy.policy import Policy
from halabot.policy.portfolio import ShadowPortfolio
from halabot.policy.shadow import ShadowPolicyRunner
from halabot.policy.sizing import PolicyConfig
from halabot.risk.engine import BasicRiskEngine, RiskConfig

logger = logging.getLogger(__name__)


class _DisabledLLM:
    """LLM gate reporting unavailable — keeps the shadow deterministic + free."""

    def available(self) -> bool:
        return False

    def breaker_open(self) -> bool:
        return True


class _NoThesis:
    async def write(self, belief: BeliefState) -> str:
        return ""


def _make_halt_check(db_engine: AsyncEngine) -> Any:
    """A coroutine the policy calls to read the operator kill-switch (hb_control).
    Defined here (not via the api package) to avoid a composition→api dependency."""
    import sqlalchemy as sa

    from halabot.platform.db import control

    async def halted() -> bool:
        async with db_engine.connect() as conn:
            row = (await conn.execute(sa.select(control.c.halted).where(control.c.id == 1))).first()
        return bool(row[0]) if row is not None else False

    return halted


@dataclass
class Engine:
    """A running read-only engine: its bus is the ingress for observations."""

    bus: InProcessEventBus
    store: PgBeliefStore
    updater: BeliefUpdater
    router: CognitionRouter
    shadow: ShadowPolicyRunner
    outcomes: ShadowOutcomeTracker
    db_engine: AsyncEngine
    worker: CoalescingBeliefWorker | None = None
    conviction_writer: ConvictionScoreWriter | None = None
    target_writer: TargetWeightWriter | None = None
    retrainer: CalibratorRetrainer | None = None

    async def stop(self) -> None:
        self.router.stop()
        if self.worker is not None:
            await self.worker.stop()  # flush queued belief writes before teardown
        self.shadow.stop()
        self.outcomes.stop()
        if self.retrainer is not None:
            await self.retrainer.aclose()  # cancel any in-flight background retrain
        if self.conviction_writer is not None:
            self.conviction_writer.stop()
        if self.target_writer is not None:
            self.target_writer.stop()
        await self.db_engine.dispose()


async def build_engine(
    *,
    database_url: str | None = None,
    db_engine: AsyncEngine | None = None,
    clock: Clock | None = None,
    settings: HalabotSettings | None = None,
    updater_config: UpdaterConfig | None = None,
    policy_config: PolicyConfig | None = None,
    risk_config: RiskConfig | None = None,
    thesis_writer: Any | None = None,
    llm_gate: Any | None = None,
    positions: Any | None = None,
    coalesce: bool = False,
    bootstrap: bool = False,
) -> Engine:
    """Assemble + start the read-only engine. Provide ``db_engine`` (tests) or
    ``database_url``. Configs default from ``settings`` (or ``get_settings()``);
    explicit ``*_config`` args override. Subscriptions are live on return; feed
    ``observation.*`` events into ``engine.bus`` to drive it."""
    if db_engine is None:
        if database_url is None:
            raise ValueError("build_engine requires db_engine or database_url")
        db_engine = make_engine(database_url)
    await bootstrap_schema(db_engine)

    s = settings or get_settings()
    updater_config = updater_config or UpdaterConfig(
        long_threshold=s.belief.long_threshold,
        evidence_decay_halflife_min=s.belief.evidence_decay_halflife_min,
        catalyst_impact_threshold=s.belief.catalyst_impact_threshold,
        max_thesis_age=timedelta(hours=s.belief.thesis_max_age_h),
        llm_thesis_enabled=s.cognition.llm_thesis_enabled,
    )
    policy_config = policy_config or PolicyConfig(
        conviction_entry_band=s.policy.conviction_entry_band,
        conviction_exit_band=s.policy.conviction_exit_band,
        max_weight_per_asset=s.policy.max_weight_per_asset,
        max_gross_exposure=s.policy.max_gross_exposure,
        target_rebalance_threshold=s.policy.target_rebalance_threshold,
        max_open_positions=s.engine.max_open_positions,
        relstrength_gate=s.policy.relstrength_gate,
    )
    risk_config = risk_config or RiskConfig(
        max_portfolio_heat_pct=s.risk.max_portfolio_heat_pct,
        max_drawdown_pct=s.risk.max_drawdown_pct,
        daily_loss_limit=s.risk.daily_loss_limit,
    )

    clock = clock or SystemClock()
    bus = InProcessEventBus(PgEventLog(db_engine))
    store = PgBeliefStore(db_engine)
    buffer = BarBuffer()
    prices = BufferPriceSource(buffer)
    # One shadow book, shared as the updater's PositionSource so the engine's
    # INV-7 lapsed-compliance force-exit path keys off what the shadow "holds"
    # (no broker positions exist in shadow). An explicit `positions` overrides.
    shadow_book = ShadowPortfolio()

    # Sparse LLM thesis writer (the only LLM touch — INV-1, triple-guarded by the
    # updater). OFF by default (costs money); enable via cognition.llm_thesis_enabled,
    # which lazily wires the legacy FallbackLLM. Explicit args override.
    if thesis_writer is None and s.cognition.llm_thesis_enabled:
        from halal_trader.core.llm import create_llm

        _llm = create_llm()
        thesis_writer = LlmThesisWriter(_llm)
        if llm_gate is None:
            llm_gate = LlmGate(_llm)

    # Self-activates once the learning loop (L8) accumulates leakage-free
    # outcomes and calls fit(); identity until then (cold-start safe).
    calibrator = FittedCalibrator(min_samples=s.conviction.min_samples_to_calibrate)
    # Trading-time decay (R-09): stocks use RTH minutes so a Friday-close belief
    # doesn't decay over the weekend (→ Monday mass-exit). Continuous = 24/7 venues
    # or evidence_decay_trading_time=False.
    calendar = (
        RegularHoursCalendar() if s.belief.evidence_decay_trading_time else ContinuousCalendar()
    )
    updater = BeliefUpdater(
        store=store,
        bus=bus,
        clock=clock,
        calendar=calendar,
        regime=EvidenceRegimeClassifier(),
        levels=BarLevelEngine(buffer),
        calibrator=calibrator,
        thesis_writer=thesis_writer or _NoThesis(),
        prices=prices,
        positions=positions or shadow_book,
        llm=llm_gate or _DisabledLLM(),
        config=updater_config,
    )
    # Belief-write sink: single-worker ts-coalescing for the live loop, inline
    # (synchronous) for --once/tests. The worker preserves global write order so
    # the shadow's whole-portfolio recompute can't race (Appendix F).
    worker: CoalescingBeliefWorker | None = None
    sink: BeliefSink
    if coalesce:
        worker = CoalescingBeliefWorker(updater)
        sink = worker
    else:
        sink = InlineBeliefSink(updater)

    # Cheap, always-on interpreters (LLM-free, INV-1). drift wires up
    # conviction_raw's drift down-weight; multiframe/forecaster are flag-gated.
    interpreters: list[Interpreter] = [
        IndicatorInterpreter(buffer),
        RsiInterpreter(buffer),
        TrendAlignmentInterpreter(buffer),
        AnomalyInterpreter(buffer),
        DriftInterpreter(buffer),
        NewsLexiconInterpreter(),
    ]
    if s.cognition.multiframe_enabled:
        interpreters.append(MultiFrameInterpreter(buffer))
    if s.cognition.volume_enabled:
        interpreters.append(VolumeConfirmationInterpreter(buffer))
    if s.cognition.structure_enabled:
        interpreters.append(SupportResistanceInterpreter(buffer))
    if s.cognition.relstrength_enabled:
        interpreters.append(
            RelativeStrengthInterpreter(buffer, benchmark=s.cognition.benchmark_symbol)
        )
    if s.cognition.forecaster_enabled:
        interpreters.append(ForecasterInterpreter(buffer))
    if s.cognition.news_llm_enabled:
        from halabot.cognition.thesis import LlmHeadlineScorer
        from halal_trader.core.llm import create_llm

        interpreters.append(NewsLlmInterpreter(LlmHeadlineScorer(create_llm())))
    router = CognitionRouter(
        bus=bus,
        sink=sink,
        updater=updater,  # gives the router an inline path for bootstrap replay
        buffer=buffer,
        interpreters=interpreters,
    )
    shadow = ShadowPolicyRunner(
        bus=bus,
        store=store,
        policy=Policy(policy_config),
        portfolio=shadow_book,
        risk_engine=BasicRiskEngine(risk_config),
        clock=clock,
        prices=prices,
        history=buffer,  # closes() feed the risk engine's correlation pass
        compliance_ttl=timedelta(hours=s.halal.cache_ttl_h),
        halt_check=_make_halt_check(db_engine),  # operator kill-switch (hb_control)
        # Market-regime gate ("don't fight the tape"): reuses the benchmark bars
        # already fed for relative strength. Inert unless that benchmark is fed.
        benchmark=s.cognition.benchmark_symbol if s.cognition.relstrength_enabled else None,
        market_gate=s.policy.market_gate_enabled and s.cognition.relstrength_enabled,
        market_sma_window=s.policy.market_sma_window,
    )
    # Learning loop (L8): refit the calibrator off closed outcomes every N closes.
    retrainer = CalibratorRetrainer(
        engine=db_engine,
        calibrator=calibrator,
        retrain_every=s.conviction.min_samples_to_calibrate // 2 or 10,
    )
    outcomes = ShadowOutcomeTracker(
        bus=bus,
        engine=db_engine,
        store=store,
        win_threshold_pct=s.conviction.win_threshold_pct,
        on_close=retrainer.on_outcome_closed,
    )
    conviction_writer = ConvictionScoreWriter(bus=bus, engine=db_engine)
    target_writer = TargetWeightWriter(bus=bus, engine=db_engine)
    # Bootstrap warm-start (Appendix F): replay recent observations to warm
    # beliefs BEFORE subscribing to the live stream + starting the worker, so
    # replay completes in isolation and event_id dedup absorbs any overlap.
    if bootstrap and s.belief.bootstrap_window_min > 0:
        now = clock.now()
        since = now - timedelta(minutes=s.belief.bootstrap_window_min)
        await router.bootstrap(since=since, until=now, now=now)
    if worker is not None:
        worker.start()
    router.start()
    shadow.start()
    outcomes.start()
    # Telemetry writers subscribe AFTER bootstrap so replay scores aren't logged.
    conviction_writer.start()
    target_writer.start()
    logger.info(
        "halabot engine assembled (read-only shadow); coalesce=%s, LLM thesis off by default",
        coalesce,
    )
    return Engine(
        bus=bus,
        store=store,
        updater=updater,
        router=router,
        shadow=shadow,
        outcomes=outcomes,
        db_engine=db_engine,
        worker=worker,
        conviction_writer=conviction_writer,
        target_writer=target_writer,
        retrainer=retrainer,
    )
