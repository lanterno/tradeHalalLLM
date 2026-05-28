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

from halabot.belief.evidence import ContinuousCalendar
from halabot.belief.schema import BeliefState
from halabot.belief.store import PgBeliefStore
from halabot.belief.updater import BeliefUpdater, UpdaterConfig
from halabot.cognition.bars import BarBuffer, BufferPriceSource
from halabot.cognition.interpreters import (
    AnomalyInterpreter,
    IndicatorInterpreter,
    NewsLexiconInterpreter,
    RsiInterpreter,
    TrendAlignmentInterpreter,
)
from halabot.cognition.level_engine import BarLevelEngine
from halabot.cognition.regime import EvidenceRegimeClassifier
from halabot.cognition.router import CognitionRouter
from halabot.conviction.raw import IdentityCalibrator
from halabot.learning.shadow_outcomes import ShadowOutcomeTracker
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


class _NoPositions:
    def has_position(self, asset: str) -> bool:
        return False


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

    async def stop(self) -> None:
        self.router.stop()
        self.shadow.stop()
        self.outcomes.stop()
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

    updater = BeliefUpdater(
        store=store,
        bus=bus,
        clock=clock,
        calendar=ContinuousCalendar(),
        regime=EvidenceRegimeClassifier(),
        levels=BarLevelEngine(buffer),
        calibrator=IdentityCalibrator(),
        thesis_writer=thesis_writer or _NoThesis(),
        prices=prices,
        positions=positions or _NoPositions(),
        llm=llm_gate or _DisabledLLM(),
        config=updater_config,
    )
    router = CognitionRouter(
        bus=bus,
        updater=updater,
        buffer=buffer,
        interpreters=[
            IndicatorInterpreter(buffer),
            RsiInterpreter(buffer),
            TrendAlignmentInterpreter(buffer),
            AnomalyInterpreter(buffer),
            NewsLexiconInterpreter(),
        ],
    )
    shadow = ShadowPolicyRunner(
        bus=bus,
        store=store,
        policy=Policy(policy_config),
        portfolio=ShadowPortfolio(),
        risk_engine=BasicRiskEngine(risk_config),
        clock=clock,
        prices=prices,
    )
    outcomes = ShadowOutcomeTracker(bus=bus, engine=db_engine, store=store)
    router.start()
    shadow.start()
    outcomes.start()
    logger.info("halabot engine assembled (read-only shadow); LLM thesis off by default")
    return Engine(
        bus=bus,
        store=store,
        updater=updater,
        router=router,
        shadow=shadow,
        outcomes=outcomes,
        db_engine=db_engine,
    )
