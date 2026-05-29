"""Backtest harness (Direction A) — measure the engine on historical bars.

Replays a historical bar series through the REAL cognition→belief→policy pipeline
with a deterministic event-time clock, and scores the shadow proposals' hypothetical
P&L. This is how we tune the entry bands / calibration / interpreters in minutes
instead of waiting weeks of live shadow — and how we'll know whether the
understanding engine actually has an edge before flipping anything live.

Runs fully IN-MEMORY (InMemory store/log, no Postgres, no MCP), so a config sweep
is fast and side-effect-free. The pipeline objects are the same ones build_engine
wires for the live shadow, so a backtest result reflects the real engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from halabot.belief.evidence import ContinuousCalendar, RegularHoursCalendar
from halabot.belief.schema import BeliefState
from halabot.belief.store import InMemoryBeliefStore
from halabot.belief.updater import BeliefUpdater, UpdaterConfig
from halabot.cognition.bars import Bar, BarBuffer, BufferPriceSource
from halabot.cognition.base import Interpreter
from halabot.cognition.interpreters import (
    AnomalyInterpreter,
    DriftInterpreter,
    IndicatorInterpreter,
    MultiFrameInterpreter,
    NewsLexiconInterpreter,
    RelativeStrengthInterpreter,
    RsiInterpreter,
    SupportResistanceInterpreter,
    TrendAlignmentInterpreter,
    VolumeConfirmationInterpreter,
)
from halabot.cognition.level_engine import BarLevelEngine
from halabot.cognition.regime import EvidenceRegimeClassifier
from halabot.cognition.router import CognitionRouter
from halabot.conviction.raw import IdentityCalibrator
from halabot.platform.bus import InProcessEventBus
from halabot.platform.clock import FakeClock
from halabot.platform.event_log import InMemoryEventLog
from halabot.platform.events import Event, EventType, new_event
from halabot.policy.policy import Policy
from halabot.policy.portfolio import ShadowPortfolio
from halabot.policy.shadow import ShadowPolicyRunner
from halabot.policy.sizing import PolicyConfig
from halabot.risk.engine import BasicRiskEngine, RiskConfig

logger = logging.getLogger(__name__)
_EPS = 1e-9


class _NoThesis:
    async def write(self, belief: BeliefState) -> str:
        return ""


class _OffLLM:
    def available(self) -> bool:
        return False

    def breaker_open(self) -> bool:
        return True


@dataclass
class _Pos:
    weight: float
    vwap: float
    open_ts: datetime


@dataclass
class BacktestResult:
    proposals: int = 0
    closed: int = 0
    win_rate: float | None = None
    avg_return_pct: float | None = None
    profit_factor: float | None = None  # gross win / gross loss
    total_return: float = 0.0  # Σ return_pct × closed_weight (book-weighted)
    max_drawdown: float = 0.0  # peak-to-trough of the realized-equity curve
    returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        wr = f"{self.win_rate:.0%}" if self.win_rate is not None else "n/a"
        avg = f"{self.avg_return_pct:+.2%}" if self.avg_return_pct is not None else "n/a"
        pf = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "n/a"
        return (
            f"proposals={self.proposals} closed={self.closed} win={wr} avg={avg} "
            f"profit_factor={pf} total={self.total_return:+.4f} max_dd={self.max_drawdown:.2%}"
        )


class _Book:
    """In-memory hypothetical book: marks shadow proposals to their decision price,
    records realized returns on reductions, and tracks a realized-equity curve.
    Mirrors ShadowOutcomeTracker's VWAP logic without touching the DB."""

    def __init__(
        self, *, win_threshold_pct: float, prices: BufferPriceSource, cost_frac: float = 0.0
    ) -> None:
        self._win = win_threshold_pct
        self._prices = prices
        # One-way transaction cost (slippage + commission) as a fraction of
        # notional: a buy fills slightly HIGHER, a sell slightly LOWER, so each
        # round-trip pays ~2× — which is exactly what penalizes churn honestly.
        self._cost = cost_frac
        self._pos: dict[str, _Pos] = {}
        self.returns: list[float] = []  # per-closed-trade NET return_pct (after costs)
        self.weights: list[float] = []  # closed_weight, parallel to returns
        self.proposals = 0
        self._cum = 0.0
        self._peak = 0.0
        self.max_dd = 0.0

    async def on_proposal(self, event: Event) -> None:
        self.proposals += 1
        asset = event.asset
        p = event.payload
        price = p.get("price")
        if asset is None or not price or price <= 0:
            return
        delta = float(p.get("weight_delta", 0.0))
        ts = event.ts
        pos = self._pos.get(asset)
        if delta > 0:  # open / add → fill worse by the cost, blend VWAP
            fill = price * (1.0 + self._cost)
            if pos is None or pos.weight <= _EPS:
                self._pos[asset] = _Pos(weight=delta, vwap=fill, open_ts=ts)
            else:
                total = pos.weight + delta
                pos.vwap = (pos.vwap * pos.weight + fill * delta) / total
                pos.weight = total
            return
        if pos is None or pos.weight <= _EPS:  # reduce with nothing held
            return
        self._realize(asset, pos, abs(delta), price)

    def _realize(self, asset: str, pos: _Pos, qty: float, price: float) -> None:
        closed = min(qty, pos.weight)
        exit_fill = price * (1.0 - self._cost)  # sell fills worse (lower) by the cost
        ret = (exit_fill - pos.vwap) / pos.vwap if pos.vwap > 0 else 0.0
        self.returns.append(ret)
        self.weights.append(closed)
        self._cum += ret * closed
        self._peak = max(self._peak, self._cum)
        self.max_dd = max(self.max_dd, self._peak - self._cum)
        pos.weight -= closed
        if pos.weight <= _EPS:
            self._pos.pop(asset, None)

    def finalize(self) -> None:
        """Mark any still-open positions to their last price (close the book)."""
        for asset, pos in list(self._pos.items()):
            last = self._prices.last_price(asset)
            if last is not None and last > 0:
                self._realize(asset, pos, pos.weight, last)

    def result(self) -> BacktestResult:
        n = len(self.returns)
        wins = [r for r in self.returns if r > self._win]
        gross_win = sum(r for r in self.returns if r > 0)
        gross_loss = -sum(r for r in self.returns if r < 0)
        return BacktestResult(
            proposals=self.proposals,
            closed=n,
            win_rate=(len(wins) / n) if n else None,
            avg_return_pct=(sum(self.returns) / n) if n else None,
            profit_factor=(gross_win / gross_loss) if gross_loss > _EPS else None,
            total_return=sum(r * w for r, w in zip(self.returns, self.weights)),
            max_drawdown=self.max_dd,
            returns=list(self.returns),
        )


class Backtester:
    """Composes the in-memory pipeline and replays bars deterministically."""

    def __init__(
        self,
        *,
        policy_config: PolicyConfig | None = None,
        updater_config: UpdaterConfig | None = None,
        risk_config: RiskConfig | None = None,
        trading_hours: bool = True,
        win_threshold_pct: float = 0.002,
        cost_bps: float = 5.0,
        interpreters: list[Interpreter] | None = None,
    ) -> None:
        self._policy_config = policy_config or PolicyConfig()
        self._updater_config = updater_config or UpdaterConfig(llm_thesis_enabled=False)
        self._risk_config = risk_config or RiskConfig()
        self._cost_frac = max(0.0, cost_bps) / 10_000.0  # one-way cost as a fraction
        self._trading_hours = trading_hours
        self._win = win_threshold_pct
        self._interpreters = interpreters

    async def run(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        *,
        start: datetime | None = None,
        benchmark: str | None = None,
        relstrength: bool = True,
    ) -> BacktestResult:
        # Flatten to a single chronological stream across symbols (event-time replay).
        stream: list[tuple[datetime, str, Bar]] = sorted(
            ((b.ts, sym, b) for sym, bars in bars_by_symbol.items() for b in bars),
            key=lambda x: x[0],
        )
        if not stream:
            return BacktestResult()
        clock = FakeClock(start or stream[0][0])
        log = InMemoryEventLog()
        bus = InProcessEventBus(log)
        store = InMemoryBeliefStore()
        buffer = BarBuffer()
        prices = BufferPriceSource(buffer)
        calendar = RegularHoursCalendar() if self._trading_hours else ContinuousCalendar()

        updater = BeliefUpdater(
            store=store, bus=bus, clock=clock, calendar=calendar,
            regime=EvidenceRegimeClassifier(), levels=BarLevelEngine(buffer),
            calibrator=IdentityCalibrator(), thesis_writer=_NoThesis(),
            prices=prices, positions=ShadowPortfolio(), llm=_OffLLM(),
            config=self._updater_config,
        )
        interpreters = self._interpreters or [
            IndicatorInterpreter(buffer), RsiInterpreter(buffer),
            TrendAlignmentInterpreter(buffer), AnomalyInterpreter(buffer),
            DriftInterpreter(buffer), MultiFrameInterpreter(buffer),
            VolumeConfirmationInterpreter(buffer), SupportResistanceInterpreter(buffer),
            NewsLexiconInterpreter(),
        ]
        if benchmark and relstrength and self._interpreters is None:
            interpreters.append(RelativeStrengthInterpreter(buffer, benchmark=benchmark))
        router = CognitionRouter(bus=bus, updater=updater, buffer=buffer, interpreters=interpreters)
        shadow = ShadowPolicyRunner(
            bus=bus, store=store, policy=Policy(self._policy_config),
            portfolio=ShadowPortfolio(), risk_engine=BasicRiskEngine(self._risk_config),
            clock=clock, prices=prices, history=buffer,
        )
        book = _Book(win_threshold_pct=self._win, prices=prices, cost_frac=self._cost_frac)
        bus.subscribe({EventType.POLICY_TRADE_PROPOSED}, book.on_proposal)
        router.start()
        shadow.start()

        # Seed compliance halal for the universe (membership = halal, as in the
        # shadow). The benchmark is fed for relative-strength but NEVER traded, so
        # it gets no verdict (the halal gate blocks any benchmark buy).
        for sym in bars_by_symbol:
            if sym == benchmark:
                continue
            await bus.publish(
                new_event(clock, EventType.COMPLIANCE_VERDICT, source="backtest", asset=sym,
                          payload={"status": "halal", "transient_error": False})
            )
        # Replay bars in event-time order.
        for ts, sym, bar in stream:
            clock.set(ts)
            await bus.publish(
                new_event(clock, EventType.OBSERVATION_BAR, source="backtest", asset=sym,
                          payload={"o": bar.o, "h": bar.h, "low": bar.low, "c": bar.c, "v": bar.v})
            )

        book.finalize()
        router.stop()
        shadow.stop()
        return book.result()
