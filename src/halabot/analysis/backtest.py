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
    ForecasterInterpreter,
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
from halabot.cognition.structure import sma_trend_state, structural_label
from halabot.conviction.raw import IdentityCalibrator
from halabot.execution.position_manager import HoldContext, decide_exit
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
    regime: str = "unknown"  # entry regime, for per-regime P&L segmentation
    structural: str = "unknown"  # entry price-structure label (rank 5)
    market: str = "unknown"  # entry market-regime label (benchmark vs its SMA)


@dataclass
class RegimeStats:
    """Per-entry-regime closed-trade P&L (the regime-attribution view)."""

    regime: str
    n: int = 0
    win_rate: float | None = None
    avg_return_pct: float | None = None
    profit_factor: float | None = None
    total_return: float = 0.0  # Σ return_pct × closed_weight for this regime

    def line(self) -> str:
        wr = f"{self.win_rate:.0%}" if self.win_rate is not None else "n/a"
        avg = f"{self.avg_return_pct:+.2%}" if self.avg_return_pct is not None else "n/a"
        pf = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "n/a"
        return (
            f"{self.regime:<14} n={self.n:<3} win={wr:<4} avg={avg:<8} "
            f"profit_factor={pf:<5} total={self.total_return:+.4f}"
        )


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
    by_regime: list[RegimeStats] = field(default_factory=list)  # entry-regime split
    by_structure: list[RegimeStats] = field(default_factory=list)  # price-structure split
    by_market: list[RegimeStats] = field(default_factory=list)  # market-regime split

    def summary(self) -> str:
        wr = f"{self.win_rate:.0%}" if self.win_rate is not None else "n/a"
        avg = f"{self.avg_return_pct:+.2%}" if self.avg_return_pct is not None else "n/a"
        pf = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "n/a"
        return (
            f"proposals={self.proposals} closed={self.closed} win={wr} avg={avg} "
            f"profit_factor={pf} total={self.total_return:+.4f} max_dd={self.max_drawdown:.2%}"
        )

    def regime_summary(self) -> str:
        """Multi-line per-entry-regime breakdown (most-traded regime first)."""
        if not self.by_regime:
            return "  (no closed trades)"
        return "\n".join("  " + r.line() for r in self.by_regime)

    def structure_summary(self) -> str:
        """Multi-line per-entry price-structure breakdown (rank 5)."""
        if not self.by_structure:
            return "  (no closed trades)"
        return "\n".join("  " + r.line() for r in self.by_structure)

    def market_summary(self) -> str:
        """Multi-line per-entry market-regime breakdown (benchmark vs its SMA)."""
        if not self.by_market:
            return "  (no closed trades)"
        return "\n".join("  " + r.line() for r in self.by_market)


class _Book:
    """In-memory hypothetical book: marks shadow proposals to their decision price,
    records realized returns on reductions, and tracks a realized-equity curve.
    Mirrors ShadowOutcomeTracker's VWAP logic without touching the DB."""

    def __init__(
        self,
        *,
        win_threshold_pct: float,
        prices: BufferPriceSource,
        cost_frac: float = 0.0,
        exit_ladder: bool = False,
        trailing_pct: float = 0.05,
        buffer: BarBuffer | None = None,
        structure_window: int = 20,
        structure_er_trend: float = 0.5,
        benchmark: str | None = None,
        market_sma_window: int = 50,
    ) -> None:
        self._win = win_threshold_pct
        self._prices = prices
        # Shared bar buffer (the engine's own) so the book can read price
        # GEOMETRY at entry for the structural-regime tag (rank 5) — a label
        # independent of the conviction evidence, used only to MEASURE whether
        # structure discriminates P&L (it does not feed conviction here).
        self._buffer = buffer
        self._structure_window = structure_window
        self._structure_er_trend = structure_er_trend
        # Market-wide regime read off the benchmark (e.g. SPY vs its SMA) — a
        # single global risk-on/off label, non-circular w.r.t. any one asset.
        self._benchmark = benchmark
        self._market_sma_window = market_sma_window
        # One-way transaction cost (slippage + commission) as a fraction of
        # notional: a buy fills slightly HIGHER, a sell slightly LOWER, so each
        # round-trip pays ~2× — which is exactly what penalizes churn honestly.
        self._cost = cost_frac
        # Optional Appendix-H slow-out exits (trend-break + trailing stop) via the
        # dormant decide_exit, evaluated per bar — measures the slow-out offline.
        # EMPIRICAL VERDICT (2026-05-29, controlled same-bars A/B, 15d + 30d, 1H,
        # 5bps): the ladder is flat-to-WORSE than the conviction-decay-only exit
        # at every trailing distance (and trend-break alone) — on 30d it cut PF
        # 1.33→1.07, total +0.99%→+0.71%, AND raised drawdown 0.83%→0.98%. The
        # conviction-decay path (policy target→0 as belief fades) IS the slow-out;
        # a trailing/trend stop just exits winners early and pays extra cost. So
        # this ships DEFAULT-OFF — kept as a measurable knob for other timeframes.
        self._ladder = exit_ladder
        self._trailing_pct = trailing_pct
        self._trail_high: dict[str, float] = {}
        self._trail_stop: dict[str, float] = {}
        self._pos: dict[str, _Pos] = {}
        self.returns: list[float] = []  # per-closed-trade NET return_pct (after costs)
        self.weights: list[float] = []  # closed_weight, parallel to returns
        self.regimes: list[str] = []  # entry regime, parallel to returns
        self.structurals: list[str] = []  # entry structural label, parallel to returns
        self.markets: list[str] = []  # entry market-regime label, parallel to returns
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
                self._pos[asset] = _Pos(
                    weight=delta, vwap=fill, open_ts=ts,
                    regime=str(p.get("regime", "unknown")),
                    structural=self._structural_at_entry(asset),
                    market=self._market_at_entry(),
                )
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
        self.regimes.append(pos.regime)
        self.structurals.append(pos.structural)
        self.markets.append(pos.market)
        self._cum += ret * closed
        self._peak = max(self._peak, self._cum)
        self.max_dd = max(self.max_dd, self._peak - self._cum)
        pos.weight -= closed
        if pos.weight <= _EPS:
            self._pos.pop(asset, None)
            self._trail_high.pop(asset, None)
            self._trail_stop.pop(asset, None)

    def tick(self, asset: str, price: float, sma: float | None) -> None:
        """Per-bar slow-out evaluation (Appendix-H rungs 4–6 via decide_exit):
        ratchet a trailing stop and exit on a trend-break or a trailing-stop hit.
        No-op unless the exit ladder is enabled and the asset is held."""
        if not self._ladder or price <= 0:
            return
        pos = self._pos.get(asset)
        if pos is None or pos.weight <= _EPS:
            return
        high = max(self._trail_high.get(asset, pos.vwap), price)
        self._trail_high[asset] = high
        ctx = HoldContext(
            asset=asset,
            price=price,
            stop=self._trail_stop.get(asset),
            sma=sma,
            is_winner=price > pos.vwap,
            trailing_high=high,
            trailing_pct=self._trailing_pct,
            target_weight=1.0,  # conviction-decay (rung 7) stays on the proposal path
        )
        decision = decide_exit(ctx)
        if decision.action == "tighten" and decision.new_stop is not None:
            self._trail_stop[asset] = decision.new_stop
        elif decision.action == "exit":
            self._realize(asset, pos, pos.weight, price)

    def finalize(self) -> None:
        """Mark any still-open positions to their last price (close the book)."""
        for asset, pos in list(self._pos.items()):
            last = self._prices.last_price(asset)
            if last is not None and last > 0:
                self._realize(asset, pos, pos.weight, last)

    def _structural_at_entry(self, asset: str) -> str:
        """Price-structure label from the shared buffer at the entry bar."""
        if self._buffer is None:
            return "unknown"
        return structural_label(
            self._buffer.highs(asset),
            self._buffer.lows(asset),
            self._buffer.closes(asset),
            window=self._structure_window,
            er_trend=self._structure_er_trend,
        )

    def _market_at_entry(self) -> str:
        """Market-regime label (benchmark vs its SMA) at the entry bar:
        risk_on (above) / risk_off (below) / unknown."""
        if self._buffer is None or self._benchmark is None:
            return "unknown"
        state = sma_trend_state(self._buffer.closes(self._benchmark), self._market_sma_window)
        return {"above": "risk_on", "below": "risk_off"}.get(state, "unknown")

    def _bucket_stats(self, keys: list[str]) -> list[RegimeStats]:
        """Group closed trades by a per-trade key (regime / structure), most-traded
        first. ``keys`` is parallel to ``returns``/``weights``."""
        buckets: dict[str, list[tuple[float, float]]] = {}
        for r, w, key in zip(self.returns, self.weights, keys):
            buckets.setdefault(key, []).append((r, w))
        out: list[RegimeStats] = []
        for key, rows in buckets.items():
            rets = [r for r, _ in rows]
            n = len(rets)
            gw = sum(r for r in rets if r > 0)
            gl = -sum(r for r in rets if r < 0)
            out.append(
                RegimeStats(
                    regime=key,
                    n=n,
                    win_rate=(sum(1 for r in rets if r > self._win) / n) if n else None,
                    avg_return_pct=(sum(rets) / n) if n else None,
                    profit_factor=(gw / gl) if gl > _EPS else None,
                    total_return=sum(r * w for r, w in rows),
                )
            )
        out.sort(key=lambda s: s.n, reverse=True)
        return out

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
            by_regime=self._bucket_stats(self.regimes),
            by_structure=self._bucket_stats(self.structurals),
            by_market=self._bucket_stats(self.markets),
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
        exit_ladder: bool = False,
        trailing_pct: float = 0.05,
        sma_window: int = 20,
        structure_window: int = 20,
        structure_er_trend: float = 0.5,
        market_gate: bool = False,
        market_sma_window: int = 50,
        forecaster: str = "",  # "" | "ols" | "chronos" — append to the default stack
        chronos_model: str = "amazon/chronos-bolt-small",
        interpreters: list[Interpreter] | None = None,
    ) -> None:
        self._policy_config = policy_config or PolicyConfig()
        self._updater_config = updater_config or UpdaterConfig(llm_thesis_enabled=False)
        self._risk_config = risk_config or RiskConfig()
        self._cost_frac = max(0.0, cost_bps) / 10_000.0  # one-way cost as a fraction
        self._trading_hours = trading_hours
        self._win = win_threshold_pct
        self._exit_ladder = exit_ladder
        self._trailing_pct = trailing_pct
        self._sma_window = sma_window
        self._structure_window = structure_window
        self._structure_er_trend = structure_er_trend
        self._market_gate = market_gate
        self._market_sma_window = market_sma_window
        self._forecaster = forecaster
        self._chronos_model = chronos_model
        self._interpreters = interpreters

    def _build_forecaster(self, buffer: BarBuffer) -> Interpreter:
        """Construct the configured forecaster interpreter (both emit
        source="forecaster", so at most one is active)."""
        if self._forecaster == "chronos":
            from halabot.cognition.chronos_forecaster import (
                ChronosForecasterInterpreter,
                load_chronos_pipeline,
            )

            return ChronosForecasterInterpreter(buffer, load_chronos_pipeline(self._chronos_model))
        return ForecasterInterpreter(buffer)  # "ols" (default for any non-chronos value)

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
        if self._forecaster and self._interpreters is None:
            interpreters.append(self._build_forecaster(buffer))
        router = CognitionRouter(bus=bus, updater=updater, buffer=buffer, interpreters=interpreters)
        shadow = ShadowPolicyRunner(
            bus=bus, store=store, policy=Policy(self._policy_config),
            portfolio=ShadowPortfolio(), risk_engine=BasicRiskEngine(self._risk_config),
            clock=clock, prices=prices, history=buffer,
            benchmark=benchmark, market_gate=self._market_gate,
            market_sma_window=self._market_sma_window,
        )
        book = _Book(
            win_threshold_pct=self._win, prices=prices, cost_frac=self._cost_frac,
            exit_ladder=self._exit_ladder, trailing_pct=self._trailing_pct,
            buffer=buffer, structure_window=self._structure_window,
            structure_er_trend=self._structure_er_trend, benchmark=benchmark,
        )
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
            # Slow-out exit pass for this bar's asset (after its belief updated).
            if self._exit_ladder and sym != benchmark:
                closes = buffer.closes(sym)
                sma = (
                    sum(closes[-self._sma_window:]) / min(len(closes), self._sma_window)
                    if closes
                    else None
                )
                book.tick(sym, bar.c, sma)

        book.finalize()
        router.stop()
        shadow.stop()
        return book.result()
