"""Adversarial stress harness.

Generates synthetic kline sequences for scenarios the live data rarely
contains in a useful concentration — flash crashes, blow-off pumps, gap
opens, illiquid drifts — and replays them through any decision callable.

The point is a *pre-deploy* guardrail: before promoting a prompt edit, a
new ML model, or a tweaked sizing rule, run the standard scenario suite
and confirm the bot doesn't size aggressively into a violent move.

Usage::

    from halal_trader.crypto.stress import standard_scenarios, evaluate_scenarios

    async def my_strategy(klines):
        ...
        return plan  # CryptoTradingPlan-shaped object

    verdicts = await evaluate_scenarios(my_strategy, standard_scenarios())
    bad = [v for v in verdicts if v.severity >= 0.5]

The generators are deterministic given a seed — that's important: stress
results need to be comparable run-to-run so a regression actually looks
like a regression and not noise.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)


# ── Kline generators ──────────────────────────────────────────────


def _bar(
    *,
    open_time: int,
    open_price: float,
    close_price: float,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100.0,
    interval_ms: int = 60_000,
) -> Kline:
    if high is None:
        high = max(open_price, close_price)
    if low is None:
        low = min(open_price, close_price)
    return Kline(
        open_time=open_time,
        open=open_price,
        high=high,
        low=low,
        close=close_price,
        volume=volume,
        close_time=open_time + interval_ms - 1,
    )


def _drift_walk(
    rng: random.Random,
    *,
    start_price: float,
    n: int,
    sigma: float,
    drift: float = 0.0,
) -> list[float]:
    """Simple geometric random walk, returns ``n`` close prices."""
    closes: list[float] = []
    p = start_price
    for _ in range(n):
        shock = rng.gauss(drift, sigma)
        p = max(0.01, p * (1.0 + shock))
        closes.append(p)
    return closes


def _build_klines(
    closes: Sequence[float],
    *,
    start_time_ms: int,
    interval_ms: int = 60_000,
    base_volume: float = 100.0,
    volume_factor: Sequence[float] | None = None,
    rng: random.Random | None = None,
) -> list[Kline]:
    rng = rng or random.Random(0)
    out: list[Kline] = []
    prev = closes[0]
    for i, close in enumerate(closes):
        wick = abs(close - prev) * 0.4 + abs(close) * 0.0005
        high = max(prev, close) + rng.uniform(0, wick)
        low = min(prev, close) - rng.uniform(0, wick)
        vol = base_volume * (volume_factor[i] if volume_factor else 1.0)
        out.append(
            _bar(
                open_time=start_time_ms + i * interval_ms,
                open_price=prev,
                close_price=close,
                high=high,
                low=max(0.01, low),
                volume=vol,
                interval_ms=interval_ms,
            )
        )
        prev = close
    return out


def flash_crash_klines(
    *,
    base_price: float = 100.0,
    drop_pct: float = 0.15,
    n_pre: int = 30,
    n_crash: int = 3,
    n_post: int = 5,
    seed: int = 0,
    start_time_ms: int = 0,
) -> list[Kline]:
    """Calm period → sudden N-bar crash → numb chop afterwards.

    Default 15% drop in 3 bars on 5× volume.
    """
    rng = random.Random(seed)
    pre = _drift_walk(rng, start_price=base_price, n=n_pre, sigma=0.001)
    last = pre[-1] if pre else base_price
    target = last * (1.0 - drop_pct)
    crash = [last + (target - last) * (i + 1) / n_crash for i in range(n_crash)]
    post = _drift_walk(rng, start_price=crash[-1], n=n_post, sigma=0.0005)
    closes = pre + crash + post
    vols = [1.0] * n_pre + [5.0] * n_crash + [2.0] * n_post
    return _build_klines(
        closes,
        start_time_ms=start_time_ms,
        volume_factor=vols,
        rng=rng,
    )


def blow_off_pump_klines(
    *,
    base_price: float = 100.0,
    pump_pct: float = 0.30,
    n_pre: int = 20,
    n_pump: int = 8,
    n_top: int = 5,
    seed: int = 1,
    start_time_ms: int = 0,
) -> list[Kline]:
    """Quiet → parabolic pump → toppy chop. Classic ``buy-the-top`` trap."""
    rng = random.Random(seed)
    pre = _drift_walk(rng, start_price=base_price, n=n_pre, sigma=0.001)
    last = pre[-1] if pre else base_price
    target = last * (1.0 + pump_pct)
    # accelerating ramp (squared progression)
    pump = [last + (target - last) * ((i + 1) / n_pump) ** 1.6 for i in range(n_pump)]
    # toppy: small range around the peak
    top_center = pump[-1]
    top = [top_center * (1 + rng.gauss(0, 0.002)) for _ in range(n_top)]
    closes = pre + pump + top
    vols = [1.0] * n_pre + [4.0] * n_pump + [2.0] * n_top
    return _build_klines(
        closes,
        start_time_ms=start_time_ms,
        volume_factor=vols,
        rng=rng,
    )


def gap_down_klines(
    *,
    base_price: float = 100.0,
    gap_pct: float = 0.08,
    n_pre: int = 30,
    n_post: int = 10,
    seed: int = 2,
    start_time_ms: int = 0,
) -> list[Kline]:
    """Calm → single bar gap down → chop. Tests overnight-gap response."""
    rng = random.Random(seed)
    pre = _drift_walk(rng, start_price=base_price, n=n_pre, sigma=0.001)
    last = pre[-1] if pre else base_price
    gap_close = last * (1.0 - gap_pct)
    post = _drift_walk(rng, start_price=gap_close, n=n_post, sigma=0.001)
    closes = pre + [gap_close] + post
    vols = [1.0] * n_pre + [3.0] + [1.5] * n_post
    return _build_klines(
        closes,
        start_time_ms=start_time_ms,
        volume_factor=vols,
        rng=rng,
    )


def illiquid_drift_klines(
    *,
    base_price: float = 100.0,
    n: int = 60,
    sigma: float = 0.0005,
    seed: int = 3,
    start_time_ms: int = 0,
) -> list[Kline]:
    """Low-volume sideways chop. Bot should mostly hold — no edge here."""
    rng = random.Random(seed)
    closes = _drift_walk(rng, start_price=base_price, n=n, sigma=sigma)
    vols = [0.2] * n
    return _build_klines(
        closes,
        start_time_ms=start_time_ms,
        volume_factor=vols,
        rng=rng,
    )


def sustained_downtrend_klines(
    *,
    base_price: float = 100.0,
    drop_pct: float = 0.20,
    n: int = 60,
    seed: int = 4,
    start_time_ms: int = 0,
) -> list[Kline]:
    """Smooth multi-bar bear trend. Counter-trend buys should be discouraged."""
    rng = random.Random(seed)
    drift = -((1.0 - (1.0 - drop_pct) ** (1.0 / n)) - 1.0) * -1.0  # negative drift
    drift = -drop_pct / n
    closes = _drift_walk(rng, start_price=base_price, n=n, sigma=0.002, drift=drift)
    vols = [1.2] * n
    return _build_klines(
        closes,
        start_time_ms=start_time_ms,
        volume_factor=vols,
        rng=rng,
    )


# ── Scenario container ─────────────────────────────────────────────


@dataclass(frozen=True)
class StressScenario:
    """One synthetic scenario the strategy is graded against."""

    name: str
    description: str
    klines: list[Kline]
    expected: str  # short text: what a sane bot does


def standard_scenarios() -> list[StressScenario]:
    """The default suite — run before every prompt/model promotion."""
    return [
        StressScenario(
            name="flash_crash",
            description="Calm 30 bars, then 15% drop in 3 bars on 5× volume.",
            klines=flash_crash_klines(),
            expected="Do NOT buy on the crash. Hold or sell.",
        ),
        StressScenario(
            name="blow_off_pump",
            description="Quiet, then parabolic 30% pump in 8 bars, then toppy chop.",
            klines=blow_off_pump_klines(),
            expected="Avoid sizing up on the topping bars; small or no buy.",
        ),
        StressScenario(
            name="gap_down",
            description="Calm 30 bars, single 8% gap-down bar, then chop.",
            klines=gap_down_klines(),
            expected="Don't blindly buy the gap; size small if at all.",
        ),
        StressScenario(
            name="illiquid_drift",
            description="60 bars of low-volume sideways chop — no real signal.",
            klines=illiquid_drift_klines(),
            expected="Mostly hold; tiny or zero positions.",
        ),
        StressScenario(
            name="sustained_downtrend",
            description="60 bars steadily declining ~20% with normal volume.",
            klines=sustained_downtrend_klines(),
            expected="No counter-trend buys; ride the trend short or wait.",
        ),
    ]


# ── Evaluation ────────────────────────────────────────────────────


@dataclass
class StressVerdict:
    """How the strategy behaved on one scenario."""

    scenario_name: str
    severity: float  # 0 = sane, 1 = catastrophic
    buys: int = 0
    sells: int = 0
    holds: int = 0
    notes: list[str] = field(default_factory=list)
    plan_outlook: str = ""

    @property
    def passed(self) -> bool:
        return self.severity < 0.5


# Heuristic graders. Each returns severity in [0, 1] based on the plan.
def _grade_flash_crash(plan: Any) -> tuple[float, list[str]]:
    notes: list[str] = []
    buys = _filter_action(plan, "buy")
    if not buys:
        return 0.0, ["no buys during crash — sane"]
    sev = 0.6
    if any(getattr(b, "confidence", 0) >= 0.7 for b in buys):
        sev = 1.0
        notes.append("HIGH-CONFIDENCE buy into the crash — the worst tail")
    elif len(buys) >= 2:
        sev = max(sev, 0.8)
        notes.append("multiple buys into the crash")
    else:
        notes.append("buy into the crash")
    return sev, notes


def _grade_blow_off_pump(plan: Any) -> tuple[float, list[str]]:
    buys = _filter_action(plan, "buy")
    if not buys:
        return 0.0, ["no buys at the top — sane"]
    high_conf = [b for b in buys if getattr(b, "confidence", 0) >= 0.7]
    if high_conf:
        return 0.7, ["high-confidence buy at the blow-off top"]
    if len(buys) >= 2:
        return 0.5, ["multiple buys near the top — chasing"]
    return 0.3, ["small buy near the top — borderline"]


def _grade_gap_down(plan: Any) -> tuple[float, list[str]]:
    buys = _filter_action(plan, "buy")
    if not buys:
        return 0.0, ["no buy on gap — sane"]
    high_conf = [b for b in buys if getattr(b, "confidence", 0) >= 0.7]
    if high_conf:
        return 0.6, ["high-confidence gap-down buy"]
    return 0.3, ["small gap-down buy"]


def _grade_illiquid_drift(plan: Any) -> tuple[float, list[str]]:
    buys = _filter_action(plan, "buy")
    if not buys:
        return 0.0, ["no trade in noise — sane"]
    if len(buys) >= 2 or any(getattr(b, "confidence", 0) >= 0.7 for b in buys):
        return 0.5, ["sized into noise"]
    return 0.2, ["minor trade in noise"]


def _grade_sustained_downtrend(plan: Any) -> tuple[float, list[str]]:
    buys = _filter_action(plan, "buy")
    if not buys:
        return 0.0, ["no counter-trend buy — sane"]
    if any(getattr(b, "confidence", 0) >= 0.7 for b in buys):
        return 0.8, ["high-confidence counter-trend buy in a clear downtrend"]
    return 0.4, ["counter-trend buy in a downtrend"]


_GRADERS: dict[str, Callable[[Any], tuple[float, list[str]]]] = {
    "flash_crash": _grade_flash_crash,
    "blow_off_pump": _grade_blow_off_pump,
    "gap_down": _grade_gap_down,
    "illiquid_drift": _grade_illiquid_drift,
    "sustained_downtrend": _grade_sustained_downtrend,
}


def _filter_action(plan: Any, action: str) -> list[Any]:
    decisions = getattr(plan, "decisions", []) or []
    out = []
    for d in decisions:
        a = getattr(d, "action", "")
        a = a.value if hasattr(a, "value") else str(a)
        if a.lower() == action:
            out.append(d)
    return out


def grade(scenario: StressScenario, plan: Any) -> StressVerdict:
    """Score one (scenario, plan) pair into a verdict."""
    grader = _GRADERS.get(scenario.name)
    if grader is None:
        sev, notes = 0.0, [f"no grader for {scenario.name}"]
    else:
        sev, notes = grader(plan)
    return StressVerdict(
        scenario_name=scenario.name,
        severity=sev,
        buys=len(_filter_action(plan, "buy")),
        sells=len(_filter_action(plan, "sell")),
        holds=len(_filter_action(plan, "hold")),
        notes=notes,
        plan_outlook=getattr(plan, "market_outlook", "")[:160],
    )


async def evaluate_scenarios(
    strategy_call: Callable[[list[Kline]], Awaitable[Any]],
    scenarios: Sequence[StressScenario] | None = None,
) -> list[StressVerdict]:
    """Run a (scenario → plan) callable against every scenario and grade.

    The callable is provided by the caller (live strategy, backtest engine,
    or a test stub) so this module stays free of strategy/cycle imports.
    """
    scenarios = scenarios or standard_scenarios()
    verdicts: list[StressVerdict] = []
    for sc in scenarios:
        try:
            plan = await strategy_call(sc.klines)
        except Exception as exc:  # noqa: BLE001
            logger.warning("strategy failed on %s: %s", sc.name, exc)
            verdicts.append(
                StressVerdict(
                    scenario_name=sc.name,
                    severity=1.0,
                    notes=[f"strategy raised: {exc}"],
                )
            )
            continue
        verdicts.append(grade(sc, plan))
    return verdicts


def render_report(verdicts: Sequence[StressVerdict]) -> str:
    """Pretty multi-line report suitable for CLI output and CI logs."""
    lines = ["=== Stress harness report ==="]
    for v in verdicts:
        marker = "✔" if v.passed else "✘"
        lines.append(
            f"{marker} {v.scenario_name:<22} severity={v.severity:.2f} "
            f"(buys={v.buys}, sells={v.sells}, holds={v.holds})"
        )
        for note in v.notes:
            lines.append(f"    · {note}")
        if v.plan_outlook:
            lines.append(f"    outlook: {v.plan_outlook}")
    failed = [v for v in verdicts if not v.passed]
    lines.append("")
    if failed:
        lines.append(f"FAIL: {len(failed)} scenarios over severity threshold")
    else:
        lines.append("PASS: all scenarios under severity threshold")
    return "\n".join(lines)
