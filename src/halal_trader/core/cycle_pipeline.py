"""Wave B cycle-pipeline primitives.

Three building blocks the crypto and stock cycles share:

* :class:`CycleState` — dataclass carrier with one field per
  prompt-context block. Each cycle builds a fresh instance, runs it
  through a stage list, then reads every text field off the state to
  assemble ``analyze_kwargs``.
* :class:`StageOutcome` — captures one stage's elapsed time + error
  state, even when the stage raised; collected for the per-cycle
  replay snapshot.
* :func:`run_stages` — driver that walks a stage list against a
  ``CycleState``, wraps each stage in :func:`stage` instrumentation
  (Prometheus histogram + ``/ws/cycle`` start/end events), and
  optionally short-circuits when ``state.halt`` is set.

The :func:`stage` async context manager is also exported standalone
for the few inline cycle blocks that haven't been promoted to stage
classes yet (e.g. the crypto cycle's `strategy_analyze` block, which
must propagate exceptions rather than swallow them).
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from halal_trader.core.cycle_stages import CycleStage  # noqa: F401
    from halal_trader.core.event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class StageOutcome:
    """Result of one stage execution — captured even when stage raised."""

    name: str
    elapsed_ms: float
    error: str | None = None
    skipped: bool = False
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class CycleState:
    """Per-cycle data carrier — the carrier the Wave B stage list mutates.

    Both ``CryptoCycleService._run_cycle_impl`` and
    ``TradingCycleService._run_cycle_impl`` build a fresh ``CycleState``
    each cycle, run a list of :class:`CycleStage` instances against it
    via :func:`run_stages`, then read every prompt-context field off
    the state to assemble ``analyze_kwargs`` for the LLM call. Each
    stage takes a state, mutates one or two fields, returns it.

    The dataclass is intentionally permissive (every field has a
    default) so partial states are valid mid-pipeline; a stage that
    needs ``indicators_cache`` early-returns when the cache is empty
    rather than raising. Stage exceptions are swallowed by the
    instrumentation context (see :func:`stage`) so a regional failure
    leaves the field at its default — the cycle then proceeds with an
    empty block in the LLM prompt rather than crashing.
    """

    cycle_id: str = ""

    # ── Inputs gathered up front ───────────────────────────────
    account: Any = None
    halal_pairs: list[str] = field(default_factory=list)
    open_positions: list[Any] = field(default_factory=list)
    today_pnl: float = 0.0

    # ── Market data (per cycle) ────────────────────────────────
    klines_by_symbol: dict[str, list[Any]] = field(default_factory=dict)
    indicators_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    orderbooks: dict[str, dict[str, Any]] = field(default_factory=dict)
    snapshots: dict[str, Any] = field(default_factory=dict)  # stocks-side
    bars: dict[str, Any] = field(default_factory=dict)  # stocks-side
    current_prices: dict[str, float] = field(default_factory=dict)  # crypto WS prices

    # ── Prompt-context blocks (each owned by one stage) ────────
    risk_text: str = ""
    regime_text: str = ""
    sentiment_text: str = ""
    timeframe_text: str = ""
    ml_signals_text: str = ""
    forecasts_text: str = ""  # Chronos / price-forecast text — seed for ml-signals
    microstructure_text: str = ""
    news_text: str = ""
    catalysts_text: str = ""
    performance_text: str = ""
    exchange_rules_text: str = ""
    active_adjustments: str = ""
    # Wave G: predicted slippage per halal pair surfaced to the LLM so
    # the strategy can reason about expected execution cost (e.g.
    # downgrade a low-edge buy when slippage is high).
    slippage_text: str = ""

    # ── Outputs ────────────────────────────────────────────────
    plan: Any = None
    halt: bool = False  # if any earlier stage decided the cycle should not trade
    # Structured risk state from the risk stage — kept alongside
    # ``risk_text`` so the dashboard's /api/risk/state can render the
    # raw fields, not just the formatted prompt block.
    risk_state: Any = None

    # Free-form per-stage timing / extras for the replay snapshot.
    stage_outcomes: list[StageOutcome] = field(default_factory=list)


async def run_stages(
    state: "CycleState",
    stages: "list[Any]",
    *,
    bus: "EventBus | None" = None,
    stop_on_halt: bool = False,
) -> "CycleState":
    """Drive a stage list against ``state``, wrapping each in instrumentation.

    Each :class:`CycleStage` runs inside the existing ``stage(bus, name)``
    context manager so its name lands in the Prometheus latency
    histogram and on the live ``/ws/cycle`` event stream automatically.

    Stages mutate ``state`` in place; the function returns the same
    object for convenient chaining. Stage exceptions are swallowed by
    default (matching the cycle's "best-effort augmentation" semantics)
    so a regional failure can't take the cycle down.

    When ``stop_on_halt=True``, the driver breaks out of the loop the
    first time a stage sets ``state.halt = True``. This lets risk
    stages short-circuit downstream augmentation without the caller
    having to split the stage list around the halt check.
    """
    for s in stages:
        async with stage(bus, s.name):
            await s.run(state)
        if stop_on_halt and state.halt:
            break
    return state


@contextlib.asynccontextmanager
async def stage(
    bus: "EventBus | None",
    name: str,
    *,
    swallow: bool = True,
    **attrs: object,
) -> AsyncIterator[StageOutcome]:
    """Run a cycle stage with bus + log + histogram instrumentation.

    Usage::

        async with stage(bus, "fetch_klines", pair_count=len(pairs)) as o:
            klines = await broker.fetch(pairs)
            o.extra["n_klines"] = sum(len(k) for k in klines.values())

    On entry, publishes ``cycle.stage.start``; on exit ``cycle.stage.end``
    (with ``error`` populated when the body raised). When ``swallow=True``
    (default), the stage body's exception is logged but not re-raised —
    matching the cycle's "best-effort augmentation" semantics. Mark
    ``swallow=False`` for stages that *must* succeed (the LLM call, the
    executor) so a real failure halts the cycle.

    The Prometheus histogram (``halal_trader_stage_latency_ms``) is
    updated unconditionally so even errored stages count in p95.
    """
    outcome = StageOutcome(name=name, elapsed_ms=0.0)
    t0 = time.monotonic()
    if bus is not None:
        try:
            await bus.publish("cycle.stage.start", {"name": name, **attrs})
        except Exception:  # noqa: BLE001
            pass
    try:
        yield outcome
    except Exception as exc:  # noqa: BLE001
        outcome.error = repr(exc)
        if not swallow:
            raise
        logger.debug("cycle stage %r failed (swallowed): %s", name, exc)
    finally:
        outcome.elapsed_ms = (time.monotonic() - t0) * 1000.0
        # Histogram (Wave J wires the actual prometheus_client metric).
        try:
            from halal_trader.core.metrics import observe_stage_latency

            observe_stage_latency(name=name, ms=outcome.elapsed_ms, error=outcome.error)
        except Exception:  # noqa: BLE001
            pass
        if bus is not None:
            try:
                await bus.publish(
                    "cycle.stage.end",
                    {
                        "name": name,
                        "elapsed_ms": outcome.elapsed_ms,
                        "error": outcome.error,
                        **outcome.extra,
                        **attrs,
                    },
                )
            except Exception:  # noqa: BLE001
                pass
