"""Per-stage instrumentation for the cycle pipeline.

Wave B introduces a lightweight ``CycleStage`` primitive: each named
stage is run inside a context that records its elapsed time, emits a
structured event to the bus, and updates a Prometheus histogram. The
cycle code keeps its existing helper functions; it just wraps each
call in ``async with stage(bus, "name"):``.

This is intentionally minimal — the full "stages as classes that
mutate a CycleState" refactor would balloon to a multi-day rewrite
without changing observable behavior. The seam we land here is the
one that pays off across Waves I (live event stream) and J
(Prometheus histograms).
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
