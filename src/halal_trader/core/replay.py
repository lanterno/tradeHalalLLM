"""Bit-perfect cycle replay.

To debug a bad trade or honestly score a prompt change, you need to be
able to recreate the *exact* inputs the bot saw on that cycle and re-run
its decision pipeline against them. Without that, "this prompt is
better" reduces to handwaving over noisy live results.

This module is the seam:

* :class:`CycleSnapshot` — the input bundle (klines, account, sentiment,
  positions, indicators, regime, etc.) for one cycle. Pure data, JSON
  serializable, versioned by ``schema_version``.
* :class:`ReplayStore` — write/read cycle snapshots from a directory of
  JSON files keyed by ``cycle_id``. Wraps the on-disk format so callers
  can swap to sqlite / blob storage later.
* :func:`replay_cycle` — given a cycle_id and a "decision callable", load
  the snapshot, call the callable on the inputs, and return its plan.

The cycle code captures snapshots once via :func:`record_snapshot` (no
behavior change to the cycle besides one IO write per cycle). The
replay harness reconstructs ``CycleSnapshot`` objects exactly — the LLM
output won't be bit-identical because LLMs aren't deterministic, but
*everything the LLM saw* is.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from halal_trader.domain.models import Kline

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1


# ── Snapshot ──────────────────────────────────────────────────────


@dataclass
class CycleSnapshot:
    """Frozen inputs of one cycle. Pass back through ``replay_cycle``."""

    cycle_id: str
    cycle_started_at: str  # ISO timestamp
    market: str = "crypto"  # "crypto" | "stocks"
    schema_version: int = SCHEMA_VERSION

    halal_pairs: list[str] = field(default_factory=list)
    klines_by_symbol: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    indicators_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    orderbooks: dict[str, dict[str, Any]] = field(default_factory=dict)

    account: dict[str, Any] = field(default_factory=dict)
    positions_text: str = ""
    open_position_count: int = 0
    today_pnl: float = 0.0

    sentiment_text: str = ""
    timeframe_text: str = ""
    ml_signals_text: str = ""
    regime_text: str = ""
    risk_text: str = ""
    microstructure_text: str = ""
    news_text: str = ""
    performance_text: str = ""
    exchange_rules_text: str = ""
    active_adjustments: str = ""

    extra: dict[str, Any] = field(default_factory=dict)

    # ── helpers ──────────────────────────────────────────────────

    @classmethod
    def from_inputs(
        cls,
        *,
        cycle_id: str,
        klines_by_symbol: dict[str, list[Kline]] | None = None,
        **kwargs: Any,
    ) -> "CycleSnapshot":
        """Build from native inputs — flattens Pydantic ``Kline`` objects."""
        flat: dict[str, list[dict[str, Any]]] = {}
        for sym, ks in (klines_by_symbol or {}).items():
            flat[sym] = [_kline_to_dict(k) for k in ks]
        return cls(
            cycle_id=cycle_id,
            cycle_started_at=datetime.now(UTC).isoformat(),
            klines_by_symbol=flat,
            **kwargs,
        )

    def klines_native(self) -> dict[str, list[Kline]]:
        """Materialise the flat klines back into ``Kline`` objects."""
        out: dict[str, list[Kline]] = {}
        for sym, rows in self.klines_by_symbol.items():
            out[sym] = [Kline(**r) for r in rows]
        return out


def _kline_to_dict(k: Kline | dict[str, Any]) -> dict[str, Any]:
    if isinstance(k, dict):
        return dict(k)
    if hasattr(k, "model_dump"):
        return k.model_dump()
    if hasattr(k, "dict"):
        return k.dict()  # pydantic v1 fallback
    return asdict(k)


# ── On-disk store ─────────────────────────────────────────────────


@dataclass
class ReplayStore:
    """JSON-per-cycle directory store. Simple, ops-friendly, swappable."""

    root: Path
    max_keep: int = 5_000

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cycle_id: str) -> Path:
        # cycle_id is "cycle-XXXXXXXX" — sanitise just in case
        safe = "".join(c for c in cycle_id if c.isalnum() or c in "-_")
        return self.root / f"{safe}.json"

    def write(self, snapshot: CycleSnapshot) -> Path:
        path = self._path(snapshot.cycle_id)
        path.write_text(json.dumps(asdict(snapshot), default=_json_default, indent=2))
        self._gc()
        return path

    def read(self, cycle_id: str) -> CycleSnapshot:
        path = self._path(cycle_id)
        raw = json.loads(path.read_text())
        version = int(raw.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"snapshot {cycle_id} schema_version {version} > supported {SCHEMA_VERSION}"
            )
        return CycleSnapshot(**raw)

    def list_cycle_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def _gc(self) -> None:
        if self.max_keep <= 0:
            return
        files = sorted(
            self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for stale in files[self.max_keep :]:
            try:
                stale.unlink()
            except OSError:
                pass


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ── Recording / replay ────────────────────────────────────────────


def record_snapshot(store: ReplayStore, snapshot: CycleSnapshot) -> Path:
    """Persist a snapshot — failure logged, never raised to the cycle."""
    try:
        return store.write(snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay snapshot write failed (%s): %s", snapshot.cycle_id, exc)
        return Path()


async def replay_cycle(
    store: ReplayStore,
    cycle_id: str,
    decide: Callable[[CycleSnapshot], Awaitable[Any]],
) -> Any:
    """Load the snapshot, hand it to ``decide``, return the result.

    The harness is intentionally agnostic about *what* ``decide`` does —
    it can be:
      * a real strategy instance (re-runs the LLM call),
      * a stub that just inspects the inputs (debugging),
      * the adversarial-co-bot critic alone,
      * the prompt-evolution GA fitness function.
    """
    snap = store.read(cycle_id)
    return await decide(snap)


def diff_snapshots(a: CycleSnapshot, b: CycleSnapshot) -> dict[str, Any]:
    """Shallow diff of two snapshots — useful when reproducing bugs.

    Reports which top-level fields differ (and a count of klines that
    differ per symbol). Not a deep semantic diff; intentionally cheap.
    """
    diff: dict[str, Any] = {}
    a_d = asdict(a)
    b_d = asdict(b)
    for key in set(a_d) | set(b_d):
        if a_d.get(key) != b_d.get(key):
            if key == "klines_by_symbol":
                ksum: dict[str, int] = {}
                for sym in set(a.klines_by_symbol) | set(b.klines_by_symbol):
                    a_ks = a.klines_by_symbol.get(sym, [])
                    b_ks = b.klines_by_symbol.get(sym, [])
                    if a_ks != b_ks:
                        ksum[sym] = max(len(a_ks), len(b_ks))
                diff[key] = ksum
            else:
                diff[key] = {"a": a_d.get(key), "b": b_d.get(key)}
    return diff
