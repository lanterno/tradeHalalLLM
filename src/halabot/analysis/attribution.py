"""Outcome attribution (Direction C) — which evidence actually predicts wins.

Reads closed ``hb_outcome`` rows and breaks realized performance down by regime
and by evidence source (from the entry-belief snapshot). This is the spec's
"compounding edge": learn which kinds of evidence and which regimes genuinely
predict winners, so the calibrator and interpreter weights can be steered toward
them. Read-only; entry-snapshot features only (no mid-trade leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.platform.db import outcome as _outcome


@dataclass(frozen=True)
class Bucket:
    key: str
    n: int
    win_rate: float
    avg_return_pct: float

    def line(self) -> str:
        return (
            f"{self.key:24s} n={self.n:4d}  win={self.win_rate:5.0%}  "
            f"avg={self.avg_return_pct:+.2%}"
        )


def _bucket(key: str, rows: list[tuple[float, int]]) -> Bucket:
    n = len(rows)
    wins = sum(1 for _, label in rows if label)
    avg = sum(r for r, _ in rows) / n if n else 0.0
    return Bucket(key=key, n=n, win_rate=(wins / n if n else 0.0), avg_return_pct=avg)


@dataclass
class Attribution:
    total: int
    by_regime: list[Bucket]
    by_source: list[Bucket]


async def attribution(engine: AsyncEngine, *, min_n: int = 1) -> Attribution:
    """Per-regime and per-source win-rate / avg-return over all closed outcomes.

    A trade contributes to a source's bucket if that source was present in its
    entry belief (so a trade counts toward every source that informed it)."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                sa.select(_outcome.c.return_pct, _outcome.c.label, _outcome.c.entry_belief)
            )
        ).all()

    by_regime: dict[str, list[tuple[float, int]]] = {}
    by_source: dict[str, list[tuple[float, int]]] = {}
    for return_pct, label, entry in rows:
        rec = (float(return_pct), int(label))
        eb = entry or {}
        regime = str(eb.get("regime", "unknown"))
        by_regime.setdefault(regime, []).append(rec)
        for src in eb.get("sources", []) or []:
            by_source.setdefault(str(src), []).append(rec)

    def _buckets(d: dict[str, list[tuple[float, int]]]) -> list[Bucket]:
        out = [_bucket(k, v) for k, v in d.items() if len(v) >= min_n]
        return sorted(out, key=lambda b: -b.avg_return_pct)

    return Attribution(
        total=len(rows), by_regime=_buckets(by_regime), by_source=_buckets(by_source)
    )
