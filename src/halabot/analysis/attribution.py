"""Outcome attribution (Direction C) — which evidence actually predicts wins.

Breaks realized performance down by regime and by evidence source (from the
entry-belief snapshot). This is the spec's "compounding edge": learn which kinds
of evidence and which regimes genuinely predict winners, so the calibrator and
interpreter weights can be steered toward them.

Closed ``hb_outcome`` alone is SURVIVORSHIP-BIASED — a "slow out" holds winners,
so at any snapshot the open positions skew toward winners and a closed-only view
under-counts them. By default we UNION the marked-to-market open positions
(``hb_open_position``) with a provisional label (unrealized > win threshold), for
an unbiased read. The calibrator is unaffected (it reads only closed
``hb_outcome``; open trades have no realized label). Read-only; entry-snapshot
features only (no mid-trade leakage)."""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from halabot.platform.db import open_position as _open_position
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
    total: int  # closed outcomes
    open_count: int  # open positions included (marked-to-market, provisional label)
    by_regime: list[Bucket]
    by_source: list[Bucket]


async def attribution(
    engine: AsyncEngine, *, min_n: int = 1, include_open: bool = True,
    win_threshold_pct: float = 0.002,
) -> Attribution:
    """Per-regime and per-source win-rate / avg-return over closed outcomes plus
    (by default) the marked-to-market open positions — the bias-corrected view.

    A trade contributes to a source's bucket if that source was present in its
    entry belief (so a trade counts toward every source that informed it)."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                sa.select(_outcome.c.return_pct, _outcome.c.label, _outcome.c.entry_belief)
            )
        ).all()
        open_rows = (
            (
                await conn.execute(
                    sa.select(
                        _open_position.c.unrealized_return_pct, _open_position.c.entry_belief
                    )
                )
            ).all()
            if include_open
            else []
        )

    by_regime: dict[str, list[tuple[float, int]]] = {}
    by_source: dict[str, list[tuple[float, int]]] = {}

    def _add(return_pct: float, label: int, entry: dict[str, object] | None) -> None:
        rec = (return_pct, label)
        eb = entry or {}
        by_regime.setdefault(str(eb.get("regime", "unknown")), []).append(rec)
        sources = eb.get("sources") or []
        if isinstance(sources, list):
            for src in sources:
                by_source.setdefault(str(src), []).append(rec)

    for return_pct, label, entry in rows:
        _add(float(return_pct), int(label), entry)
    for unrealized, entry in open_rows:  # provisional label from unrealized P&L
        _add(float(unrealized), 1 if float(unrealized) > win_threshold_pct else 0, entry)

    def _buckets(d: dict[str, list[tuple[float, int]]]) -> list[Bucket]:
        out = [_bucket(k, v) for k, v in d.items() if len(v) >= min_n]
        return sorted(out, key=lambda b: -b.avg_return_pct)

    return Attribution(
        total=len(rows), open_count=len(open_rows),
        by_regime=_buckets(by_regime), by_source=_buckets(by_source),
    )
