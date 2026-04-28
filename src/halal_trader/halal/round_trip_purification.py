"""Round-trip purification accounting (capital-gains flavour).

The existing :mod:`halal_trader.halal.purification` module covers
*dividend* purification — the income side. This module covers the
*capital-gains* side: when a halal-screened equity earns a small
fraction of revenue from non-compliant lines (under the AAOIFI 5%
threshold, say), the realised gain on a buy-then-sell round trip
includes a proportional impure component the holder is expected to
purify.

It uses a separate primitive set so neither module needs to know about
the other:

* :class:`RoundTripRule` — symbol → impure-revenue ratio.
* :class:`RoundTripEntry` — one purification accrual (in-memory shape).
* :class:`RoundTripLedger` — async DB-backed ledger over
  ``round_trip_purification``.
* :func:`compute_round_trip_purification` — pure helper.
* :func:`record_round_trip` — idempotent (trade_id, symbol) recorder.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import select

from halal_trader.db.models import RoundTripPurificationRow

logger = logging.getLogger(__name__)


# ── Rules ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoundTripRule:
    """One symbol's impure-revenue ratio (0..1)."""

    symbol: str
    impure_ratio: float
    source: str = "manual"
    note: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.impure_ratio <= 1.0):
            raise ValueError(
                f"impure_ratio must be in [0,1], got {self.impure_ratio!r} for {self.symbol}"
            )


# ── Ledger entry (in-memory) ─────────────────────────────────────


@dataclass
class RoundTripEntry:
    """One purification accrual on a closed round trip."""

    entry_id: str
    symbol: str
    gain_amount_usd: float
    impure_ratio: float
    purification_due_usd: float
    timestamp: str
    source_ref: str = ""
    note: str = ""
    disbursed: bool = False
    disbursed_at: str | None = None
    disbursed_to: str = ""


# ── Pure compute ─────────────────────────────────────────────────


def compute_round_trip_purification(
    *, gain_usd: float, impure_ratio: float, decimals: int = 2
) -> float:
    """Return ``round(gain * impure_ratio, decimals)``.

    Purification is owed only on positive realised gains. Losses and
    flats produce zero — that's both the rule and a hard floor here.
    """
    if gain_usd <= 0 or impure_ratio <= 0:
        return 0.0
    raw = Decimal(str(gain_usd)) * Decimal(str(impure_ratio))
    quant = Decimal(10) ** -decimals
    return float(raw.quantize(quant, rounding=ROUND_HALF_UP))


# ── Ledger ───────────────────────────────────────────────────────


def _row_to_entry(row: RoundTripPurificationRow) -> RoundTripEntry:
    return RoundTripEntry(
        entry_id=row.entry_id,
        symbol=row.symbol,
        gain_amount_usd=row.gain_amount_usd,
        impure_ratio=row.impure_ratio,
        purification_due_usd=row.purification_due_usd,
        timestamp=row.timestamp.isoformat() if row.timestamp else "",
        source_ref=row.source_ref,
        note=row.note,
        disbursed=row.disbursed,
        disbursed_at=row.disbursed_at.isoformat() if row.disbursed_at else None,
        disbursed_to=row.disbursed_to,
    )


def _parse_ts(raw: str) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


@dataclass
class RoundTripLedger:
    """Async DB-backed ledger of round-trip purification entries."""

    engine: AsyncEngine

    @property
    def _sm(self) -> "async_sessionmaker[Any]":
        return async_sessionmaker(self.engine, expire_on_commit=False)

    async def record(self, entry: RoundTripEntry) -> bool:
        """Idempotent on ``entry_id`` — returns True iff inserted."""
        async with self._sm() as s:
            existing = await s.get(RoundTripPurificationRow, entry.entry_id)
            if existing is not None:
                return False
            s.add(
                RoundTripPurificationRow(
                    entry_id=entry.entry_id,
                    symbol=entry.symbol,
                    gain_amount_usd=entry.gain_amount_usd,
                    impure_ratio=entry.impure_ratio,
                    purification_due_usd=entry.purification_due_usd,
                    timestamp=_parse_ts(entry.timestamp),
                    source_ref=entry.source_ref,
                    note=entry.note,
                    disbursed=entry.disbursed,
                    disbursed_at=(_parse_ts(entry.disbursed_at) if entry.disbursed_at else None),
                    disbursed_to=entry.disbursed_to,
                )
            )
            await s.commit()
            return True

    async def mark_disbursed(self, entry_id: str, *, to: str = "") -> bool:
        async with self._sm() as s:
            row = await s.get(RoundTripPurificationRow, entry_id)
            if row is None or row.disbursed:
                return False
            row.disbursed = True
            row.disbursed_at = datetime.now(UTC)
            row.disbursed_to = to
            s.add(row)
            await s.commit()
            return True

    async def outstanding(self) -> float:
        async with self._sm() as s:
            rows = (
                (
                    await s.execute(
                        select(RoundTripPurificationRow).where(
                            RoundTripPurificationRow.disbursed.is_(False)
                        )
                    )
                )
                .scalars()
                .all()
            )
            return sum(r.purification_due_usd for r in rows)

    async def disbursed_total(self) -> float:
        async with self._sm() as s:
            rows = (
                (
                    await s.execute(
                        select(RoundTripPurificationRow).where(
                            RoundTripPurificationRow.disbursed.is_(True)
                        )
                    )
                )
                .scalars()
                .all()
            )
            return sum(r.purification_due_usd for r in rows)

    async def by_symbol(self) -> dict[str, float]:
        async with self._sm() as s:
            rows = (
                (
                    await s.execute(
                        select(RoundTripPurificationRow).where(
                            RoundTripPurificationRow.disbursed.is_(False)
                        )
                    )
                )
                .scalars()
                .all()
            )
        out: dict[str, float] = {}
        for r in rows:
            out[r.symbol] = out.get(r.symbol, 0.0) + r.purification_due_usd
        return out

    async def all_entries(self) -> list[RoundTripEntry]:
        async with self._sm() as s:
            rows = (await s.execute(select(RoundTripPurificationRow))).scalars().all()
            return [_row_to_entry(r) for r in rows]

    async def count(self) -> int:
        from sqlalchemy import func

        async with self._sm() as s:
            r = await s.execute(select(func.count()).select_from(RoundTripPurificationRow))
            return int(r.scalar_one())


# ── Convenience ──────────────────────────────────────────────────


async def record_round_trip(
    ledger: RoundTripLedger,
    rules: Mapping[str, RoundTripRule],
    *,
    trade_id: str,
    symbol: str,
    gain_usd: float,
    timestamp: str | None = None,
    note: str = "",
) -> RoundTripEntry | None:
    """Compute + record purification for one closed round-trip trade.

    Returns the new entry, or ``None`` when no purification is due
    (no rule, zero ratio, or non-positive gain). Idempotent in
    ``(symbol, trade_id)`` so the cycle can call it freely.
    """
    rule = rules.get(symbol.upper()) or rules.get(symbol)
    if rule is None or rule.impure_ratio <= 0 or gain_usd <= 0:
        return None
    due = compute_round_trip_purification(gain_usd=gain_usd, impure_ratio=rule.impure_ratio)
    if due <= 0:
        return None
    entry = RoundTripEntry(
        entry_id=f"{symbol}:{trade_id}",
        symbol=symbol,
        gain_amount_usd=gain_usd,
        impure_ratio=rule.impure_ratio,
        purification_due_usd=due,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        source_ref=trade_id,
        note=note,
    )
    if await ledger.record(entry):
        return entry
    return None


async def outstanding_round_trip_due(ledger: RoundTripLedger) -> dict[str, Any]:
    return {
        "total_usd": await ledger.outstanding(),
        "by_symbol": await ledger.by_symbol(),
        "disbursed_total_usd": await ledger.disbursed_total(),
        "n_entries": await ledger.count(),
    }


def load_rules_from_dicts(rows: Iterable[Mapping]) -> dict[str, RoundTripRule]:
    out: dict[str, RoundTripRule] = {}
    for row in rows:
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        try:
            ratio = float(row.get("impure_ratio", 0))
        except (TypeError, ValueError) as _exc:  # noqa: F841 — keep parens, ruff format strips them otherwise
            continue
        out[sym] = RoundTripRule(
            symbol=sym,
            impure_ratio=ratio,
            source=str(row.get("source", "manual")),
            note=str(row.get("note", "")),
        )
    return out
