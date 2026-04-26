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
* :class:`RoundTripEntry` — one purification accrual.
* :class:`RoundTripLedger` — JSON-on-disk sidecar ledger.
* :func:`compute_round_trip_purification` — pure helper.
* :func:`record_round_trip` — idempotent (trade_id, symbol) recorder.

Persistence is JSON sidecar; no schema migration. When the operator
later wants to merge the dividend ledger and this one into one DB
table, the sidecar is the source of truth for backfill.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

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


# ── Ledger entry ─────────────────────────────────────────────────


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


@dataclass
class RoundTripLedger:
    """Append-only JSON-on-disk ledger of round-trip purification entries."""

    path: Path
    entries: dict[str, RoundTripEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("round-trip purification ledger unreadable: %s — starting fresh", exc)
            return
        entries = raw.get("entries", {})
        self.entries = {k: RoundTripEntry(**v) for k, v in entries.items() if isinstance(v, dict)}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"entries": {k: asdict(v) for k, v in self.entries.items()}},
                indent=2,
                sort_keys=True,
            )
        )

    def record(self, entry: RoundTripEntry) -> bool:
        if entry.entry_id in self.entries:
            return False
        self.entries[entry.entry_id] = entry
        self._save()
        return True

    def mark_disbursed(self, entry_id: str, *, to: str = "") -> bool:
        e = self.entries.get(entry_id)
        if e is None or e.disbursed:
            return False
        e.disbursed = True
        e.disbursed_at = datetime.now(UTC).isoformat()
        e.disbursed_to = to
        self._save()
        return True

    def outstanding(self) -> float:
        return sum(e.purification_due_usd for e in self.entries.values() if not e.disbursed)

    def disbursed_total(self) -> float:
        return sum(e.purification_due_usd for e in self.entries.values() if e.disbursed)

    def by_symbol(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries.values():
            if e.disbursed:
                continue
            out[e.symbol] = out.get(e.symbol, 0.0) + e.purification_due_usd
        return out


# ── Convenience ──────────────────────────────────────────────────


def record_round_trip(
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
    if ledger.record(entry):
        return entry
    return None


def outstanding_round_trip_due(ledger: RoundTripLedger) -> dict[str, float]:
    return {
        "total_usd": ledger.outstanding(),
        "by_symbol": ledger.by_symbol(),
        "disbursed_total_usd": ledger.disbursed_total(),
        "n_entries": len(ledger.entries),
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
