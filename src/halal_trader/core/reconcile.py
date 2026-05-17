"""DB-vs-broker reconciliation — surface drift between recorded trades and exchange state.

Runs periodically (every 5 min for crypto; after each cycle for stocks).
On drift > ``threshold_pct`` for any symbol it emits a structured
``reconcile.drift`` event, writes a :class:`ReconciliationLog` row, and
sends a rate-limited Telegram alert via :class:`AlertSink`.

The reconciler is **surface-only** — it never auto-heals. Operators are
expected to investigate the log row and either close the ghost trade in
the UI / via ``halt --close-all`` or accept the discrepancy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Iterable

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from halal_trader.core import events
from halal_trader.db.models import ReconciliationLog

if TYPE_CHECKING:
    from halal_trader.notifications.telegram import AlertSink

logger = logging.getLogger(__name__)

DEFAULT_DRIFT_THRESHOLD = 0.01  # 1%


@dataclass(frozen=True)
class Drift:
    market: str
    symbol: str
    db_quantity: float
    broker_quantity: float
    drift_pct: float
    drift_usd: float | None = None
    notes: str | None = None


@dataclass
class ReconcileReport:
    market: str
    drifts: list[Drift] = field(default_factory=list)
    checked_symbols: int = 0

    @property
    def has_drift(self) -> bool:
        return bool(self.drifts)


def _drift_pct(db_qty: float, broker_qty: float) -> float:
    """Symmetric drift: |db - broker| / max(|db|, |broker|, eps)."""
    delta = abs(db_qty - broker_qty)
    base = max(abs(db_qty), abs(broker_qty), 1e-9)
    return delta / base


# ── Crypto reconciler ──────────────────────────────────────────


async def reconcile_crypto(
    *,
    engine: AsyncEngine,
    broker: Any,
    threshold_pct: float = DEFAULT_DRIFT_THRESHOLD,
    alerts: "AlertSink | None" = None,
) -> ReconcileReport:
    """Compare aggregate open-trade quantity per asset against broker balances.

    Crypto positions on Binance are tracked as raw balances (no per-position
    rows). We sum open BUY trades by base asset (e.g. ``BTC``) and compare
    against the corresponding ``free + locked`` balance the exchange reports.
    """
    from halal_trader.db.repository import Repository

    repo = Repository(engine)
    open_trades = await repo.get_open_crypto_trades()

    db_by_asset: dict[str, float] = {}
    for trade in open_trades:
        if getattr(trade, "side", "") != "buy":
            continue
        base = trade.pair.upper().removesuffix("USDT").removesuffix("BUSD") if trade.pair else ""
        if not base:
            continue
        db_by_asset[base] = db_by_asset.get(base, 0.0) + float(trade.quantity or 0)

    balances = await broker.get_balances()
    broker_by_asset: dict[str, float] = {
        b.asset.upper(): float(b.free) + float(b.locked) for b in balances
    }

    report = ReconcileReport(market="crypto")
    seen: set[str] = set()
    for asset, db_qty in db_by_asset.items():
        seen.add(asset)
        report.checked_symbols += 1
        broker_qty = broker_by_asset.get(asset, 0.0)
        pct = _drift_pct(db_qty, broker_qty)
        if pct > threshold_pct:
            price = _safe_cached_price(broker, f"{asset}USDT")
            drift_usd = abs(db_qty - broker_qty) * price if price is not None else None
            drift = Drift(
                market="crypto",
                symbol=asset,
                db_quantity=db_qty,
                broker_quantity=broker_qty,
                drift_pct=pct,
                drift_usd=drift_usd,
            )
            report.drifts.append(drift)

    # Surface broker-side surplus too (asset present in balances but no DB row).
    for asset, broker_qty in broker_by_asset.items():
        if asset in seen or broker_qty <= 0:
            continue
        if asset == "USDT" or asset == "BUSD":
            continue
        report.checked_symbols += 1
        pct = _drift_pct(0.0, broker_qty)
        if pct > threshold_pct:
            price = _safe_cached_price(broker, f"{asset}USDT")
            drift_usd = broker_qty * price if price is not None else None
            if drift_usd is not None and drift_usd < 5.0:
                # Exchange dust below the dust threshold — too noisy to flag.
                continue
            report.drifts.append(
                Drift(
                    market="crypto",
                    symbol=asset,
                    db_quantity=0.0,
                    broker_quantity=broker_qty,
                    drift_pct=pct,
                    drift_usd=drift_usd,
                    notes="balance present on exchange with no open trade row",
                )
            )

    await _persist_and_alert(engine, report, alerts)
    return report


def _safe_cached_price(broker: Any, symbol: str) -> float | None:
    fn = getattr(broker, "get_cached_price", None)
    if not callable(fn):
        return None
    try:
        price = fn(symbol)
        return float(price) if price else None
    except Exception:
        return None


# ── Stock reconciler ───────────────────────────────────────────


async def reconcile_stocks(
    *,
    engine: AsyncEngine,
    broker: Any,
    threshold_pct: float = DEFAULT_DRIFT_THRESHOLD,
    alerts: "AlertSink | None" = None,
) -> ReconcileReport:
    """Compare aggregate open-trade quantity per ticker against broker positions.

    For stocks we sum signed quantities (buys positive, sells negative) per
    symbol and compare to ``Position.qty`` from Alpaca.
    """
    from halal_trader.db.repository import Repository

    repo = Repository(engine)
    recent = await repo.get_recent_trades(limit=500)

    db_by_symbol: dict[str, float] = {}
    for row in recent:
        symbol = (row.get("symbol") or "").upper()
        if not symbol:
            continue
        side = (row.get("side") or "").lower()
        qty = float(row.get("filled_quantity") or row.get("quantity") or 0)
        sign = 1 if side == "buy" else -1 if side == "sell" else 0
        db_by_symbol[symbol] = db_by_symbol.get(symbol, 0.0) + sign * qty

    positions = await broker.get_all_positions()
    broker_by_symbol: dict[str, float] = {p.symbol.upper(): float(p.qty) for p in positions}

    report = ReconcileReport(market="stocks")
    seen: set[str] = set()
    for symbol, db_qty in db_by_symbol.items():
        seen.add(symbol)
        report.checked_symbols += 1
        broker_qty = broker_by_symbol.get(symbol, 0.0)
        pct = _drift_pct(db_qty, broker_qty)
        if pct > threshold_pct:
            report.drifts.append(
                Drift(
                    market="stocks",
                    symbol=symbol,
                    db_quantity=db_qty,
                    broker_quantity=broker_qty,
                    drift_pct=pct,
                )
            )

    for symbol, broker_qty in broker_by_symbol.items():
        if symbol in seen or broker_qty == 0:
            continue
        report.checked_symbols += 1
        pct = _drift_pct(0.0, broker_qty)
        if pct > threshold_pct:
            report.drifts.append(
                Drift(
                    market="stocks",
                    symbol=symbol,
                    db_quantity=0.0,
                    broker_quantity=broker_qty,
                    drift_pct=pct,
                    notes="position present on broker with no recent trade row",
                )
            )

    await _persist_and_alert(engine, report, alerts)
    return report


# ── Persistence + alerting ────────────────────────────────────


async def _persist_and_alert(
    engine: AsyncEngine,
    report: ReconcileReport,
    alerts: "AlertSink | None",
) -> None:
    if not report.has_drift:
        logger.debug(
            "Reconciliation clean: %s checked_symbols=%d",
            report.market,
            report.checked_symbols,
        )
        return

    rows: list[ReconciliationLog] = []
    for drift in report.drifts:
        # "Untracked broker balance" (bot has no record, exchange does)
        # is informational — the bot has no claim and nothing to reconcile.
        # Real drift on a tracked position (db_quantity > 0 with a mismatch)
        # is the one that needs operator attention.
        is_untracked_broker_balance = drift.db_quantity == 0.0 and drift.broker_quantity > 0.0
        log_fn = logger.info if is_untracked_broker_balance else logger.warning
        log_fn(
            "Reconcile drift: %s/%s db=%.8f broker=%.8f drift=%.2f%%",
            drift.market,
            drift.symbol,
            drift.db_quantity,
            drift.broker_quantity,
            drift.drift_pct * 100,
            extra={
                "event": events.RECONCILE_DRIFT,
                "market": drift.market,
                "symbol": drift.symbol,
                "db_quantity": drift.db_quantity,
                "broker_quantity": drift.broker_quantity,
                "drift_pct": drift.drift_pct,
                "drift_usd": drift.drift_usd,
                "notes": drift.notes,
            },
        )
        # Skip persisting "untracked broker balance" rows — these fire
        # every reconcile cycle for dust / testnet faucet balances we
        # don't trade. With Binance testnet's ~50 pre-seeded assets,
        # this was 48k rows/day. The runtime log still captures them
        # at INFO if anyone needs to triage.
        if is_untracked_broker_balance:
            continue
        rows.append(
            ReconciliationLog(
                timestamp=datetime.now(UTC),
                market=drift.market,
                symbol=drift.symbol,
                db_quantity=drift.db_quantity,
                broker_quantity=drift.broker_quantity,
                drift_pct=drift.drift_pct,
                drift_usd=drift.drift_usd,
                notes=drift.notes,
            )
        )

    if rows:
        async with AsyncSession(engine) as session:
            session.add_all(rows)
            await session.commit()

    if alerts is not None:
        summary = _summarize_drifts(report.drifts)
        await alerts.notify(events.RECONCILE_DRIFT, summary)


def _summarize_drifts(drifts: Iterable[Drift]) -> str:
    parts: list[str] = []
    for d in list(drifts)[:5]:
        usd = f" (~${d.drift_usd:.2f})" if d.drift_usd is not None else ""
        parts.append(
            f"{d.market}/{d.symbol}: db={d.db_quantity:g} broker={d.broker_quantity:g} "
            f"drift={d.drift_pct * 100:.1f}%{usd}"
        )
    return "Reconciliation drift detected:\n" + "\n".join(parts)


# ── Repository helper ─────────────────────────────────────────


async def get_recent_logs(engine: AsyncEngine, *, limit: int = 25) -> list[dict[str, Any]]:
    """Return the most recent reconciliation log rows as dicts."""
    from sqlmodel import col, select

    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(ReconciliationLog).order_by(col(ReconciliationLog.timestamp).desc()).limit(limit)
        )
        return [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "market": row.market,
                "symbol": row.symbol,
                "db_quantity": row.db_quantity,
                "broker_quantity": row.broker_quantity,
                "drift_pct": row.drift_pct,
                "drift_usd": row.drift_usd,
                "notes": row.notes,
            }
            for row in result.scalars().all()
        ]
