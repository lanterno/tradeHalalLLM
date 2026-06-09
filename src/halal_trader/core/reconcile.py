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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from halal_trader.core import events
from halal_trader.db.models import ReconciliationLog
from halal_trader.domain.status import TradeStatus

if TYPE_CHECKING:
    from halal_trader.notifications.telegram import AlertSink

logger = logging.getLogger(__name__)

DEFAULT_DRIFT_THRESHOLD = 0.01  # 1%

# Statuses that mean "no shares actually changed hands" — these Trade
# rows must not contribute to the DB-side position sum. Without this
# filter, a rejected order with quantity=380 (requested) and
# filled_quantity=0 was being counted as a 380-share phantom because
# the legacy ``or quantity`` fallback masked the zero fill.
_NON_EXECUTED_STATUSES: frozenset[str] = frozenset(
    {
        TradeStatus.REJECTED.value,
        TradeStatus.CANCELED.value,
        TradeStatus.ERROR.value,
        TradeStatus.PENDING.value,
        TradeStatus.SUBMITTED.value,
    }
)

# A stocks fill recorded within this window may not yet be visible on
# Alpaca's ``/v2/positions`` cache. When drift is observed on such a
# symbol we downgrade the alert to a settlement-race note instead of a
# warning — the next reconcile pass will confirm or escalate.
_STOCKS_SETTLEMENT_GRACE = timedelta(seconds=10)


@dataclass(frozen=True)
class Drift:
    market: str
    symbol: str
    db_quantity: float
    broker_quantity: float
    drift_pct: float
    drift_usd: float | None = None
    notes: str | None = None
    # True when the freshest fill on this symbol is within the broker's
    # settlement-propagation window. The drift is real on paper but
    # most likely a race; persist + log at INFO, skip the operator
    # alert. The next reconcile pass after the broker catches up will
    # either upgrade or clear it.
    is_settling: bool = False


@dataclass
class ReconcileReport:
    market: str
    drifts: list[Drift] = field(default_factory=list)
    checked_symbols: int = 0

    @property
    def has_drift(self) -> bool:
        return any(not d.is_settling for d in self.drifts)


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

    ``get_open_crypto_trades`` only filters out ``rejected``; we additionally
    skip ``pending`` / ``submitted`` / ``canceled`` / ``error`` rows so that
    an order which never actually settled doesn't carry its requested
    quantity into the DB sum (this was the same orphan path that produced
    100% phantom drift on the stocks side).
    """
    from halal_trader.db.repos import RepoBundle

    repos = RepoBundle.from_engine(engine)
    open_trades = await repos.crypto_trades.get_open_crypto_trades()

    db_by_asset: dict[str, float] = {}
    for trade in open_trades:
        if getattr(trade, "side", "") != "buy":
            continue
        status = str(getattr(trade, "status", "") or "").lower()
        if status in _NON_EXECUTED_STATUSES:
            continue
        base = trade.pair.upper().removesuffix("USDT").removesuffix("BUSD") if trade.pair else ""
        if not base:
            continue
        # Prefer filled_quantity (broker truth) over the requested
        # quantity. Fall back to quantity for legacy rows that predate
        # the fill-confirmer.
        filled = float(getattr(trade, "filled_quantity", 0) or 0)
        qty = filled if filled > 0 else float(getattr(trade, "quantity", 0) or 0)
        if qty <= 0:
            continue
        db_by_asset[base] = db_by_asset.get(base, 0.0) + qty

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


def _parse_iso(value: Any) -> datetime | None:
    """Best-effort ISO-string → tz-aware UTC datetime parser.

    Trade rows from ``model_dump()`` serialize ``filled_at`` as a string;
    rows from a direct ORM query keep it as ``datetime``. Accept either.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _aggregate_stocks_positions(
    rows: Iterable[dict[str, Any]],
    *,
    now: datetime,
    grace: timedelta,
) -> tuple[dict[str, float], set[str]]:
    """Fold Trade rows into (signed-quantity per symbol, symbols still settling).

    Skips rows whose ``status`` is non-executed (pending, rejected, …) —
    a row that never moved shares must not inflate the DB-side sum;
    this filter is the core fix for the historical orphan-counting bug.
    For executed rows, ``filled_quantity`` is preferred but the legacy
    ``quantity`` is accepted as a fallback (older fills predate the
    fill-confirmer that populates ``filled_quantity``). ``filled_at``
    newer than (now - grace) marks the symbol as "settling" so callers
    can downgrade the alert during the broker's propagation window.
    """
    db_by_symbol: dict[str, float] = {}
    settling: set[str] = set()
    for row in rows:
        symbol = (row.get("symbol") or "").upper()
        if not symbol:
            continue
        status = (row.get("status") or "").lower()
        # Drop non-executed rows entirely — they never moved shares.
        # Unknown/empty statuses default to "trust the row" so legacy
        # fixtures keep their semantics.
        if status in _NON_EXECUTED_STATUSES:
            continue
        filled_raw = row.get("filled_quantity")
        try:
            qty = float(filled_raw) if filled_raw is not None else 0.0
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            # Executed status but no fill column — fall back to the
            # requested quantity. Real fills set filled_quantity > 0;
            # this branch only matters for legacy rows / tests.
            qty_raw = row.get("quantity")
            try:
                qty = float(qty_raw) if qty_raw is not None else 0.0
            except (TypeError, ValueError):
                qty = 0.0
        if qty <= 0:
            continue
        side = (row.get("side") or "").lower()
        sign = 1 if side == "buy" else -1 if side == "sell" else 0
        if sign == 0:
            continue
        db_by_symbol[symbol] = db_by_symbol.get(symbol, 0.0) + sign * qty
        filled_at = _parse_iso(row.get("filled_at"))
        if filled_at is not None and (now - filled_at) <= grace:
            settling.add(symbol)
    return db_by_symbol, settling


async def reconcile_stocks(
    *,
    engine: AsyncEngine,
    broker: Any,
    threshold_pct: float = DEFAULT_DRIFT_THRESHOLD,
    alerts: "AlertSink | None" = None,
    settlement_grace: timedelta = _STOCKS_SETTLEMENT_GRACE,
) -> ReconcileReport:
    """Compare aggregate filled-quantity per ticker against broker positions.

    Sums signed ``filled_quantity`` (buys positive, sells negative) per
    symbol — never the requested ``quantity`` — and skips rows whose
    ``status`` indicates the order never transferred shares. A symbol
    with a fresh fill (within ``settlement_grace``) is flagged
    ``is_settling`` so the alert sink can drop the noise from Alpaca's
    REST-position cache lag.
    """
    from halal_trader.db.repos import RepoBundle

    repos = RepoBundle.from_engine(engine)
    recent = await repos.trades.get_recent_trades(limit=500)

    now = datetime.now(UTC)
    db_by_symbol, settling_symbols = _aggregate_stocks_positions(
        recent, now=now, grace=settlement_grace
    )

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
                    is_settling=symbol in settling_symbols,
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
    if not report.drifts:
        logger.debug(
            "Reconciliation clean: %s checked_symbols=%d",
            report.market,
            report.checked_symbols,
        )
        return

    rows: list[ReconciliationLog] = []
    untracked_count = 0
    untracked_usd = 0.0
    for drift in report.drifts:
        # Settlement-race drift: a fill landed within the propagation
        # window. The numbers are real but expected to resolve on the
        # next pass — log at INFO and skip persistence + alert so we
        # don't burn the operator's attention on every fresh fill.
        if drift.is_settling:
            logger.info(
                "Reconcile settling: %s/%s db=%.8f broker=%.8f drift=%.2f%% "
                "(fresh fill within grace window)",
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
                    "settling": True,
                },
            )
            continue

        # "Untracked broker balance" (bot has no record, exchange does)
        # is informational — the bot has no claim and nothing to reconcile.
        # Real drift on a tracked position (db_quantity > 0 with a mismatch)
        # is the one that needs operator attention.
        is_untracked_broker_balance = drift.db_quantity == 0.0 and drift.broker_quantity > 0.0
        if is_untracked_broker_balance:
            # Aggregate into a single summary line at the end of the pass
            # instead of one INFO log per asset (testnet has 50+ faucet
            # assets → 50 lines per reconcile pass → console spam).
            untracked_count += 1
            untracked_usd += drift.drift_usd or 0.0
            continue

        # Real drift — log per-symbol at WARNING (operator-actionable)
        # and persist to reconciliation_log.
        logger.warning(
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

    # Single summary line for untracked broker balances — replaces the
    # per-asset INFO spam (was 50+ lines per pass on testnet).
    if untracked_count:
        logger.info(
            "Reconcile: %d untracked broker balance(s) totaling ~$%.2f (informational)",
            untracked_count,
            untracked_usd,
            extra={
                "event": events.RECONCILE_DRIFT,
                "market": report.market,
                "untracked_count": untracked_count,
                "untracked_usd": untracked_usd,
            },
        )

    if rows:
        async with AsyncSession(engine) as session:
            session.add_all(rows)
            await session.commit()

    if alerts is not None:
        # Crypto: untracked broker balances are informational, not
        # operator-actionable — Binance testnet has ~50 faucet assets,
        # alerting on each one burns the rate limit. Stocks: any
        # broker position with no DB row IS actionable (the equities
        # universe is small enough that an untracked AAPL position is
        # signal, not noise). Settling drifts (fresh-fill race) never
        # alert regardless of market.
        if report.market == "crypto":
            actionable = [
                d
                for d in report.drifts
                if not d.is_settling
                and not (d.db_quantity == 0.0 and d.broker_quantity > 0.0)
            ]
        else:
            actionable = [d for d in report.drifts if not d.is_settling]
        if actionable:
            summary = _summarize_drifts(actionable)
            await alerts.notify(
                events.RECONCILE_DRIFT,
                summary,
                market=report.market,
                severity="warning",
            )


def _summarize_drifts(drifts: Iterable[Drift]) -> str:
    parts: list[str] = []
    for d in list(drifts)[:5]:
        usd = f" (~${d.drift_usd:.2f})" if d.drift_usd is not None else ""
        parts.append(
            f"{d.market}/{d.symbol}: db={d.db_quantity:g} broker={d.broker_quantity:g} "
            f"drift={d.drift_pct * 100:.1f}%{usd}"
        )
    return "Reconciliation drift detected:\n" + "\n".join(parts)


# ── One-time orphan-fix backfill ───────────────────────────────


@dataclass(frozen=True)
class OrphanFix:
    """One Trade row the backfill considered."""

    trade_id: int
    symbol: str
    side: str
    quantity: float
    order_id: str
    old_status: str
    new_status: str
    source: str  # "broker" | "no-order-id" | "skipped"
    notes: str | None = None


@dataclass
class OrphanFixReport:
    candidates: int = 0
    updated: int = 0
    fixes: list[OrphanFix] = field(default_factory=list)


async def fix_stocks_orphans(
    *,
    engine: AsyncEngine,
    broker: Any | None = None,
    min_age_minutes: int = 5,
    dry_run: bool = True,
) -> OrphanFixReport:
    """Find pending Trade rows that have no real fill and resolve them.

    A row qualifies as an orphan when ``status='pending'`` AND
    ``filled_quantity`` is 0 / NULL AND ``closed_at`` is unset AND the
    row is at least ``min_age_minutes`` old. For each candidate:

    * if ``order_id`` is non-empty and a broker is supplied, call
      ``get_order_by_id`` and adopt the broker's terminal status;
    * if ``order_id`` is empty (the order never made it to the
      exchange — e.g. yesterday's Pydantic-validation rejections),
      mark the row ``rejected`` directly.

    ``dry_run=True`` (the default) writes nothing — useful for the
    confirmation step in the CLI.
    """
    from sqlmodel import col, select

    from halal_trader.db.models import Trade

    report = OrphanFixReport()
    cutoff = datetime.now(UTC) - timedelta(minutes=min_age_minutes)
    async with AsyncSession(engine) as session:
        stmt = (
            select(Trade)
            .where(Trade.status == TradeStatus.PENDING.value)
            .where(col(Trade.closed_at).is_(None))
            .where(Trade.timestamp < cutoff)
            .order_by(col(Trade.timestamp).asc())
        )
        result = await session.execute(stmt)
        rows = list(result.scalars().all())

    for trade in rows:
        filled = float(trade.filled_quantity or 0.0)
        if filled > 0:
            # Not actually orphaned — the executor recorded a fill but
            # forgot to bump status (a separate bug if it ever happens);
            # don't touch it here.
            continue
        report.candidates += 1

        order_id = (trade.order_id or "").strip()
        new_status: str
        source: str
        notes: str | None = None

        if order_id and broker is not None:
            try:
                order = await broker.get_order_by_id(order_id)
            except Exception as exc:  # noqa: BLE001
                source = "broker-error"
                new_status = trade.status
                notes = f"broker get_order_by_id failed: {exc!r}"
            else:
                raw_status = ""
                filled_qty_broker = 0.0
                filled_avg = None
                if isinstance(order, dict):
                    raw_status = str(order.get("status", "")).lower()
                    try:
                        filled_qty_broker = float(order.get("filled_qty") or 0)
                    except (TypeError, ValueError):
                        filled_qty_broker = 0.0
                    fa_raw = order.get("filled_avg_price")
                    if isinstance(fa_raw, (int, float, str)) and fa_raw != "":
                        try:
                            filled_avg = float(fa_raw)
                        except (TypeError, ValueError):
                            filled_avg = None
                # Map Alpaca order statuses to TradeStatus values.
                if raw_status == "filled":
                    new_status = TradeStatus.FILLED.value
                elif raw_status == "partially_filled":
                    new_status = TradeStatus.PARTIALLY_FILLED.value
                elif raw_status in {"canceled", "cancelled", "expired"}:
                    new_status = TradeStatus.CANCELED.value
                elif raw_status == "rejected":
                    new_status = TradeStatus.REJECTED.value
                elif raw_status in {"new", "pending_new", "accepted"}:
                    # Still really pending — leave it alone, the
                    # executor's confirm loop will catch it next cycle.
                    new_status = trade.status
                    notes = (
                        "broker reports still open ("
                        f"{raw_status!r}); skipping"
                    )
                elif not raw_status:
                    # Broker had no record — treat as rejected so the
                    # row stops counting against drift.
                    new_status = TradeStatus.REJECTED.value
                    notes = "broker returned no status"
                else:
                    new_status = TradeStatus.REJECTED.value
                    notes = f"unknown broker status {raw_status!r}; treating as rejected"
                source = "broker"

                if new_status in {
                    TradeStatus.FILLED.value,
                    TradeStatus.PARTIALLY_FILLED.value,
                } and filled_qty_broker > 0:
                    # Update fill columns from broker truth.
                    trade.filled_quantity = filled_qty_broker
                    trade.filled_price = filled_avg
                    trade.filled_at = trade.filled_at or datetime.now(UTC)
        elif not order_id:
            # Order never made it past the bot's own request layer
            # (e.g. upstream MCP validation rejected it before
            # assigning an id). The reconciler must ignore it.
            new_status = TradeStatus.REJECTED.value
            source = "no-order-id"
            notes = "no broker order id ever assigned"
        else:
            # order_id present but no broker supplied — can't verify.
            new_status = trade.status
            source = "skipped"
            notes = "order_id present but no broker available to query"

        if new_status == trade.status:
            report.fixes.append(
                OrphanFix(
                    trade_id=trade.id or -1,
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=float(trade.quantity or 0),
                    order_id=order_id,
                    old_status=trade.status,
                    new_status=new_status,
                    source=source,
                    notes=notes,
                )
            )
            continue

        old_status = trade.status
        trade.status = new_status
        report.fixes.append(
            OrphanFix(
                trade_id=trade.id or -1,
                symbol=trade.symbol,
                side=trade.side,
                quantity=float(trade.quantity or 0),
                order_id=order_id,
                old_status=old_status,
                new_status=new_status,
                source=source,
                notes=notes,
            )
        )

        if not dry_run:
            async with AsyncSession(engine) as session:
                session.add(trade)
                await session.commit()
            report.updated += 1

    # ── Reverse orphan: broker holds a position the DB never recorded ──
    # The forward pass above resolves DB rows that have no real fill. The
    # reverse case is a position present on the broker (a fill that
    # landed after a crash, a pre-existing position) with no DB row at
    # all. Left alone it surfaces as permanent "position present on
    # broker with no recent trade row" drift and — worse — the monitor
    # never enforces SL/TP on it because it isn't tracked. Import it as a
    # filled BUY using the broker's ``avg_entry_price`` as cost basis so
    # it nets out in reconcile and gets risk-managed like any other.
    if broker is not None:
        await _import_broker_only_positions(engine, broker, report, dry_run)

    return report


async def _import_broker_only_positions(
    engine: AsyncEngine,
    broker: Any,
    report: OrphanFixReport,
    dry_run: bool,
) -> None:
    """Import broker positions the DB has no open row for (reverse orphan)."""
    from halal_trader.db.repos import RepoBundle

    repos = RepoBundle.from_engine(engine)
    try:
        positions = await broker.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        logger.debug("reverse-orphan import: get_all_positions failed: %s", exc)
        return

    recent = await repos.trades.get_recent_trades(limit=500)
    # grace=0: we only care whether the DB already tracks net-long shares,
    # not whether a fill is mid-settlement.
    db_net, _ = _aggregate_stocks_positions(
        recent, now=datetime.now(UTC), grace=timedelta(0)
    )

    for p in positions:
        sym = str(getattr(p, "symbol", "") or "").upper()
        broker_qty = float(getattr(p, "qty", 0) or 0)
        if not sym or broker_qty <= 0:
            continue
        # DB already nets long on this symbol → forward path / fine.
        if db_net.get(sym, 0.0) > 0:
            continue
        report.candidates += 1

        entry = float(getattr(p, "avg_entry_price", 0) or 0)
        if entry <= 0:
            entry = float(getattr(p, "current_price", 0) or 0)
        if entry <= 0:
            # No usable cost basis — don't store a $0-basis row (it would
            # poison P&L the same way the EOD $0 close did).
            report.fixes.append(
                OrphanFix(
                    trade_id=-1,
                    symbol=sym,
                    side="buy",
                    quantity=broker_qty,
                    order_id="",
                    old_status="(broker-only)",
                    new_status="(skipped)",
                    source="broker-import",
                    notes="broker reported no usable entry price",
                )
            )
            continue

        now = datetime.now(UTC)
        if not dry_run:
            await repos.trades.record_trade(
                sym,
                "buy",
                broker_qty,
                price=entry,
                order_id="",
                status=TradeStatus.FILLED.value,
                llm_reasoning="reverse-orphan import (broker held position, DB had no row)",
                submitted_at=now,
                filled_at=now,
                filled_price=entry,
                filled_quantity=broker_qty,
                entry_type="broker_import",
            )
            report.updated += 1
        report.fixes.append(
            OrphanFix(
                trade_id=-1,
                symbol=sym,
                side="buy",
                quantity=broker_qty,
                order_id="",
                old_status="(broker-only)",
                new_status=TradeStatus.FILLED.value,
                source="broker-import",
                notes=f"imported at entry={entry:.2f}",
            )
        )


# ── One-time DB→broker drift reconciliation ───────────────────


@dataclass(frozen=True)
class DriftFix:
    """One symbol's proposed/applied balancing adjustment."""

    symbol: str
    db_net: float
    broker_qty: float
    delta: float  # broker_qty - db_net (the balancing amount)
    side: str  # "buy" | "sell" — the adjustment side
    price: float
    applied: bool


@dataclass
class DriftFixReport:
    fixes: list[DriftFix] = field(default_factory=list)
    applied_count: int = 0


async def reconcile_db_to_broker(
    *,
    engine: AsyncEngine,
    broker: Any,
    threshold_shares: float = 0.5,
    dry_run: bool = True,
) -> DriftFixReport:
    """Bring each symbol's signed DB net into line with the broker's
    actual position via a single P&L-neutral balancing entry.

    For accumulated historical drift (e.g. the pre-fix EOD synthetic-SELL
    over-counts and zero-fill orphan BUYs), classifying which specific
    legacy row is bogus is unreliable — multiple bugs stack per symbol.
    The robust fix is to treat the broker as truth: for each symbol where
    ``broker_qty - db_net`` exceeds ``threshold_shares``, record one
    clearly-tagged adjustment (``entry_type='reconcile_adjustment'``,
    ``order_id='RECONCILE-ADJ'``) that nets the DB to the broker.

    The adjustment is recorded ``closed`` with ``exit_price == filled_price``
    so it is **P&L-neutral** and never appears as a managed open position.
    It uses the exact same recent-window aggregation the live reconciler
    uses, so after ``--apply`` the next reconcile pass reads clean.

    ``dry_run=True`` (default) proposes without writing.
    """
    from halal_trader.db.repos import RepoBundle

    repos = RepoBundle.from_engine(engine)
    recent = await repos.trades.get_recent_trades(limit=500)
    db_net, _ = _aggregate_stocks_positions(
        recent, now=datetime.now(UTC), grace=timedelta(0)
    )

    broker_net: dict[str, float] = {}
    broker_price: dict[str, float] = {}
    try:
        positions = await broker.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile_db_to_broker: get_all_positions failed: %s", exc)
        positions = []
    for p in positions:
        sym = str(getattr(p, "symbol", "") or "").upper()
        if not sym:
            continue
        broker_net[sym] = float(getattr(p, "qty", 0) or 0)
        broker_price[sym] = float(getattr(p, "avg_entry_price", 0) or 0) or float(
            getattr(p, "current_price", 0) or 0
        )

    report = DriftFixReport()
    for sym in sorted(set(db_net) | set(broker_net)):
        d = db_net.get(sym, 0.0)
        b = broker_net.get(sym, 0.0)
        delta = b - d
        if abs(delta) <= threshold_shares:
            continue
        side = "buy" if delta > 0 else "sell"
        price = broker_price.get(sym, 0.0)
        applied = False
        if not dry_run:
            await _insert_reconcile_adjustment(engine, sym, side, abs(delta), price)
            report.applied_count += 1
            applied = True
        report.fixes.append(
            DriftFix(
                symbol=sym,
                db_net=d,
                broker_qty=b,
                delta=delta,
                side=side,
                price=price,
                applied=applied,
            )
        )
    return report


async def _insert_reconcile_adjustment(
    engine: AsyncEngine, symbol: str, side: str, qty: float, price: float
) -> None:
    """Insert one closed, P&L-neutral balancing Trade row."""
    from halal_trader.db.models import Trade

    now = datetime.now(UTC)
    px = price if price > 0 else None
    row = Trade(
        symbol=symbol,
        side=side,
        quantity=qty,
        price=px,
        order_id="RECONCILE-ADJ",
        status=TradeStatus.FILLED.value,
        llm_reasoning=f"one-time DB->broker reconcile {now.date().isoformat()}",
        submitted_at=now,
        filled_at=now,
        filled_price=px,
        filled_quantity=qty,
        entry_type="reconcile_adjustment",
        exit_price=px,
        exit_reason="reconcile_adjustment",
        closed_at=now,
    )
    async with AsyncSession(engine) as session:
        session.add(row)
        await session.commit()


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
