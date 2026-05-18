"""Per-trade halal compliance receipts.

For every trade that carries a ``halal_screening_id`` we can produce a
structured JSON receipt joining the trade row with its screening row.
Use cases:

* Compliance reporting (export to send to a scholar / auditor).
* Operator self-audit ("show me every trade in BTC last month, with the
  source that approved it").
* Backstop for the post-trade purification ledger landing in Phase 3.

The exporter intentionally returns plain dicts (not Pydantic models)
because the downstream consumers — JSON exports, CLI tables — don't
benefit from validation, and dicts let us add fields without a schema
migration. The shape is documented in :func:`build_receipt`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from halal_trader.db.models import CryptoTrade, HalalScreening, Trade


@dataclass(frozen=True)
class Receipt:
    """Wrapper so the CLI can pretty-print without re-parsing JSON."""

    payload: dict[str, Any]

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, default=str)


def _serialize_trade(trade: Trade | CryptoTrade) -> dict[str, Any]:
    data = trade.model_dump()
    for k, v in list(data.items()):
        if isinstance(v, datetime):
            data[k] = v.isoformat()
    return data


def _serialize_screening(screening: HalalScreening) -> dict[str, Any]:
    data = screening.model_dump()
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    if data.get("criteria"):
        try:
            data["criteria"] = json.loads(data["criteria"])
        except Exception:
            pass
    return data


def build_receipt(trade: Trade | CryptoTrade, screening: HalalScreening | None) -> Receipt:
    """Compose the trade + screening rows into a single audit receipt.

    ``screening`` may be ``None`` for legacy trades that pre-date the FK;
    the receipt records that gap explicitly so a downstream auditor can
    flag those rows for manual review rather than silently treating them
    as compliant.
    """
    asset_class = "crypto" if isinstance(trade, CryptoTrade) else "stock"
    payload: dict[str, Any] = {
        "asset_class": asset_class,
        "trade": _serialize_trade(trade),
        "screening": _serialize_screening(screening) if screening else None,
        "compliance_status": (screening.decision if screening else "unattested"),
    }
    return Receipt(payload=payload)


async def export_receipt(
    engine: AsyncEngine,
    *,
    trade_id: int,
    asset_class: str,
    sign: bool = False,
    data_dir: Any = None,
) -> Receipt | Any:
    """Fetch a trade by id (and its screening, if any) and build the receipt.

    Returns ``None`` when no trade with that id exists.

    Round-4 wave 2.A: when ``sign=True``, the receipt is signed with
    the operator's Ed25519 keypair (loaded / generated under
    ``data_dir``) and a :class:`halal.signing.SignedReceipt` is
    returned instead. The signature lets a scholar / auditor verify
    the receipt without trusting our codebase. Defaults to off so
    existing CLI / web callers see no behaviour change.
    """
    if asset_class not in ("stock", "crypto"):
        raise ValueError(f"asset_class must be 'stock' or 'crypto'; got {asset_class!r}")

    trade_model = CryptoTrade if asset_class == "crypto" else Trade
    async with AsyncSession(engine) as session:
        trade = await session.get(trade_model, trade_id)
        if trade is None:
            return None
        screening: HalalScreening | None = None
        if trade.halal_screening_id is not None:
            screening = await session.get(HalalScreening, trade.halal_screening_id)
    receipt = build_receipt(trade, screening)
    if sign:
        from pathlib import Path

        from halal_trader.halal.signing import get_or_create_signer

        if data_dir is None:
            raise ValueError("sign=True requires a data_dir for the operator's keypair")
        signer = get_or_create_signer(Path(data_dir))
        return signer.sign(receipt)
    return receipt


async def export_for_symbol(
    engine: AsyncEngine,
    *,
    symbol: str,
    asset_class: str,
    limit: int = 50,
) -> list[Receipt]:
    """Bulk receipts for the most recent ``limit`` trades on ``symbol``.

    Useful for "give me the audit trail for AAPL this quarter."
    """
    if asset_class not in ("stock", "crypto"):
        raise ValueError(f"asset_class must be 'stock' or 'crypto'; got {asset_class!r}")

    receipts: list[Receipt] = []
    async with AsyncSession(engine) as session:
        if asset_class == "crypto":
            stmt = (
                select(CryptoTrade)
                .where(CryptoTrade.pair == symbol)
                .order_by(CryptoTrade.id.desc())
                .limit(limit)
            )
        else:
            stmt = (
                select(Trade).where(Trade.symbol == symbol).order_by(Trade.id.desc()).limit(limit)
            )
        trades = (await session.exec(stmt)).all()
        screening_ids = [t.halal_screening_id for t in trades if t.halal_screening_id is not None]
        screenings: dict[int, HalalScreening] = {}
        if screening_ids:
            screen_rows = (
                await session.exec(
                    select(HalalScreening).where(HalalScreening.id.in_(screening_ids))
                )
            ).all()
            screenings = {s.id: s for s in screen_rows}

        for trade in trades:
            scr = (
                screenings.get(trade.halal_screening_id)
                if trade.halal_screening_id is not None
                else None
            )
            receipts.append(build_receipt(trade, scr))
    return receipts
