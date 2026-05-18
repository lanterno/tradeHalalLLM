"""predicted_slippage_pct on trades + crypto_trades

Revision ID: 552a7d6e5862
Revises: efe56a359835
Create Date: 2026-05-18 07:56:49.723065

Wave G: the replay-fitted slippage model in ``ml/slippage.py`` returns
a per-fill prediction; we stamp it onto every trade row at fill time
so the same model can be scored against realised live_slippage_pct
after the fact (calibration error → retrain cadence).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "552a7d6e5862"
down_revision: str | Sequence[str] | None = "efe56a359835"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("predicted_slippage_pct", sa.Float(), nullable=True),
    )
    op.add_column(
        "crypto_trades",
        sa.Column("predicted_slippage_pct", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("crypto_trades", "predicted_slippage_pct")
    op.drop_column("trades", "predicted_slippage_pct")
