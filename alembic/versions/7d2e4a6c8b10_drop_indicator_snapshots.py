"""Drop indicator_snapshots.

Every day-trader buy wrote its indicator vector here for the ML retrainer,
deleted on 2026-10-01; nothing has read a row since. The indicators at any
entry can be recomputed from the stored bars and the trade's time. The
database was rebuilt on 2026-10-07 and holds none. Downgrade recreates it
empty.

Revision ID: 7d2e4a6c8b10
Revises: 3c5e7a9b1d24
Create Date: 2026-10-08 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7d2e4a6c8b10"
down_revision: str | Sequence[str] | None = "3c5e7a9b1d24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index(op.f("ix_indicator_snapshots_trade_id"), table_name="indicator_snapshots")
    op.drop_table("indicator_snapshots")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "indicator_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("trade_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.VARCHAR(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rsi_14", sa.Float(), nullable=True),
        sa.Column("macd_histogram", sa.Float(), nullable=True),
        sa.Column("volume_ratio", sa.Float(), nullable=True),
        sa.Column("atr_14", sa.Float(), nullable=True),
        sa.Column("bb_position", sa.Float(), nullable=True),
        sa.Column("price_change_5m", sa.Float(), nullable=True),
        sa.Column("ema_9", sa.Float(), nullable=True),
        sa.Column("ema_21", sa.Float(), nullable=True),
        sa.Column("vwap", sa.Float(), nullable=True),
        sa.Column("label", sa.Integer(), nullable=True),
        sa.Column("return_pct", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_indicator_snapshots_trade_id"), "indicator_snapshots", ["trade_id"], unique=False
    )
