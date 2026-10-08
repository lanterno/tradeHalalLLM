"""Drop the crypto tables.

Crypto trading was abandoned on 2026-10-01 and its code deleted; nothing has
read or written crypto_trades, crypto_daily_pnl or crypto_halal_cache since.
Downgrade recreates them empty.

Revision ID: 732583ae4034
Revises: b7d3f1a9c5e2
Create Date: 2026-10-08 05:53:43.098261

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "732583ae4034"
down_revision: str | Sequence[str] | None = "b7d3f1a9c5e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_table("crypto_daily_pnl")
    op.drop_table("crypto_trades")
    op.drop_table("crypto_halal_cache")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "crypto_halal_cache",
        sa.Column("symbol", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("compliance", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("category", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "market_cap", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column(
            "screening_criteria",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column(
            "updated_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.PrimaryKeyConstraint("symbol", name=op.f("crypto_halal_cache_pkey")),
    )
    op.create_table(
        "crypto_trades",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "timestamp", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("pair", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("side", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "quantity", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False
        ),
        sa.Column("price", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
        sa.Column("order_id", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("exchange", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("status", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("llm_reasoning", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "entry_price", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column(
            "stop_loss", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column(
            "target_price", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column(
            "exit_price", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column("exit_reason", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column(
            "closed_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True
        ),
        sa.Column(
            "submitted_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True
        ),
        sa.Column(
            "filled_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True
        ),
        sa.Column(
            "filled_price", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column(
            "filled_quantity", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column("halal_screening_id", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "paper_slippage_pct",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column(
            "live_slippage_pct",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column(
            "predicted_slippage_pct",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["halal_screening_id"],
            ["halal_screenings.id"],
            name=op.f("crypto_trades_halal_screening_id_fkey"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("crypto_trades_pkey")),
    )
    op.create_table(
        "crypto_daily_pnl",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("date", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column(
            "starting_equity",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "ending_equity", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column(
            "realized_pnl", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False
        ),
        sa.Column(
            "return_pct", sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True
        ),
        sa.Column("trades_count", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("crypto_daily_pnl_pkey")),
        sa.UniqueConstraint(
            "date",
            name=op.f("crypto_daily_pnl_date_key"),
            postgresql_include=[],
            postgresql_nulls_not_distinct=False,
        ),
    )
