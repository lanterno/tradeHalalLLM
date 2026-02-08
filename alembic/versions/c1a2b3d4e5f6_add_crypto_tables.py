"""add crypto tables

Revision ID: c1a2b3d4e5f6
Revises: b9da4efd8872
Create Date: 2026-02-08 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision: str = "c1a2b3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "b9da4efd8872"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add crypto_trades, crypto_daily_pnl, and crypto_halal_cache tables."""
    op.create_table(
        "crypto_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("pair", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("side", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("order_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("exchange", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("llm_reasoning", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "crypto_daily_pnl",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("date", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("starting_equity", sa.Float(), nullable=False),
        sa.Column("ending_equity", sa.Float(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=False),
        sa.Column("return_pct", sa.Float(), nullable=True),
        sa.Column("trades_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )
    op.create_table(
        "crypto_halal_cache",
        sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("compliance", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("market_cap", sa.Float(), nullable=True),
        sa.Column("screening_criteria", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )


def downgrade() -> None:
    """Remove crypto tables."""
    op.drop_table("crypto_halal_cache")
    op.drop_table("crypto_daily_pnl")
    op.drop_table("crypto_trades")
