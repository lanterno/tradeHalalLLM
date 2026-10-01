"""research data store: market_assets and daily_bars

Revision ID: e7a3c9b5d2f4
Revises: d2f6b8c4e1a9
Create Date: 2026-10-01 18:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7a3c9b5d2f4"
down_revision: Union[str, Sequence[str], None] = "d2f6b8c4e1a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "market_assets",
        sa.Column("symbol", S(), nullable=False),
        sa.Column("name", S(), nullable=False),
        sa.Column("exchange", S(), nullable=False),
        sa.Column("tradable", sa.Boolean(), nullable=False),
        sa.Column("fractionable", sa.Boolean(), nullable=False),
        sa.Column("status", S(), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )
    op.create_index(op.f("ix_market_assets_exchange"), "market_assets", ["exchange"])
    op.create_table(
        "daily_bars",
        sa.Column("symbol", S(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("adjustment", S(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("vwap", sa.Float(), nullable=True),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "day", "adjustment"),
    )


def downgrade() -> None:
    op.drop_table("daily_bars")
    op.drop_index(op.f("ix_market_assets_exchange"), table_name="market_assets")
    op.drop_table("market_assets")
