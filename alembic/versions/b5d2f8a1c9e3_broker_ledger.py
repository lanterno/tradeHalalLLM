"""broker ledger: Alpaca account activities and daily equity

The broker's own record becomes the ledger of truth: broker_activities holds
every Alpaca account activity (fills, fees, dividends, transfers) keyed by
Alpaca's id; broker_equity holds the daily portfolio-history equity curve.

Revision ID: b5d2f8a1c9e3
Revises: a3c9e1f7b2d4
Create Date: 2026-10-01 17:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5d2f8a1c9e3"
down_revision: Union[str, Sequence[str], None] = "a3c9e1f7b2d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "broker_activities",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("activity_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("transaction_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("side", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("qty", sa.Float(), nullable=True),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("net_amount", sa.Float(), nullable=True),
        sa.Column("order_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_broker_activities_activity_type"), "broker_activities", ["activity_type"]
    )
    op.create_index(
        op.f("ix_broker_activities_transaction_time"), "broker_activities", ["transaction_time"]
    )
    op.create_index(op.f("ix_broker_activities_symbol"), "broker_activities", ["symbol"])
    op.create_table(
        "broker_equity",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("profit_loss", sa.Float(), nullable=False),
        sa.Column("profit_loss_pct", sa.Float(), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("day"),
    )


def downgrade() -> None:
    op.drop_table("broker_equity")
    op.drop_index(op.f("ix_broker_activities_symbol"), table_name="broker_activities")
    op.drop_index(op.f("ix_broker_activities_transaction_time"), table_name="broker_activities")
    op.drop_index(op.f("ix_broker_activities_activity_type"), table_name="broker_activities")
    op.drop_table("broker_activities")
