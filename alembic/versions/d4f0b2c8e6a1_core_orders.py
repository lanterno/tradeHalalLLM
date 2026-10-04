"""core_orders, core_runs: the core portfolio's orders and runs

Revision ID: d4f0b2c8e6a1
Revises: c2e8a4b6d0f3
Create Date: 2026-10-05 01:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4f0b2c8e6a1"
down_revision: Union[str, Sequence[str], None] = "c2e8a4b6d0f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "core_orders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", S(), nullable=False),
        sa.Column("side", S(), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("est_price", sa.Float(), nullable=False),
        sa.Column("notional", sa.Float(), nullable=False),
        sa.Column("reason", S(), nullable=False),
        sa.Column("screen_as_of", sa.Date(), nullable=True),
        sa.Column("status", S(), nullable=False),
        sa.Column("broker_order_id", S(), nullable=True),
        sa.Column("response", JSONB, nullable=True),
    )
    op.create_index("ix_core_orders_submitted", "core_orders", ["submitted_at"])
    op.create_table(
        "core_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_on", sa.Date(), nullable=False),
        sa.Column("monthly", sa.Boolean(), nullable=False),
        sa.Column("executed", sa.Boolean(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("orders", sa.Integer(), nullable=False),
        sa.Column("halted", S(), nullable=True),
        sa.Column("screen_as_of", sa.Date(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("core_runs")
    op.drop_index("ix_core_orders_submitted", table_name="core_orders")
    op.drop_table("core_orders")
