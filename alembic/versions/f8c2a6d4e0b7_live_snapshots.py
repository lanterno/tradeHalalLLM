"""account_snapshots, quotes: live account values and prices for the home page

Revision ID: f8c2a6d4e0b7
Revises: e6a2c4f8b0d3
Create Date: 2026-10-05 23:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f8c2a6d4e0b7"
down_revision: Union[str, Sequence[str], None] = "e6a2c4f8b0d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    # The newest value only: one row per account and per symbol, overwritten
    # each minute of the session. History lives in broker_equity/daily_bars.
    op.create_table(
        "account_snapshots",
        sa.Column("account", S(), primary_key=True),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("last_equity", sa.Float(), nullable=True),
        sa.Column("positions", JSONB, nullable=False),
    )
    op.create_table(
        "quotes",
        sa.Column("symbol", S(), primary_key=True),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("prev_close", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("quotes")
    op.drop_table("account_snapshots")
