"""round_trip_purification table

Revision ID: 3c8d1a4b9f02
Revises: 929702cb2112
Create Date: 2026-04-28 22:00:00.000000

Promotes the JSON sidecar (`data/analytics/round_trip_purification.json`)
into a real Postgres table so the dashboard can join on disbursement
state and the close hook doesn't fight file-locking with concurrent
multi-bot writes.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c8d1a4b9f02"
down_revision: Union[str, Sequence[str], None] = "929702cb2112"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "round_trip_purification",
        sa.Column("entry_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("gain_amount_usd", sa.Float(), nullable=False),
        sa.Column("impure_ratio", sa.Float(), nullable=False),
        sa.Column("purification_due_usd", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_ref", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("note", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("disbursed", sa.Boolean(), nullable=False),
        sa.Column("disbursed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disbursed_to", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("entry_id"),
    )
    op.create_index(
        op.f("ix_round_trip_purification_symbol"),
        "round_trip_purification",
        ["symbol"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_round_trip_purification_symbol"),
        table_name="round_trip_purification",
    )
    op.drop_table("round_trip_purification")
