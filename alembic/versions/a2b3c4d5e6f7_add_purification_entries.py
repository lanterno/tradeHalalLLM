"""add purification_entries table

Revision ID: a2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-04-26 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "a2b3c4d5e6f7"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": name},
    )
    return result.first() is not None


def upgrade() -> None:
    if _table_exists("purification_entries"):
        return
    op.create_table(
        "purification_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dividend_usd", sa.Float(), nullable=False),
        sa.Column("haram_pct", sa.Float(), nullable=False),
        sa.Column("purification_usd", sa.Float(), nullable=False),
        sa.Column("notes", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("paid_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_purification_entries_symbol", "purification_entries", ["symbol"])
    op.create_index("ix_purification_entries_paid_at", "purification_entries", ["paid_at"])


def downgrade() -> None:
    if _table_exists("purification_entries"):
        op.drop_index("ix_purification_entries_paid_at", table_name="purification_entries")
        op.drop_index("ix_purification_entries_symbol", table_name="purification_entries")
        op.drop_table("purification_entries")
