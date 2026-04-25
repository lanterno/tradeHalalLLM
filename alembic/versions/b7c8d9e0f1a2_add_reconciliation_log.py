"""add reconciliation_log table

Revision ID: b7c8d9e0f1a2
Revises: a6b7c8d9e0f1
Create Date: 2026-04-25 17:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "a6b7c8d9e0f1"
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
    if _table_exists("reconciliation_log"):
        return
    op.create_table(
        "reconciliation_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("market", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("db_quantity", sa.Float(), nullable=False),
        sa.Column("broker_quantity", sa.Float(), nullable=False),
        sa.Column("drift_pct", sa.Float(), nullable=False),
        sa.Column("drift_usd", sa.Float(), nullable=True),
        sa.Column("notes", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reconciliation_log_timestamp", "reconciliation_log", ["timestamp"])


def downgrade() -> None:
    if _table_exists("reconciliation_log"):
        op.drop_index("ix_reconciliation_log_timestamp", table_name="reconciliation_log")
        op.drop_table("reconciliation_log")
