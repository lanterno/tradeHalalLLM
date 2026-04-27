"""add halal_screenings table + halal_screening_id FK on trades

Revision ID: d9e0f1a2b3c4
Revises: c8d9e0f1a2b3
Create Date: 2026-04-26 11:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "d9e0f1a2b3c4"
down_revision: Union[str, None] = "c8d9e0f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": name},
    )
    return result.first() is not None


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    if not _table_exists("halal_screenings"):
        op.create_table(
            "halal_screenings",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("asset_class", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("source", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("decision", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("criteria", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_halal_screenings_symbol", "halal_screenings", ["symbol"])
        op.create_index("ix_halal_screenings_timestamp", "halal_screenings", ["timestamp"])

    # FK column on each trades table. Nullable for back-compat; a future
    # migration will tighten to NOT NULL once all callers populate it.
    for table in ("trades", "crypto_trades"):
        if not _column_exists(table, "halal_screening_id"):
            op.add_column(table, sa.Column("halal_screening_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    for table in ("crypto_trades", "trades"):
        if _column_exists(table, "halal_screening_id"):
            op.drop_column(table, "halal_screening_id")
    if _table_exists("halal_screenings"):
        op.drop_index("ix_halal_screenings_timestamp", table_name="halal_screenings")
        op.drop_index("ix_halal_screenings_symbol", table_name="halal_screenings")
        op.drop_table("halal_screenings")
