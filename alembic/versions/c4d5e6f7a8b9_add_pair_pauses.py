"""add pair_pauses table

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-04-26 20:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: Union[str, None] = "b3c4d5e6f7a8"
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
    if _table_exists("pair_pauses"):
        return
    op.create_table(
        "pair_pauses",
        sa.Column("pair", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("set_by", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("set_at", sa.DateTime(), nullable=False),
        sa.Column("reason", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("pair"),
    )


def downgrade() -> None:
    if _table_exists("pair_pauses"):
        op.drop_table("pair_pauses")
