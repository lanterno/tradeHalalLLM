"""add runtime_config table

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-04-26 21:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, None] = "c4d5e6f7a8b9"
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
    if _table_exists("runtime_config"):
        return
    op.create_table(
        "runtime_config",
        sa.Column("key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("value", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("set_by", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("set_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    if _table_exists("runtime_config"):
        op.drop_table("runtime_config")
