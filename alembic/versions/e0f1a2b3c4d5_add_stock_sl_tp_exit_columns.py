"""add SL/TP + close lifecycle columns to trades

Revision ID: e0f1a2b3c4d5
Revises: d9e0f1a2b3c4
Create Date: 2026-04-26 14:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "e0f1a2b3c4d5"
down_revision: Union[str, None] = "d9e0f1a2b3c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_NEW_COLUMNS = [
    ("stop_loss", sa.Float()),
    ("target_price", sa.Float()),
    ("exit_price", sa.Float()),
    ("exit_reason", sqlmodel.sql.sqltypes.AutoString()),
    ("closed_at", sa.DateTime()),
]


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    for name, col_type in _NEW_COLUMNS:
        if not _column_exists("trades", name):
            op.add_column("trades", sa.Column(name, col_type, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_NEW_COLUMNS):
        if _column_exists("trades", name):
            op.drop_column("trades", name)
