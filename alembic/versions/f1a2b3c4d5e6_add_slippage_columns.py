"""add paper / live slippage columns to trades and crypto_trades

Revision ID: f1a2b3c4d5e6
Revises: e0f1a2b3c4d5
Create Date: 2026-04-26 16:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "e0f1a2b3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_NEW_COLUMNS = [
    ("paper_slippage_pct", sa.Float()),
    ("live_slippage_pct", sa.Float()),
]


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    for table in ("trades", "crypto_trades"):
        for name, col_type in _NEW_COLUMNS:
            if not _column_exists(table, name):
                op.add_column(table, sa.Column(name, col_type, nullable=True))


def downgrade() -> None:
    for table in ("crypto_trades", "trades"):
        for name, _ in reversed(_NEW_COLUMNS):
            if _column_exists(table, name):
                op.drop_column(table, name)
