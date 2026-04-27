"""add position monitor columns to crypto_trades

Revision ID: d3e4f5a6b7c8
Revises: c1a2b3d4e5f6
Create Date: 2026-03-07 02:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, None] = "c1a2b3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_column_if_not_exists(table: str, column: sa.Column) -> None:
    """SQLite-safe column addition (no IF NOT EXISTS support)."""
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    existing = {row[1] for row in result}
    if column.name not in existing:
        op.add_column(table, column)


def upgrade() -> None:
    _add_column_if_not_exists("crypto_trades", sa.Column("entry_price", sa.Float(), nullable=True))
    _add_column_if_not_exists("crypto_trades", sa.Column("stop_loss", sa.Float(), nullable=True))
    _add_column_if_not_exists("crypto_trades", sa.Column("target_price", sa.Float(), nullable=True))
    _add_column_if_not_exists("crypto_trades", sa.Column("exit_price", sa.Float(), nullable=True))
    _add_column_if_not_exists("crypto_trades", sa.Column("exit_reason", sa.String(), nullable=True))
    _add_column_if_not_exists("crypto_trades", sa.Column("closed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("crypto_trades", "closed_at")
    op.drop_column("crypto_trades", "exit_reason")
    op.drop_column("crypto_trades", "exit_price")
    op.drop_column("crypto_trades", "target_price")
    op.drop_column("crypto_trades", "stop_loss")
    op.drop_column("crypto_trades", "entry_price")
