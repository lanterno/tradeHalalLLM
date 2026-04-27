"""add llm cost & cache columns to llm_decisions

Revision ID: c8d9e0f1a2b3
Revises: b7c8d9e0f1a2
Create Date: 2026-04-26 10:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "c8d9e0f1a2b3"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_NEW_COLUMNS = [
    ("prompt_version", sqlmodel.sql.sqltypes.AutoString()),
    ("input_tokens", sa.Integer()),
    ("output_tokens", sa.Integer()),
    ("cache_read_tokens", sa.Integer()),
    ("cache_write_tokens", sa.Integer()),
    ("cost_usd", sa.Float()),
]


def _column_exists(table: str, column: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    return any(row[1] == column for row in result)


def upgrade() -> None:
    for name, col_type in _NEW_COLUMNS:
        if not _column_exists("llm_decisions", name):
            op.add_column("llm_decisions", sa.Column(name, col_type, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_NEW_COLUMNS):
        if _column_exists("llm_decisions", name):
            op.drop_column("llm_decisions", name)
