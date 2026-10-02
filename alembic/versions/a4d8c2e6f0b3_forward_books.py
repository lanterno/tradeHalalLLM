"""forward_books: strategies run forward on paper, one append-only row per session

Revision ID: a4d8c2e6f0b3
Revises: f9b1d7e3a5c2
Create Date: 2026-10-01 23:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a4d8c2e6f0b3"
down_revision: Union[str, Sequence[str], None] = "f9b1d7e3a5c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "forward_books",
        sa.Column("name", S(), nullable=False),
        sa.Column("strategy", S(), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "forward_book_days",
        sa.Column("book", S(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("nav", sa.Float(), nullable=False),
        sa.Column("day_return", sa.Float(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=False),
        sa.Column("weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rebalance_next", sa.Boolean(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["book"], ["forward_books.name"]),
        sa.PrimaryKeyConstraint("book", "day"),
    )


def downgrade() -> None:
    op.drop_table("forward_book_days")
    op.drop_table("forward_books")
