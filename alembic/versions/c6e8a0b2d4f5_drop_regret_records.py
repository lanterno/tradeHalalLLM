"""Drop regret_records.

Its writer (core/regret) was deleted on 2026-10-01. Its one reader, the
shadow-vs-live promotion gate, now takes the live leg from the day-trader's
closed round trips in ``trades``, so the gate can pass once both sides have
enough closed trades. Empty since the 2026-10-07 rebuild. Downgrade recreates
it empty.

Revision ID: c6e8a0b2d4f5
Revises: b4d6f8a0c2e3
Create Date: 2026-10-08 12:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6e8a0b2d4f5"
down_revision: str | Sequence[str] | None = "b4d6f8a0c2e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index(op.f("ix_regret_records_symbol"), table_name="regret_records")
    op.drop_table("regret_records")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "regret_records",
        sa.Column("trade_id", sa.VARCHAR(), nullable=False),
        sa.Column("symbol", sa.VARCHAR(), nullable=False),
        sa.Column("regret", sa.Float(), nullable=False),
        sa.Column("optimal_size_pct", sa.Float(), nullable=False),
        sa.Column("actual_size_pct", sa.Float(), nullable=False),
        sa.Column("pnl_pct", sa.Float(), nullable=False),
        sa.Column("note", sa.VARCHAR(), nullable=False),
        sa.Column("setup_type", sa.VARCHAR(), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("trade_id"),
    )
    op.create_index(op.f("ix_regret_records_symbol"), "regret_records", ["symbol"], unique=False)
