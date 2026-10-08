"""Drop the legacy purification ledger.

``purification_entries`` held dividend purification typed in by hand through
an admin API no page called; ``purification_accruals`` (compliance/
purification.py) has been the dividend ledger since 2026-10-04, and the
round-trip ledger holds the capital-gains side. The database was rebuilt on
2026-10-07 with nothing in it. Downgrade recreates it empty.

Revision ID: 3c5e7a9b1d24
Revises: 0afe09d31725
Create Date: 2026-10-08 10:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c5e7a9b1d24"
down_revision: str | Sequence[str] | None = "0afe09d31725"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index(op.f("ix_purification_entries_symbol"), table_name="purification_entries")
    op.drop_table("purification_entries")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "purification_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.VARCHAR(), nullable=False),
        sa.Column("dividend_usd", sa.Float(), nullable=False),
        sa.Column("haram_pct", sa.Float(), nullable=False),
        sa.Column("purification_usd", sa.Float(), nullable=False),
        sa.Column("notes", sa.VARCHAR(), nullable=True),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_purification_entries_symbol"), "purification_entries", ["symbol"], unique=False
    )
