"""annual_fundamentals: gross profit and total assets per filer per calendar year

Revision ID: f4c8e2a6d0b5
Revises: e2a6c0f4b8d1
Create Date: 2026-10-02 05:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f4c8e2a6d0b5"
down_revision: str | Sequence[str] | None = "e2a6c0f4b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "annual_fundamentals",
        sa.Column("cik", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("gross_profit", sa.Float(), nullable=True),
        sa.Column("assets", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("cik", "year"),
    )


def downgrade() -> None:
    op.drop_table("annual_fundamentals")
