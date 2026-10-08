"""ticker_ciks: SEC CIKs for tickers SEC's current ticker file no longer lists

Revision ID: e2a6c0f4b8d1
Revises: c6e0a4b8d2f5
Create Date: 2026-10-02 04:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e2a6c0f4b8d1"
down_revision: str | Sequence[str] | None = "c6e0a4b8d2f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "ticker_ciks",
        sa.Column("symbol", S(), nullable=False),
        sa.Column("status", S(), nullable=False),
        sa.Column("cik", sa.Integer(), nullable=True),
        sa.Column("asset_name", S(), nullable=True),
        sa.Column("filer_name", S(), nullable=True),
        sa.Column("matched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )


def downgrade() -> None:
    op.drop_table("ticker_ciks")
