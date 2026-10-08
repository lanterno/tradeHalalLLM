"""etf_holdings: halal index ETFs' N-PORT holdings, dated by filing

Revision ID: b7d1f3a9c5e2
Revises: f4c8e2a6d0b5
Create Date: 2026-10-02 06:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d1f3a9c5e2"
down_revision: str | Sequence[str] | None = "f4c8e2a6d0b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "etf_holdings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("etf", S(), nullable=False),
        sa.Column("filed", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("ticker", S(), nullable=True),
        sa.Column("name", S(), nullable=False),
        sa.Column("cusip", S(), nullable=False),
        sa.Column("weight_pct", sa.Float(), nullable=False),
    )
    op.create_index("ix_etf_holdings_etf_filed", "etf_holdings", ["etf", "filed"])


def downgrade() -> None:
    op.drop_index("ix_etf_holdings_etf_filed", table_name="etf_holdings")
    op.drop_table("etf_holdings")
