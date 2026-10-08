"""zakat_assessments: each year's zakat by both methods, the higher chosen

Revision ID: c2e8a4b6d0f3
Revises: b1d7f3a5c9e4
Create Date: 2026-10-04 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c2e8a4b6d0f3"
down_revision: str | Sequence[str] | None = "b1d7f3a5c9e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "zakat_assessments",
        sa.Column("account", S(), nullable=False),
        sa.Column("hawl_date", sa.Date(), nullable=False),
        sa.Column("hawl_hijri", S(), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("market_value", sa.Float(), nullable=False),
        sa.Column("trade_goods_zakat", sa.Float(), nullable=False),
        sa.Column("dividends", sa.Float(), nullable=False),
        sa.Column("purified", sa.Float(), nullable=False),
        sa.Column("income_zakat", sa.Float(), nullable=False),
        sa.Column("chosen", S(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("holdings", JSONB, nullable=True),
        sa.Column("source", S(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("account", "hawl_date"),
    )


def downgrade() -> None:
    op.drop_table("zakat_assessments")
