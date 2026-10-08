"""dividends, purification_accruals: the purification ledger

Revision ID: a9c5e1f3b7d2
Revises: f7b3d9a1c5e8
Create Date: 2026-10-04 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a9c5e1f3b7d2"
down_revision: str | Sequence[str] | None = "f7b3d9a1c5e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "dividends",
        sa.Column("symbol", S(), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("payable_date", sa.Date(), nullable=True),
        sa.Column("record_date", sa.Date(), nullable=True),
        sa.Column("rate", sa.Float(), nullable=False),
        sa.Column("special", sa.Boolean(), nullable=False),
        sa.Column("source_id", S(), nullable=False),
        sa.PrimaryKeyConstraint("source_id"),
    )
    op.create_index("ix_dividends_symbol_ex", "dividends", ["symbol", "ex_date"])
    op.create_table(
        "purification_accruals",
        sa.Column("account", S(), nullable=False),
        sa.Column("dividend_id", S(), nullable=False),
        sa.Column("symbol", S(), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("payable_date", sa.Date(), nullable=True),
        sa.Column("shares", sa.Float(), nullable=False),
        sa.Column("dividend", sa.Float(), nullable=False),
        sa.Column("impure_ratio", sa.Float(), nullable=False),
        sa.Column("screen_as_of", sa.Date(), nullable=True),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("method", S(), nullable=False),
        sa.Column("accrued_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("account", "dividend_id"),
    )


def downgrade() -> None:
    op.drop_table("purification_accruals")
    op.drop_index("ix_dividends_symbol_ex", table_name="dividends")
    op.drop_table("dividends")
