"""monthly_bars: monthly SIP bars for every listed and delisted stock

Revision ID: c6e0a4b8d2f5
Revises: a4d8c2e6f0b3
Create Date: 2026-10-02 01:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6e0a4b8d2f5"
down_revision: Union[str, Sequence[str], None] = "a4d8c2e6f0b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "monthly_bars",
        sa.Column("symbol", S(), nullable=False),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("vwap", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "month"),
    )


def downgrade() -> None:
    op.drop_table("monthly_bars")
