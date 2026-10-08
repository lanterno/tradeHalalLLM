"""trades.high_water_price: persist the trailing stop's high-water mark

Revision ID: d2f6b8c4e1a9
Revises: c8e4a2d6f1b7
Create Date: 2026-10-01 18:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2f6b8c4e1a9"
down_revision: str | Sequence[str] | None = "c8e4a2d6f1b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trades", sa.Column("high_water_price", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("trades", "high_water_price")
