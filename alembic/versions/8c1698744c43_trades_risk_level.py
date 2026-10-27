"""trades.risk_level: the risk meter's level a buy was placed under (HAL-13).

Nullable: sells and the rows written before the meter carry none.

Revision ID: 8c1698744c43
Revises: e1d9ccc07ebd
Create Date: 2026-10-27 12:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c1698744c43"
down_revision: str | Sequence[str] | None = "e1d9ccc07ebd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trades", sa.Column("risk_level", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("trades", "risk_level")
