"""risk_meter: the operator's risk level for the day trader (HAL-13).

One row, like ``kill_switch``. ``level`` is the level last chosen; a raise
waits for the next trading day (``effective_from``), and until then
``previous_level`` stays in effect. No row means level 3, the configured
settings.

Revision ID: e1d9ccc07ebd
Revises: e3746e509609
Create Date: 2026-10-27 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1d9ccc07ebd"
down_revision: str | Sequence[str] | None = "e3746e509609"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "risk_meter",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("previous_level", sa.Integer(), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("reason", S(), nullable=True),
        sa.Column("set_by", S(), nullable=True),
        sa.Column("set_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("risk_meter")
