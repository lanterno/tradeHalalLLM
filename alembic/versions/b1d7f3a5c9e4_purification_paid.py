"""purification_accruals.paid_at / paid_to: recording the donation

Revision ID: b1d7f3a5c9e4
Revises: a9c5e1f3b7d2
Create Date: 2026-10-04 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1d7f3a5c9e4"
down_revision: str | Sequence[str] | None = "a9c5e1f3b7d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "purification_accruals", sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "purification_accruals",
        sa.Column("paid_to", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("purification_accruals", "paid_to")
    op.drop_column("purification_accruals", "paid_at")
