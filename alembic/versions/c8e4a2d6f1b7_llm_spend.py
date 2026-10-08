"""llm_spend: LLM cost per UTC day and consumer, for the shared daily cap

Revision ID: c8e4a2d6f1b7
Revises: b5d2f8a1c9e3
Create Date: 2026-10-01 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8e4a2d6f1b7"
down_revision: str | Sequence[str] | None = "b5d2f8a1c9e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_spend",
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("consumer", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("calls", sa.Integer(), nullable=False),
        sa.Column("spent_usd", sa.Numeric(14, 6), nullable=False),
        sa.PrimaryKeyConstraint("day", "consumer"),
    )


def downgrade() -> None:
    op.drop_table("llm_spend")
