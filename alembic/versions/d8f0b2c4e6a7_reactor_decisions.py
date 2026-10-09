"""reactor_decisions: what the news reactor did with each scored catalyst.

The reactor's entries went to shadow on 2026-10-09 (its tests failed); each
decision, shadow or placed or declined, is kept so its rebuilt versions can be
measured on what it actually would have done.

Revision ID: d8f0b2c4e6a7
Revises: c6e8a0b2d4f5
Create Date: 2026-10-09 23:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8f0b2c4e6a7"
down_revision: str | Sequence[str] | None = "c6e8a0b2d4f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "reactor_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.VARCHAR(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("tag", sa.VARCHAR(), nullable=True),
        sa.Column("headline", sa.VARCHAR(), nullable=False),
        sa.Column("status", sa.VARCHAR(), nullable=False),
        sa.Column("reason", sa.VARCHAR(), nullable=True),
        sa.Column("placed", sa.Boolean(), nullable=False),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("intraday_change", sa.Float(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("stop_loss", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_reactor_decisions_decided_at", "reactor_decisions", ["decided_at"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_reactor_decisions_decided_at", table_name="reactor_decisions")
    op.drop_table("reactor_decisions")
