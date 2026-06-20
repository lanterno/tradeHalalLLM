"""daily_recommendations table

Revision ID: 6e30a000f579
Revises: 6c4e9bdf2810
Create Date: 2026-06-20 14:05:15.841431

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '6e30a000f579'
down_revision: Union[str, Sequence[str], None] = '6c4e9bdf2810'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — daily 'stock of the day' recommendations (advisory)."""
    op.create_table(
        "daily_recommendations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("date", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("symbol", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("conviction", sa.Float(), nullable=False),
        sa.Column("thesis", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("halal_note", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("suggested_entry", sa.Float(), nullable=True),
        sa.Column("suggested_target", sa.Float(), nullable=True),
        sa.Column("suggested_stop", sa.Float(), nullable=True),
        sa.Column("catalysts", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("risks", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("universe_size", sa.Integer(), nullable=False),
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("prompt_version", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("candidates", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_daily_recommendations_date"),
        "daily_recommendations",
        ["date"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_daily_recommendations_date"), table_name="daily_recommendations"
    )
    op.drop_table("daily_recommendations")
