"""daily_recommendation outcome columns

Revision ID: 24fcae369aa0
Revises: 6e30a000f579
Create Date: 2026-06-21 05:46:10.827076

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '24fcae369aa0'
down_revision: Union[str, Sequence[str], None] = '6e30a000f579'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — outcome tracking on daily_recommendations."""
    op.add_column(
        "daily_recommendations",
        sa.Column(
            "outcome_status",
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("entry_close", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("fwd_return_1d", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("fwd_return_5d", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("fwd_return_20d", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("benchmark_return_5d", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("daily_recommendations", "benchmark_return_5d")
    op.drop_column("daily_recommendations", "fwd_return_20d")
    op.drop_column("daily_recommendations", "fwd_return_5d")
    op.drop_column("daily_recommendations", "fwd_return_1d")
    op.drop_column("daily_recommendations", "entry_close")
    op.drop_column("daily_recommendations", "scored_at")
    op.drop_column("daily_recommendations", "outcome_status")
