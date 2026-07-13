"""daily_recommendation path outcomes (realized high/low, MFE/MAE, level hits)

Phase 0 of docs/QUANT_PREDICTION_ROADMAP.md: the scorecard previously kept
only closes, so the LLM's suggested_target/suggested_stop were never scored.
These columns record the realized 5-day price path against the stated plan.

Revision ID: b3f1c7d2a9e4
Revises: 24fcae369aa0
Create Date: 2026-07-13 07:20:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3f1c7d2a9e4"
down_revision: Union[str, Sequence[str], None] = "24fcae369aa0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — 5-day path outcomes on daily_recommendations."""
    op.add_column(
        "daily_recommendations",
        sa.Column("realized_high_5d", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("realized_low_5d", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("mfe_pct", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("mae_pct", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("target_hit", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("stop_hit", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("first_hit", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("daily_recommendations", "first_hit")
    op.drop_column("daily_recommendations", "stop_hit")
    op.drop_column("daily_recommendations", "target_hit")
    op.drop_column("daily_recommendations", "mae_pct")
    op.drop_column("daily_recommendations", "mfe_pct")
    op.drop_column("daily_recommendations", "realized_low_5d")
    op.drop_column("daily_recommendations", "realized_high_5d")
