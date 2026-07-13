"""daily_recommendation plan-anchored outcomes (entry@open bracket)

Phase 0 of docs/QUANT_PREDICTION_ROADMAP.md ("Anchor outcomes to the plan,
not the close"): the pick's stated plan buys at the next open and exits at
target/stop/time — these columns record that simulated outcome next to the
close-anchored forward returns.

Revision ID: d4b8f0a2c6e5
Revises: c7a2e9f4d1b3
Create Date: 2026-07-13 10:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4b8f0a2c6e5"
down_revision: Union[str, Sequence[str], None] = "c7a2e9f4d1b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — plan-anchored outcome columns."""
    op.add_column(
        "daily_recommendations",
        sa.Column("entry_open", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("plan_return_5d", sa.Float(), nullable=True),
    )
    op.add_column(
        "daily_recommendations",
        sa.Column("plan_exit", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("daily_recommendations", "plan_exit")
    op.drop_column("daily_recommendations", "plan_return_5d")
    op.drop_column("daily_recommendations", "entry_open")
