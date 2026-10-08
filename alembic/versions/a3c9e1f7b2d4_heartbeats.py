"""heartbeats: last-seen time per long-running component

The stock bot and the dashboard run in separate containers; liveness has to
travel through the database. One row per component (stock.process,
stock.cycle, stock.monitor), upserted by the bot, read by /api/health/bot.

Revision ID: a3c9e1f7b2d4
Revises: d4b8f0a2c6e5
Create Date: 2026-10-01 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3c9e1f7b2d4"
down_revision: str | Sequence[str] | None = "d4b8f0a2c6e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "heartbeats",
        sa.Column("component", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("beat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("component"),
    )


def downgrade() -> None:
    op.drop_table("heartbeats")
