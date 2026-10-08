"""minute_bars: SIP minute bars around events (S2 intraday entry study)

Revision ID: f7b3d9a1c5e8
Revises: e5a9c3f7b1d4
Create Date: 2026-10-02 11:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f7b3d9a1c5e8"
down_revision: str | Sequence[str] | None = "e5a9c3f7b1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "minute_bars",
        sa.Column("symbol", S(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("vwap", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "ts"),
    )


def downgrade() -> None:
    op.drop_table("minute_bars")
