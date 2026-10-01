"""halal_screen_results: point-in-time in-house Shariah screening history

Revision ID: f9b1d7e3a5c2
Revises: e7a3c9b5d2f4
Create Date: 2026-10-01 18:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f9b1d7e3a5c2"
down_revision: Union[str, Sequence[str], None] = "e7a3c9b5d2f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

S = sqlmodel.sql.sqltypes.AutoString


def upgrade() -> None:
    op.create_table(
        "halal_screen_results",
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("symbol", S(), nullable=False),
        sa.Column("cik", sa.Integer(), nullable=True),
        sa.Column("sic_description", S(), nullable=False),
        sa.Column("verdict", S(), nullable=False),
        sa.Column("reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("method", S(), nullable=False),
        sa.Column("screened_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("as_of", "symbol"),
    )
    op.create_index(op.f("ix_halal_screen_results_verdict"), "halal_screen_results", ["verdict"])


def downgrade() -> None:
    op.drop_index(op.f("ix_halal_screen_results_verdict"), table_name="halal_screen_results")
    op.drop_table("halal_screen_results")
