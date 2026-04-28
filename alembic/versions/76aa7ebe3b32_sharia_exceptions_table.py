"""sharia_exceptions table

Revision ID: 76aa7ebe3b32
Revises: 78944434afb4
Create Date: 2026-04-28 21:35:00.000000

Promotes the JSON sidecar (``data/analytics/sharia_exceptions.json``)
into a real Postgres table so the dashboard, CLI, and bot all read
the same operator-decided rulings without a file-locking race.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "76aa7ebe3b32"
down_revision: Union[str, Sequence[str], None] = "78944434afb4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sharia_exceptions",
        sa.Column("entry_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("instrument", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("reasoning", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("operator_note", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("entry_id"),
    )


def downgrade() -> None:
    op.drop_table("sharia_exceptions")
