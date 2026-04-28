"""regime_snapshots table

Revision ID: 78944434afb4
Revises: ebe4fb61618e
Create Date: 2026-04-28 21:09:42.263275

Promotes the JSON sidecar (``data/analytics/regime_memory.json``)
into a real Postgres table. Features + embedding vector live in
JSON columns today; the pgvector(N) promotion is one alembic
migration away.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "78944434afb4"
down_revision: Union[str, Sequence[str], None] = "ebe4fb61618e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "regime_snapshots",
        sa.Column("date", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("features_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("vector_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("outcome_pnl_pct", sa.Float(), nullable=False),
        sa.Column("outcome_win_rate", sa.Float(), nullable=False),
        sa.Column("outcome_n_trades", sa.Integer(), nullable=False),
        sa.Column("note", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("date"),
    )


def downgrade() -> None:
    op.drop_table("regime_snapshots")
