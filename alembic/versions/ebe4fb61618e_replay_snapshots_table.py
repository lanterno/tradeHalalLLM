"""replay_snapshots table

Revision ID: ebe4fb61618e
Revises: 3c8d1a4b9f02
Create Date: 2026-04-28 20:57:56.189937

Promotes the JSON sidecar (``data/replay/<cycle_id>.json``) into a
real Postgres table. The full ``CycleSnapshot`` lives in a JSONB
column — opaque to the schema, so the dataclass stays the source of
truth for the snapshot shape.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ebe4fb61618e"
down_revision: Union[str, Sequence[str], None] = "3c8d1a4b9f02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "replay_snapshots",
        sa.Column("cycle_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("cycle_id"),
    )
    op.create_index(
        op.f("ix_replay_snapshots_created_at"),
        "replay_snapshots",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_replay_snapshots_created_at"), table_name="replay_snapshots")
    op.drop_table("replay_snapshots")
