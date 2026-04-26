"""add research_jobs table

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-04-26 22:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"),
        {"n": name},
    )
    return result.first() is not None


def upgrade() -> None:
    if _table_exists("research_jobs"):
        return
    op.create_table(
        "research_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("params", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "status", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="queued"
        ),
        sa.Column("result", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("error", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_jobs_timestamp", "research_jobs", ["timestamp"])
    op.create_index("ix_research_jobs_status", "research_jobs", ["status"])


def downgrade() -> None:
    if _table_exists("research_jobs"):
        op.drop_index("ix_research_jobs_status", table_name="research_jobs")
        op.drop_index("ix_research_jobs_timestamp", table_name="research_jobs")
        op.drop_table("research_jobs")
