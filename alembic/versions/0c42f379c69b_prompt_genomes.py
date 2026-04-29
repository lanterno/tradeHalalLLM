"""prompt_genomes

Revision ID: 0c42f379c69b
Revises: 8e2c41a6b54f
Create Date: 2026-04-29 05:44:11.332401

Wave F — table for the prompt-evolution GA. Each row is one
candidate prompt-slot mapping with its measured fitness over a
panel of replay snapshots. The dashboard "promote" button writes
ACTIVE_PROMPT_VERSION via runtime_config — this table just records
the lineage + scores.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes  # noqa: F401
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0c42f379c69b"
down_revision: Union[str, Sequence[str], None] = "8e2c41a6b54f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prompt_genomes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("genome", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("fitness", sa.Float(), nullable=False),
        sa.Column("n_cycles", sa.Integer(), nullable=False),
        sa.Column("parent_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_prompt_genomes_created_at"),
        "prompt_genomes",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_prompt_genomes_name"),
        "prompt_genomes",
        ["name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_prompt_genomes_name"), table_name="prompt_genomes")
    op.drop_index(op.f("ix_prompt_genomes_created_at"), table_name="prompt_genomes")
    op.drop_table("prompt_genomes")
